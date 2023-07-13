import argparse
from astropy import units as u
from astropy.constants import c, h
import desc_bpz
import git
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import sys
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
from tqdm import tqdm
import logging
from scipy.integrate import quad
from roman_photo_z.io import save

parser = argparse.ArgumentParser(description='RST Filter Flux Computation.')
parser.add_argument("--zmin",
                    type=float, default=0.005, help="Minimum redshift for simulation.")
parser.add_argument("--zmax",
                    type=float, default=3.505, help="Maximum redshift for simulation.")
parser.add_argument("--deltaz",
                    type=float, default=0.01, help="Step size fors redshift range.")
parser.add_argument("-v", "--verbose",
                    action="store_true", help="Outputs text to the terminal.")
parser.add_argument("-p", "--plot",
                    action="store_true", help="Display plots.")
args = parser.parse_args()
if args.zmax < args.zmin or args.deltaz < 0 or args.zmin < 0 or args.deltaz > args.zmax-args.zmin:
    print("Invalid redshift parameters requested. Exiting...")
    sys.exit(1)

base_dir = os.path.abspath(git.Repo(".", search_parent_directories=True).working_tree_dir)
current_dir, _ = os.path.split(__file__)
output_dir = os.path.join(base_dir, "output")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler(f"integrated_fluxes_Z.log", mode="w")
file_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
if args.verbose:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

mpl.style.use("tableau-colorblind10")
plt.rcParams["axes.facecolor"]='white'
plt.rcParams["savefig.facecolor"]='white'
plt.rcParams["figure.facecolor"] = 'white'



def show(sed, sed_type: str, z: float, integrated_fluxes = None, display: bool = False, output_dir = output_dir) -> None:
    """Plots specific spectral energy distributions and filter transmission profiles

    Parameters
    ----------
    sed: numpy.ndarray[numpy.float64]
        Galactic spectral energy distribution (Flux vs. Wavelength)

    sed_type: str
        Galactic spectral energy distribution type

    z: float
        redshift value

    integrated_fluxes: list[numpy.ndarray[numpy.float64]]
        Integrated fluxes for each RST filter for the given SED

    display: bool
        Determines if plots are displayed

    output_dir: str
        Directory in which to save zip file

    Returns
    -------
    None
    """
    size = 10
    fig, ax = plt.subplots(figsize=(size, size))
    twin_ax = ax.twinx()

    colors = ["red", "brown", "orange", "yellow", "green", "blue", "indigo", "darkviolet"]

    wavelength, flux = sed
    valid_index = np.where(flux > 0)
    ax.plot(wavelength[valid_index]*(z+1), flux[valid_index]/(z+1), color="gray", label=f"{sed_type}, Z = {z}")
    ax.set_ylabel("Relative Spectral Flux of Galaxy (erg s-1 cm-2 Å-1)")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylim(top=np.max(flux.value))


    filters = pd.read_excel(os.path.join(current_dir, "Roman_effarea_20210614.xlsx"), skiprows=1)
    filters.columns = [column.strip() for column in filters.columns]
    filters.drop(columns=["SNPrism", "Grism_1stOrder", "Grism_0thOrder"], inplace=True)
    filters["Wave"] *= 10000 # micron to angstrom conversion
    filters[filters.columns.drop("Wave")] /= np.pi * 1.2**2 # effective area to throughput conversion
    for (filter_id, integrated_flux) in zip(filters.drop(columns=["Wave"]).columns, integrated_fluxes):
        valid_index = np.argwhere(filters[filter_id] > 0).flatten()
        twin_ax.plot(filters["Wave"][valid_index], filters[filter_id][valid_index], color=colors.pop(), label=f"{filter_id}: {integrated_flux.value:7.2E} (erg cm-2 s-1)")    
    twin_ax.set_ylabel("Filter Throughput")
    twin_ax.set_ylim(top=1)
    twin_ax.set_xlim(right=filters["Wave"].iloc[-1])
    ax.set_xlim(0, filters["Wave"].iloc[-1])
    twin_ax.set_xlim(right=filters["Wave"].iloc[-1])
    twin_ax.legend(fontsize=size)
    save_path = f"{sed_type}_Z={z}.png"
    fig.savefig(save_path, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close(fig)

def organize(z, output_dir = output_dir) -> None:
    """Organizes files in output directory

    Parameters
    ----------
    z: float
        Redshift outputs to target for organization

    output_dir: str
        Directory in which to save zip file

    Returns
    -------
    None
    """
    save_path = os.path.join(output_dir, f"Z={z}.zip")
    if os.path.isfile(save_path):
        os.system(f"rm -rf {save_path}")
    if os.path.isdir(save_path[:-4]):
        os.system(f"rm -rf {save_path[:-4]}")
    cmd = f"zip -m -D -q {save_path} *{z}.png *{z}.log"
    os.system(cmd)

def load_filters() -> pd.DataFrame:
    """Loads filter transmission profiles

    Parameters
    ----------
    None

    Returns
    -------
    filters: pandas.DataFrame
        Filter transmission profiles as a function of wavelength
    """
    global current_dir
    filters = pd.read_excel(os.path.join(current_dir, "data_files", "FILTER", "Roman_effarea_20210614.xlsx"), skiprows=1)
    filters.columns = [column.strip() for column in filters.columns]
    if filters["Wave"][0] == 0:
        filters.drop(0, inplace=True)
    filters.drop(columns=["SNPrism", "Grism_1stOrder", "Grism_0thOrder"], inplace=True)
    filters["Wave"] *= 10000 # micron to angstrom conversion
    filters[filters.columns.drop("Wave")] /= np.pi * 1.2**2 # effective area to throughput conversion
    return filters

def throughput(sed: SourceSpectrum, transmission_profile: SpectralElement):
    """

    Parameters
    ----------
    sed: SourceSpectrum
        Sectral energy distribution in units of nJy
    
    transmission_profile: SecptralElement
        Throughput as a function of wavelength

    filter_id: int
        RST filter ID (XXX)

    Returns
    -------
    integrated_flux: float
        Integrated flux with a given SED and filter transmission profile
    """
    # assuming sed is in units of nJy (LSST standard flux units)
    integrated_flux = Observation(sed, transmission_profile).integrate()
    return integrated_flux

def spectral(wavelengths: np.ndarray) -> np.ndarray:
    """Converts wavelengths of arbitrary units into frequencies in units Hertz (Hz)

    Parameters
    ----------
    wavelenths: stropy.unit.quantity.Quantity
        Wavelengths of arbitrary units

    Returns
    -------
    frequencies: astropy.unit.quantity.Quantity
        Frequencies in units of Hertz (Hz)
    
    """
    frequencies = (wavelengths).to(u.Hz, equivalencies=u.spectral())
    return frequencies

def magnitude(sed: SourceSpectrum, transmission_profile: SpectralElement) -> float:
    """Calculates AB calibrated magnitude 

    Parameters
    ----------
    sed: synphot.units.SourceSpectrum
        Spectral energy distribution

    transmission_profile: synphot.units.SpectralElement
        Transmission profile for a particular filter
    
    Returns
    -------
    m_AB: float
        AB calibrated magnitude
    """
    observed_flux = quad(lambda f: transmission_profile(f).value * sed(f).value / (h.value * f), 0, np.inf)[0]
    reference_flux = quad(lambda f: transmission_profile(f).value * 3631 / (h.value  * f), 0, np.inf)[0]
    if reference_flux:
        m_AB = 2.5*np.log10(observed_flux/reference_flux)
        return m_AB
    return np.nan

def main() -> None:
    """Main function

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global current_dir, output_dir, logger

    bpz_dir, _ = os.path.split(desc_bpz.__file__)
    sed_dir = os.path.join(bpz_dir, "data_files", "SED")
    assert(os.path.isdir(sed_dir))
    filter_dir = os.path.join(current_dir, "data_files", "FILTER")
    assert(os.path.isdir(filter_dir))

    filter_data = load_filters()
    filter_ids = filter_data.columns.drop("Wave")
    transmission_profiles = []
    frequencies = spectral(filter_data["Wave"].to_numpy()*u.AA)
    for filter_id in filter_ids:
        transmission_profiles.append(SpectralElement(Empirical1D, points=frequencies, lookup_table=filter_data[filter_id].to_numpy()))

    # filter_data = []
    # filter_ids = []
    # for filter_file in tqdm(sorted([f for f in os.listdir(filter_dir) if "RST" in f]),
    #                         desc="Loading filter transmission profiles:",
    #                         disable=(not args.verbose), position=0):
    #     filter_ids.append(int(filter_file[filter_file.find("_")+2:filter_file.find(".")]))
    #     filter_data.append(SpectralElement.from_file(os.path.join(filter_dir, filter_file), data_start=1))
    
    
    z_range = np.linspace(start=args.zmin, stop=args.zmax, num=int((args.zmax-args.zmin)/args.deltaz))
    database = {}
    for sed_file in tqdm([os.path.join(sed_dir, sed_file) for sed_file in os.listdir(sed_dir) if "CWWSB4" not in sed_file],
                        desc="Loading SED templates:",
                        disable=(not args.verbose), position=0):
        galaxy_type = sed_file.rsplit('_', 1)[0]
        # logger.info(f"Galaxy Type:  {galaxy_type}")
        # logger.info("Filter        Integrated Flux    Magnitude")
        # logger.info("              (erg s-1 cm-2)")
        # logger.info("------------------------------------------")
        wavelength, flux = np.loadtxt(os.path.join(sed_dir, sed_file)).T # Angstrom, nJy
        if wavelength[0] == 0:
            wavelength, flux = wavelength[1:], flux[1:]
        df = pd.DataFrame(columns=["Z", *filter_ids])
        for z in tqdm(z_range, desc=f"Redshifts for {galaxy_type} Type Galaxy"):
            sed = SourceSpectrum(Empirical1D, points=spectral(wavelength*u.AA), lookup_table=flux*1e-9*u.Jy, z=z, z_type="conserve_flux")
            # integrated_fluxes = ["XXXXXXX"]
            magnitudes = []
            for (filter_id, transmission_profile) in zip(filter_ids, transmission_profiles):
                # integrated_fluxes.append(throughput(sed, transmission_profile))
                # print(sed(np.inf))
                # print(transmission_profile(np.inf))

                magnitudes.append(magnitude(sed, transmission_profile))
                # logger.info(f"RST_{filter_id}      {integrated_fluxes[-1].value:7.2E}            {magnitude[-1].value:7.2E}")
            df.loc[len(df.index)] = [z, *magnitudes]

            # show(sed=(wavelength, flux), z=z, sed_type=sed_file.rsplit("_", 1)[0], integrated_fluxes=integrated_fluxes, display=args.plot)
        # logger.info(f"")

        database[galaxy_type] = df

        # organize(z=args.redshift)
    save(database)

    sys.exit(0)


if __name__ == "__main__":
    main()
