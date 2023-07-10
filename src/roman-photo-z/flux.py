import os
import sys
import git
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
from astropy import units as u
from astropy.constants import c
import argparse
import logging
import desc_bpz

parser = argparse.ArgumentParser(description='RST Filter Flux Computation.')
parser.add_argument("-z", "--redshift",
                    type=float, default=0, help="Determines observational redshift.")
parser.add_argument("-v", "--verbose",
                    action="store_true", help="Outputs text to the terminal.")
parser.add_argument("-p", "--plot",
                    action="store_true", help="Display plots.")
args = parser.parse_args()
if args.redshift < 0:
    print("Requested redshift observation is less than 0. Please enter a positive value. Exiting...")
    sys.exit(1)

base_dir = os.path.abspath(git.Repo(".", search_parent_directories=True).working_tree_dir)
current_dir, _ = os.path.split(__file__)
output_dir = os.path.join(base_dir, "output")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

level = logging.INFO if args.verbose else logging.DEBUG
format   = "%(message)s"
handlers = [logging.FileHandler(f"integrated_fluxes_Z={args.redshift}.log", mode="w"), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

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

    filters = pd.read_csv(os.path.join(current_dir, "Roman_effarea_20210614.csv")).drop(columns=["SNPrism","Grism_1stOrder","Grism_0thOrder"])
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
        print("Removing directory")
        os.system(f"rm -rf {save_path[:-4]}")
    cmd = f"zip -m -D {save_path} *{z}.png *{z}.log"
    os.system(cmd)

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

def main() -> None:
    """Main function

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global args, current_dir, output_dir
    
    bpz_dir, _ = os.path.split(desc_bpz.__file__)
    sed_dir = os.path.join(bpz_dir, "data_files", "SED")
    assert(os.path.isdir(sed_dir))
    filter_dir = os.path.join(current_dir, "data_files", "FILTER")
    assert(os.path.isdir(filter_dir))

    filter_data = []
    filter_ids = []
    for filter_file in sorted([f for f in os.listdir(filter_dir) if "RST" in f]):
        filter_ids.append(int(filter_file[filter_file.find("_")+2:filter_file.find(".")]))
        filter_data.append(SpectralElement.from_file(os.path.join(filter_dir, filter_file), data_start=1))

    sed_files = [f for f in os.listdir(sed_dir) if "CWWSB4" not in f]
    for sed_file in sed_files:
        logging.info(f"Galaxy Type:  {sed_file.rsplit('_', 1)[0]}")
        logging.info("Filter        Integrated Flux")
        logging.info("              (erg s-1 cm-2)")
        logging.info("-----------------------------")
        wavelength, flux = np.loadtxt(os.path.join(sed_dir, sed_file)).T # Angstrom, nJy
        if wavelength[0] == 0:
            wavelength, flux = wavelength[1:], flux[1:]
        flux = units.convert_flux(wavelength*u.AA, flux*1e-9*u.Jy, out_flux_unit=units.FLAM)
        sed = SourceSpectrum(Empirical1D, points=wavelength, lookup_table=flux, z=args.redshift, z_type="conserve_flux")
        integrated_fluxes = []
        for (filter_id, transmission_profile) in zip(filter_ids, filter_data):
            integrated_flux = throughput(sed, transmission_profile)
            logging.info(f"RST_F{filter_id:03d}      {integrated_flux.value:7.2E}")
            integrated_fluxes.append(integrated_flux)
        
        show(sed=(wavelength, flux), z=args.redshift, sed_type=sed_file.rsplit("_", 1)[0], integrated_fluxes=integrated_fluxes, display=args.plot)
        logging.info(f"")

    organize(z=args.redshift)
    sys.exit(0)


if __name__ == "__main__":
    main()
