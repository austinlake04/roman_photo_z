import argparse
from astropy.constants import c, h
import astropy.units as u
import desc_bpz
from desc_bpz.bpz_tools_py3 import *
from desc_bpz.will_tools_py3 import *
from desc_bpz.paths import *
import numpy as np
import os
from scipy.integrate import quad, nquad
from synphot import units, SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
import sys
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='RST Filter Flux Computation.')
parser.add_argument("--zmin",
                    type=float, default=0.005, help="Minimum redshift for simulation.")
parser.add_argument("--zmax",
                    type=float, default=3.505, help="Maximum redshift for simulation.")
parser.add_argument("--dz",
                    type=float, default=0.01, help="Step size fors redshift range.")
parser.add_argument("-v", "--verbose",
                    action="store_true", help="Outputs text to the terminal.")
parser.add_argument("-p", "--plot",
                    action="store_true", help="Display plots.")
args = parser.parse_args()
assert 0 < args.zmin < args.zmax, "Invalid redshift boundary conditions!"
assert 0 < args.dz < args.zmax-args.zmin, "Invalid redshift step size!"

class Model:
    def wave_to_freq(wavelengths: u.Quantity) -> u.Quantity:
        """Converts wavelengths of arbitrary units into frequencies in units Hertz (Hz)

        Parameters
        ----------
        wavelengths : astropy.units.Quantity
            Wavelengths of arbitrary units

        Returns
        -------
        frequencies : astropy.units.Quantity
            Frequencies in units of Hertz (Hz)
        """
        frequencies = (wavelengths).to(u.Hz, equivalencies=u.spectral())
        return frequencies

    def total_flux(transmission_profile: np.ndarray, lower_limit: float, upper_limit: float, spectra: np.ndarray, z: float) -> float:
        """Calculates total flux through a filter with a given transmission profile and SED

        Parameters
        ----------
        transmission_profile : numpy.ndarray
            Transmission profile for a particular filter

        lower_limit : float
            Lower limit of integration
        
        upper_limit : float
            Upper limit of integration

        spectra : numpy.ndarray
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float
            Redshift
        
        Returns
        -------
        flux : float
            Flux through a filter

        error : float
            Error of flux density

        Notes
        -----
        Formula in use here is from Eric Switzer
        Throughput and sed should have a shape of (2, n) and (2, m) respective where n is the number of transmission samples and m is the number of sed samples.
        """
        assert len(transmission_profile.shape) == transmission_profile.shape[0] == len(spectra.shape) == spectra.shape[0] == 2
        flux, error = quad(lambda λ: np.interp(λ, *transmission_profile) * np.interp(λ, spectra[0]*(1+z), spectra[1]/(1+z)), lower_limit, upper_limit, limit=1000)
        return flux, error

    def photon_count(transmission_profile: np.ndarray, lower_limit: float, upper_limit: float, spectra: np.ndarray|None = None, z: float|None = None):
        """Counts photons passed through filter over a specific bandpass

        transmission_profile : numpy.ndarray
            Transmission profile for a particular filter

        lower_limit : float
            Lower limit of integration
        
        upper_limit : float
            Upper limit of integration

        spectra : numpy.ndarray|None
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float|None
            Redshift
            
        Returns
        -------
        photons : float
            Photons counted through a filter

        error : float
            Error of photon count
        """
        photons, error = quad(lambda v: np.interp(v, *transmission_profile) * (3631 if spectra or z is None else np.interp(v, spectra[0]*(1+z), spectra[1]/(1+z))) / (v * h.value), lower_limit, upper_limit, limit=1000)
        return photons, error

    def magnitude(transmission_profile: np.ndarray, lower_limit: float, upper_limit: float, spectra: np.ndarray, z: float) -> float:
        """Calculates AB calibrated magnitude 

        Parameters
        ----------
        transmission_profile : numpy.ndarray
            Transmission profile for a particular filter

        lower_limit : float
            Lower limit of integration
        
        upper_limit : float
            Upper limit of integration

        sed : numpy.ndarray
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float
            Redshift
        
        Returns
        -------
        magnitude : float
            AB magnitude
        """
        assert len(transmission_profile.shape) == transmission_profile.shape[0] == len(spectra.shape) == spectra.shape[0] == 2
        obs = Model.photon_count(transmission_profile, lower_limit, upper_limit, spectra, z)
        mod = Model.photon_count(transmission_profile, lower_limit, upper_limit)
        magnitude = -2.5*np.log10(obs[0]/mod[0])
        error = -2.5*np.log10(obs[1]/mod[1])
        return magnitude, error

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
        filters = pd.read_excel("https://roman.gsfc.nasa.gov/science/RRI/Roman_effarea_20210614.xlsx", skiprows=1)
        filters.columns = [column.strip() for column in filters.columns]
        filters.drop(columns=["SNPrism", "Grism_1stOrder", "Grism_0thOrder"], inplace=True)
        filters[filters.columns.drop("Wave")] /= np.pi * 1.2**2 # Filter response conversion: effective area [m^2] --> throughput [unitless]
        filters["Wave"] = (filters["Wave"]*10000).astype(int) # Bandpass conversion: μm --> Å
        return filters

def main():
    """BPZ Model Algorithm

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    data_dir = os.path.join(os.path.split(__file__)[0], "data_files")
    sed_dir = os.path.join(data_dir, "SED")
    ab_dir = os.path.join(data_dir, "AB")
    fil_dir = os.path.join(data_dir, "FILTER")
    set_data_dir(data_dir)
    filters = [filter_file[:-4] for filter_file in os.listdir(fil_dir) if "CWWSB4" not in filter_file]
    spectra = [sed_file[:-4] for sed_file in os.listdir(sed_dir) if "CWWSB4" not in sed_file]

    z_range = np.linspace(start=args.zmin, stop=args.zmax+args.dz, num=int((args.zmax-args.zmin)/args.dz))

    nf=len(filters)
    nt=len(spectra)
    nz=len(z_range)

    f_mod=np.empty((nz,nt,nf))
    abfiles=[]

    for it in range(nt):
        for jf in range(nf):
            if filters[jf][-4:]=='.res': filtro=filters[jf][:-4]
            else: filtro=filters[jf]
            model=spectra[it]+'.'+filtro+'.AB'
            model_path = get_ab_file(model)
            abfiles.append(model)
            if model not in os.listdir(ab_dir):
                print(f'     Generating {model} ....')
                ABflux(spectra[it],filtro,madau='no')
            zo,f_mod_0=get_data(model_path,(0,1))
            f_mod[:,it,jf]=match_resol(zo,f_mod_0,z_range)
            if less(f_mod[:,it,jf],0.).any():
                print('Warning: some values of the model AB fluxes are <0')
                print('due to the interpolation ')
                print('Clipping them to f>=0 values')
                f_mod[:,it,jf]=clip(f_mod[:,it,jf],0.,1e300)

    f_obs=np.empty_like(f_mod)
    ef_obs=np.empty_like(f_mod)

    seds = [np.loadtxt(os.path.join(sed_dir, sed_file)).T for sed_file in os.listdir(sed_dir) if "CWWSB4" not in sed_file] # SED units: flux [ergs/s/cm^2/Å] vs wavelength [Å]
    filters = Model.load_filters() # Filter units: throughput [unitless] vs wavelength [Å]
    transmission_profiles = [(filters[["Wave", filter_id]].to_numpy().T, filters[filters[filter_id]>0]["Wave"].min(),
                                                               filters[filters[filter_id]>0]["Wave"].max()) for filter_id in filters.columns.drop("Wave")]
    # for j, sed in enumerate(sed):
    #     for i, z in enumerate(tqdm(z_range, desc=f"Redshifts for SED Template #{j+1} of {len(spectra)}", disable=(not args.verbose), position=0)):
    #         for k, (transmission_profile, lower_limit, upper_limit) in enumerate(transmission_profiles):
    #             f_obs[i,j,k], ef_obs[i,j,k] = Model.magnitude(transmission_profile, lower_limit, upper_limit, sed, z)
    
    for j, spectra in enumerate(seds):
        for i, z in enumerate(tqdm(z_range, desc=f"Redshifts for SED Template #{j+1} of {len(spectra)}", disable=(not args.verbose), position=0)):
            for k, (transmission_profile, lower_limit, upper_limit) in enumerate(transmission_profiles):
                f_obs[i,j,k], ef_obs[i,j,k] = Model.total_flux(transmission_profile, lower_limit, upper_limit, spectra, z)         


    ratio = f_obs / f_mod
    weight = f_obs / ef_obs
    ratio = np.sum(ratio * weight, axis=(0, 1), dtype=np.float64)/np.sum(weight, axis=(0, 1), dtype=np.float64)

    print(ratio)

    np.save("model.npy", f_mod)
    np.save("observed.npy", f_obs)
    np.save("observed_error.npy", ef_obs)
    np.save("ratio.npy", ratio)

if __name__ == "__main__":
    main()
