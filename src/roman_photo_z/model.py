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
    def wave_to_freq(wavelengths: np.ndarray) -> np.ndarray:
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

    def total_flux(throughput: np.ndarray, a: float, b: float, sed: np.ndarray = None, z: float = None) -> float:
        """Calculates total flux through a filter with a given transmission profile and SED

        Parameters
        ----------
        throughput : numpy.ndarray
            Transmission profile for a particular filter

        a : float
            Lower limit of integration
        
        b : float
            Upper limit of integration

        sed : numpy.ndarray
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float
            Redshift
        
        Returns
        -------
        flux : float
            Flux through a filter

        flux_error : float
            Error of flux density

        Notes
        -----
        Formula in use here is from Eric Switzer
        Throughput and sed should have a shape of (2, n) and (2, m) respective where n is the number of transmission samples and m is the number of sed samples.
        """
        assert len(throughput.shape) == throughput.shape[0] == len(sed.shape) == sed.shape[0] == 2
        flux, flux_error = quad(lambda λ: np.interp(λ, *throughput) * np.interp(λ, sed[0]*(1+z), sed[1]/(1+z)), a, b, limit=1000)
        return flux, flux_error

    def photon_counts(throughput: np.ndarray, a: float, b: float, sed: np.ndarray = None, z: float = None):
        """Counts photons passed through filter over a specific bandpass

        throughput : numpy.ndarray
            Transmission profile for a particular filter

        a : float
            Lower limit of integration
        
        b : float
            Upper limit of integration

        sed : numpy.ndarray
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float
            Redshift
            
        Returns
        -------
        counts : float
            Photons counted through a filter
        """
        counts = quad(lambda v: np.interp(v, *throughput) * (3631 if sed is None else np.interp(v, sed[0]*(1+z), sed[1]/(1+z))) / (v * h.value), a, b, limit=1000)[0]
        return counts

    def magnitude(throughput: np.ndarray, a: float, b: float, sed: np.ndarray = None, z: float = None) -> float:
        """Calculates AB calibrated magnitude 

        Parameters
        ----------
        throughput : numpy.ndarray
            Transmission profile for a particular filter

        a : float
            Lower limit of integration
        
        b : float
            Upper limit of integration

        sed : numpy.ndarray
            Spectral energy distribution of a galaxy from a rest frame
        
        z : float
            Redshift
        
        Returns
        -------
        m : float
            AB magnitude
        """
        assert len(throughput.shape) == throughput.shape[0] == len(sed.shape) == sed.shape[0] == 2
        m = -2.5*np.log10(Model.counts(throughput, a, b, sed, z)[0]/Model.counts(throughput, a, b)[0])
        return m

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
        filters[filters.columns.drop("Wave")] /= np.pi * 1.2**2 # converts effective area to throughput
        filters["Wave"] = (filters["Wave"]*10000).astype(int) # μm to Å
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

    #Get the model fluxes
    f_mod=np.empty((nz,nt,nf))
    abfiles=[]

    for it in range(nt):
        for jf in range(nf):
            if filters[jf][-4:]=='.res': filtro=filters[jf][:-4]
            else: filtro=filters[jf]
            model=spectra[it]+'.'+filtro+'.AB'
            model_path = get_ab_file(model)
            abfiles.append(model)
            #Generate new ABflux files if not present
            # or if new_ab flag on
            if model not in os.listdir(ab_dir):
                #print spectra[it],filters[jf]
                print(f'     Generating {model} ....')
                ABflux(spectra[it],filtro,madau='no')
                #z_ab=arange(0.,zmax_ab,dz_ab) #zmax_ab and dz_ab are def. in bpz_tools
                # abflux=f_z_sed(spectra[it],filters[jf], z_ab,units='nu',madau=pars.d['MADAU'])
                # abflux=clip(abflux,0.,1e400)
                # buffer=join(['#',spectra[it],filters[jf], 'AB','\n'])
                #for i in range(len(z_ab)):
                #	 buffer=buffer+join([`z_ab[i]`,`abflux[i]`,'\n'])
                #open(model_path,'w').write(buffer)
                #zo=z_ab
                #f_mod_0=abflux
            #else:
                #Read the data

            zo,f_mod_0=get_data(model_path,(0,1))
        #Rebin the data to the required redshift resolution
            f_mod[:,it,jf]=match_resol(zo,f_mod_0,z_range)
            #if sometrue(less(f_mod[:,it,jf],0.)):
            if less(f_mod[:,it,jf],0.).any():
                print('Warning: some values of the model AB fluxes are <0')
                print('due to the interpolation ')
                print('Clipping them to f>=0 values')
                #To avoid rounding errors in the calculation of the likelihood
                f_mod[:,it,jf]=clip(f_mod[:,it,jf],0.,1e300)

                #We forbid f_mod to take values in the (0,1e-100) interval
                #f_mod[:,it,jf]=where(less(f_mod[:,it,jf],1e-100)*greater(f_mod[:,it,jf],0.),0.,f_mod[:,it,jf])
    with open("model.txt", "w") as f:
        print("Saving model")
        print(f_mod, file=f)

    f_obs=np.empty_like(f_mod)
    ef_obs=np.empty_like(f_mod)

    spectra = [np.loadtxt(os.path.join(sed_dir, sed_file)).T for sed_file in os.listdir(sed_dir) if "CWWSB4" not in sed_file]
    filter_data = Model.load_filters()
    filters = [(filter_data[["Wave", filter_id]].to_numpy().T, filter_data[filter_data[filter_id]>0]["Wave"].min(),
                                                               filter_data[filter_data[filter_id]>0]["Wave"].max()) for filter_id in filter_data.columns.drop("Wave")]
    for j, sed in enumerate(spectra):
        for i, z in enumerate(tqdm(z_range, desc=f"Redshifts for Galaxy Type #{j+1} of {len(spectra)}", disable=(not args.verbose), position=0)):
            for k, (throughput, a, b) in enumerate(filters):
                f_obs[i,j,k], ef_obs[i,j,k] = Model.total_flux(throughput, a, b, sed, z)
                
    with open("observed_old.txt", "w") as f:
        print("Saving simulated observations")
        print(f_obs, file=f)

    ratio = f_obs / f_mod
    weight = f_obs / ef_obs
    ratio = np.mean(ratio * weight, axis=[0, 1])

    with open("ratio.txt", "w") as f:
        print("Saving f_obs/f_model weight b y f_obs/ ratio")
        print(ratio, file=f)

if __name__ == "__main__":
    main()
