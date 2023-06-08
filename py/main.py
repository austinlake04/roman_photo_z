import numpy as np
import emcee
import corner

def main() -> None:
    """Main function

    Paramaters
    ----------
    None

    Returns
    -------
    None
    """
    print("Hello word!")

def MCMC(log_likelihood, ndim: int, labels: list[str], fname: str, nwalkers: int = 10, nsteps: int = 15000, nburn: int = 5000, verbose: bool = True):
    """Markov Chain Monte Carlo sampling for posterior estimation

    Parameters
    ----------
    log_likelihood
        Log-likelihood function on which to estimate posterior distribution
    ndim : int
        Nummber of dimensions of parameter space
    labels : list[str]
        Labels for parameter space
    fname : str
        Path/filename in which to save resulting corner plot of posterior distribution
    nwalkers : int
        Number of walkers per dimension
    nsteps : int
        Number of steps to take in MCMC run
    verbose : bool
        Determines verbosity of execution and logging
    
    Returns
    -------
    trace : numpy.ndarray[numpy.float64]
        Sample chain from MCMC run
    """
    if nwalkers < 10*ndim: # ensure proper coverage of posterior with high number of walkers
        nwalkers = 10*ndim
    
    if nstep - nburn < 10000: # ensure corner plot will have sufficient samples
        nstep = nburn + 10000
    
    initial_state = np.random.random(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
    sampler.run_mcmc(initial_state, nsteps, progress=verbose)

    # ensure sample chain convergence using Gelman-Rubin statistic
    for label, sample_set in zip(labels, sampler.chain[:, nburn:, :].T):
        global_avg = np.mean(sample_set)
        chain_avgs = np.mean(sample_set, axis=0)
        global_diff = chain_avgs-global_avg
        B = np.sum(np.square(global_diff, global_diff))*nburn/(nwalkers-1)
        W = np.sum(np.square(sample_set-chain_avgs))/nwalkers
        V = W*(nburn-1)/nburn + B/nburn
        R = np.sqrt(V/W)
        if R>1.1:
            if verbose:
                print(f"Convergence failed for {label}: increasing chain length by a factor of 2")
            return MCMC(log_likelihood, ndim, labels, fname, nwalkers, 2*nsteps, nburn)

    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
    if verbose:
        print("MCMC Result")
        for i in range(ndim):
            median = np.median(trace[i])
            sigma = abs(median-np.quantile(trace[i], .84))
            print(f"{labels[i]} = {median} ± {sigma}")
    
    return trace

if __name__ == "__main__":
    main()