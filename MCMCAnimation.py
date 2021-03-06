# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:19:26 2022

@author: ryanw

The MCMC implementation was done with help from the following tutorial: 
https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
"""

# First let's set up our packages
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from scipy import integrate
import emcee      #this is the MCMC (Markov Chain Monte Carlo) package
import corner     #this package allows us to plot a distribution/contour plot (corner plot) for each of the dimensions of the fit
from multiprocessing import Pool     #this package allows multiprocessing, i.e. distributing workload over multiple CPU instances



def ezinv(z, om=0.3, ol=0.7, orad=0.0, w0=-1.0, wa=0.0):
    """This is a function returning the inverse of the integrand used in model(), 1/E(z)"""
    ok = 1 - om - ol - orad         #calculate the curvature parameter
    wDarkEnergy = 3 * (1 + w0 + wa * (1 - (1 / (1 + z))))       #this will be used in the power of the following equation
    inside = orad * (1 + z)**4 + om * (1 + z)**3 + ok * (1 + z)**2 + ol * (1 + z)**wDarkEnergy
    if inside <= 0:     #this disregards any complex numbers or division by zero errors
        return 1
    else:       #if the value is a suitable value, take the multiplicative inverse of the sqrt
        ez = np.sqrt(inside) 
        return 1 / ez
    
def Sk(xx, ok):
    """This is the curvature correction function"""
    if ok < 0.0:
        dk = np.sin(np.sqrt(-ok) * xx) / np.sqrt(-ok)
    elif ok > 0.0:
        dk = np.sinh(np.sqrt(ok) * xx) / np.sqrt(ok)
    else:
        dk = xx
    return dk

def dist_mod(zs, om=0.3, ol=0.7, orad=0.0, w0=-1.0, wa=0.0):
    """ Calculate the distance modulus, correcting for curvature"""
    # Set some constants
    c = 299792.458          # km/s (speed of light)
    H0kmsmpc = 70.          # Hubble constant in km/s/Mpc
    ok = 1.0 - om - ol - orad
    #the following code block is vectorized (probably?). Just in case of an issue, the original equation has been left here but commented out
    # xx = np.array([integrate.quad(ezinv, 0, z, args=(om, ol, orad, w0, wa))[0] for z in zs])
    intfun = lambda z: integrate.quad(ezinv, 0, z, args=(om, ol, orad, w0, wa))[0]      #calculates the integral of ezinv from 0 to z, given the parameters.
    vec_fun = np.vectorize(intfun)      #vectorizes the function
    xx = vec_fun(zs)        #calculates the integral for each redshift value in zs
    D = Sk(xx, ok)
    lum_dist = D * (1 + zs)         #calculate luminosity distance
    dist_mod = 5 * np.log10(lum_dist)       # Distance modulus
    # Add an arbitrary constant that's approximately the log of c on Hubble constant minus absolute magnitude of -19.5
    dist_mod = dist_mod + np.log(c / H0kmsmpc) - (-19.5)  # You can actually skip this step and it won't make a difference to our fitting
    return dist_mod

def read_data(model_name):
    """This function reads data files that are in a directory called "SNe Data". Don't change this! """
    d = np.genfromtxt('SNe Data/' + model_name+'.txt', delimiter=',')
    zs = d[:, 0]
    mu = d[:, 1]
    muerr=d[:, 2]
    return zs, mu, muerr

def model(params, zs=None, mu=None, muerr=None):
    """This is the model which is fit against the data (zs, mu, muerr) given some parameters."""
    if len(params) == 2:    #this applies if only a simple model (datasets00-3) is being done.
        orad, w0, wa = 0, -1, 0         #the first 5 datasets have a guaranteed value of each of these parameters
        om, ol = params
    else:       #this is for a full model fit
        om, ol, orad, w0, wa = params
    mu_model = dist_mod(zs, om, ol, orad, w0, wa)     # calculate the distance modulus vs redshift for that model 
    # Calculate mscript for each of the models, which is the thing that determines the vertical normalisation 
    mscr = np.sum((mu_model - mu) / muerr**2) / np.sum(1 / muerr**2) 
    mu_model_norm = mu_model - mscr         # Apply the vertical offset
    return mu_model_norm

def lnlike(params, zs=None, mu=None, muerr=None):
    """This function serves as the log-likelihood function for the MCMC implementation. """
    LnLike = -0.5 * np.sum((model(params, zs, mu, muerr) - mu)**2 / muerr**2)
    if np.isnan(LnLike):
        return 0
    return LnLike

def lnprior(params):
    """This function defines the priors for each parameter (the acceptable parameter space). 
    This is a part of the MCMC implementation
    emcee requires that if a parameter is within it's allowed parameter space, this function returns 0, otherwise it returns -np.inf
    The given priors in this function should not be changed, and were defined by Tamara Davis"""
    if len(params) == 2:    #the below priors are for datasets 00 to 3 only. Since they're simple models, they only have two parameters
        om, ol = params
        if (0 <= om <= 1) and (0 <= ol <= 1):
            return 0.0      
        else:
            return -np.inf
    else:       #the below parameters are for dataset 4-6, with all 5 parameters possibly changing. 
        om, ol, orad, w0, wa = params
        if (0 <= om <= 1) and (0 <= ol <= 1) and (0 <= orad <= 1) and (-2 <= w0 <= 0) and (-1.5 <= wa <= 0.5):
            return 0.0
        else:
            return -np.inf

def lnprob(params, zs=None, mu=None, muerr=None):
    """This is the log probability function for the MCMC implementation. It effectively gives the relative probability of some model
    giving the correct fit.
    If the parameters aren't within the allowed parameter space, emcee requires that this function returns -np.inf"""
    lp = lnprior(params)        #checks if the parameters are within the prior parameter space
    if (not np.isfinite(lp)) or np.isnan(lp):       #dismisses this model if the parameters don't fit the allowed space.
        return -np.inf
    return lp + lnlike(params, zs, mu, muerr)

def mcmc(p0, nwalkers, niter, ndim, lnprob, data):
    """This is the main MCMC function, which runs niter times for nwalkers walkers. p0 is the initial parameter guess."""
    with Pool() as pool:        #this sets up a pool of multiprocessors
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
        
        #MCMC is first optimised by letting some walkers move around the parameter space to suss it out. A "burn-in" if you will
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        
        #this starts the main MCMC run and can take a while based on the number of iterations
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

def plotter(sampler, zs=None, mu=None, muerr=None):
    """This function is largely redundant and isn't actually used in this program's current state. 
    It's function is to plot 100 different MCMC states and see how they fit the model. """
    plt.ion()
    plt.scatter(zs, mu, marker='.')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(zs, model(theta, zs, mu, muerr), color="r", alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel(r'Redshift ($z$)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()




def main():
    """This is the main program which is run."""
    
    plt.rcParams['font.family'] = 'Serif'       #sets the graph font to be in the Serif family (close to the LaTeX default font)
    
    dataset = "Data0"

    zs, mu, muerr = read_data(dataset)      #read the data and assign it to arrays
    sizeReduction = 4       #value of 1 corresponds to reading all data. Value of 2 for example, fits the data against 1 in every 2 points (~ 50% time reduction)
    #for the above, sizeReduction = 4 or 7 seems to be about the sweet spot
    #the below line creates reduced size arrays to fit the model to, allowing for faster compute time. 
    zsReduced, muReduced, muerrReduced = zs[range(0, len(zs), sizeReduction)], mu[range(0, len(mu), sizeReduction)], muerr[range(0, len(muerr), sizeReduction)]
    
    bins = 20       #this is the number of bins to be used in the corner plots
    
    nwalkers = 50       #this is how many 'walkers' will be used in the MCMC
    initial = [0.8, 0.9]        #an initial guess for the MCMC model
    ndim = len(initial)     #this is the number of dimensions to analyse over
    niter = 200       #how many iterations of MCMC to do
    data = (zsReduced, muReduced, muerrReduced)     #put all data into one tuple in order to parse it into functions
    p0 = [initial + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]  #add some scatter to the initial parameter guess
    
    sampler, pos, prob, state = mcmc(p0, nwalkers, niter, ndim, lnprob, data)   #initiate the mcmc
    samples = sampler.flatchain     #this obtains the final data
    extra_params = [0, -1, 0]      #since these are the simple datasets, we need to append the orad, w0, and wa values onto the end of our parameter array

    flat_samples = sampler.get_chain(flat=True)  #i get the data twice for some reason??

    best_params = flat_samples[np.argmax(sampler.flatlnprobability)]    #this obtains the parameters of the *best* fit out of all of the data
    best_params = np.append(best_params, extra_params)      #append the extra params due to the simple model
    
    maxlikeli = np.zeros(5)         #initialize the array of the maximum likelihood parameters
    oms = []; ols = []
    for state in flat_samples:
        oms.append(state[0]); ols.append(state[1])  #append each walker parameter state to a list
    oms, ols = np.array(oms), np.array(ols)     #turn those lists into arrays
    for i, parameter in enumerate([oms, ols]):  #makes a histogram of each parameter
        y, x, _ = plt.hist(parameter, bins=bins)    #y is the frequency of parameter=x
        maxlikeli[i] = x[y.argmax()]        #finds the maximum frequency and the x value associated with it (most likely parameter value)
    maxlikeli[2:] = [0, -1, 0]  #sub in the extra parameters
    
    chains = sampler.chain
    frames = len(chains[0, :, 0])
    print(frames)
    positions = chains[0, 0, :]
    labels = ["$\Omega_m$", "$\Omega_\Lambda$", "$\Omega_r$", "$w_0$", "$w_a$"]
        
    # # Create 4x4 sub plots
    # gs = gridspec.GridSpec(4, 4)
    
    # fig = plt.figure(figsize=(12, 10), dpi=80)
    
    # axTop = fig.add_subplot(gs[:2, :]) # top row
    # axLeft = fig.add_subplot(gs[3, 0]) # bottom left corner
    # axLeftTop = fig.add_subplot(gs[2, 0]) # just above bottom left corner
    # axLeftRight = fig.add_subplot(gs[3, 1]) #just to the right of bottom left corner
    # axRight = fig.add_subplot(gs[2:, 2:]) # bottom right corner
    
    # Create 6x9 sub plots
    gs = gridspec.GridSpec(6, 9)
    
    fig = plt.figure(figsize=(12, 10), dpi=80)
    
    axTop = fig.add_subplot(gs[:2, :]) # top row
    axLeftTop = fig.add_subplot(gs[2, :3]) # just above bottom left corner
    axLeft = fig.add_subplot(gs[3:, :3]) # bottom left corner
    axLeftRight = fig.add_subplot(gs[3:, 3:5]) #just to the right of bottom left corner
    axRight = fig.add_subplot(gs[2:, 5:]) # bottom right corner
    axright = axRight.twinx();
    
    mu_om00_ox00 = dist_mod(zs, om=0.0, ol=0.0)  # We're going to use this empty model as a benchmark to compare the others to
    
    frames = niter
    fps = 10

    def animate(i):
        
        axTop.clear(); axRight.clear(); axLeft.clear(); axLeftTop.clear(); axLeftRight.clear()
        axLeftRight.cla(); axLeftTop.cla()
        
        ## -- the below does the scatter plot of the walkers -- ##
        x = chains[:, i, 0]
        y = chains[:, i, 1]
        axRight.scatter(x, y, color='r', s=2)
        axRight.set_xlim(0, 1); axRight.set_ylim(0, 1)
        
        ## -- now to make the corner plot -- ##
        oms, ols = [], []
        for j in range(0, i+1):
            for k in range(0, nwalkers):
                oms.append(chains[k, j, 0])
                ols.append(chains[k, j, 1])
        oms, ols = np.array(oms), np.array(ols)
        if i > 0:
            corner.hist2d(oms, ols, bins, plot_contours=True, plot_density=True, ax=axLeft, color='#307CC0')
        else:
            corner.hist2d(oms, ols, bins, plot_contours=False, plot_density=True, ax=axLeft, color='#307CC0')
        axLeft.set_xlim(0, 1); axLeft.set_ylim(0, 1)
        
        axLeftTop.hist(oms, bins=bins, range=(0, 1), edgecolor='#307CC0', histtype='step')
        axLeftRight.hist(ols, bins=bins, range=(0, 1), edgecolor='#307CC0', histtype='step')
        
        leftsigma_om, mean_om, rightsigma_om = np.quantile(oms, [0.16, 0.5, 0.84])
        leftsigma_ol, mean_ol, rightsigma_ol = np.quantile(ols, [0.16, 0.5, 0.84])
        leftunc_om = mean_om - leftsigma_om; rightunc_om = rightsigma_om - mean_om
        leftunc_ol = mean_ol - leftsigma_ol; rightunc_ol = rightsigma_ol - mean_ol
        
        #plot mean and sigma value lines on the histograms
        axLeftTop.axvline(leftsigma_om, linestyle='--', color='#307CC0'); axLeftTop.axvline(rightsigma_om, linestyle='--', color='#307CC0');
        axLeftTop.axvline(mean_om, linestyle='-', color='r')
        axLeftRight.axvline(leftsigma_ol, linestyle='--', color='#307CC0'); axLeftRight.axvline(rightsigma_ol, linestyle='--', color='#307CC0');
        axLeftRight.axvline(mean_ol, linestyle='-', color='r')
        #plot mean value lines on contour plot
        axLeft.axvline(mean_om, linestyle='-', color='r'); axLeft.axhline(mean_ol, linestyle='-', color='r')
        axLeft.plot(mean_om, mean_ol, 's', color='r', alpha=0.5)
        
        #set titles, axis limits, and remove ticks/labels where necessary
        axLeftTop.set_xlim(0, 1); axLeftRight.set_xlim(0, 1)
        axLeftTop.set_title(f"$\Omega_m =  {mean_om:.2f}^{{+{rightunc_om:.2f}}}_{{-{leftunc_om:.2f}}}$") 
        axLeftRight.set_title(f'$\Omega_\Lambda = {mean_ol:.2f}^{{+{rightunc_ol:.2f}}}_{{-{leftunc_ol:.2f}}}$')
        axright.set_ylabel("$\Omega_\Lambda$")
        axRight.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelbottom=False, labelleft=False)
        axLeftTop.tick_params(axis='both', which='both', left=False, right=False, top=False, bottom=False, labelbottom=False, labelleft=False)
        axLeftRight.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelbottom=False, labelleft=False)
        
        
        
        ## -- the below does the current best model fit -- ##
        mean_params = np.append(np.array([mean_om, mean_ol]), extra_params)
        plot_params = mean_params
        mu_omModel_oxModel = model(plot_params, zs, mu, muerr)
        axTop.errorbar(zs, mu-mu_om00_ox00, yerr=muerr, fmt='.', elinewidth=0.7, markersize=8, alpha=0.5)   #this is the data with associated error
        axTop.plot(zs, mu_omModel_oxModel - mu_om00_ox00, '-', color='red', label=f'({plot_params[0]:.2f}, {plot_params[1]:.2f})')
        axTop.axhline(y=0.0, ls=':', color='black')    #horizontal dotted line for the empty universe
        axTop.set_xlim(0, 1.0)     
        axTop.legend(frameon=False)
        
        ## -- housekeeping -- ##
        #set labels
        axTop.set_xlabel('Redshift ($z$)'); axTop.set_ylabel('Apparent Magnitude [Normalised to (0,0)]')
        axRight.set_xlabel("$\Omega_m$"); #axRight.set_ylabel("$\Omega_\Lambda$")
        
        axLeft.set_xlabel("$\Omega_m$"); axLeft.set_ylabel("$\Omega_\Lambda$")
        axLeftRight.set_xlabel("$\Omega_\Lambda$")
        axRight.set_title("Position of Random Walkers")
        
        
        
        #calculate chi^2 and set title
        chi2 = np.sum((mu_omModel_oxModel - mu)**2 / muerr**2)
        chi2_reduced = chi2 / (len(mu) - 2)
        fig.suptitle(f"Iteration = {i}      $\chi^2_{{dof}}$ = {chi2_reduced:.2f}")
        print(i)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4, hspace=0.6)
        return fig,
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=int(1000/fps))

    
    plt.show()

    ani.save(f'Part Two Graphs/{dataset} MCMC Animation.gif', writer='pillow')


#for multiprocessing to work, the program must be run with the __name__ == ... method
if __name__ == '__main__':
    main()