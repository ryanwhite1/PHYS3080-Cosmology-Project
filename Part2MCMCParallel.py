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
    
    runall = "Yes"      #change this if you want to run just one dataset
    
    if runall == "Yes":
        datasets = ["Data00", "Data0", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7"]       #all datasets
    else:
        datasets = ["Data4", "Data5", "Data6", "Data7"]        #change this to the dataset(s) you want to run
    
    
    for dataset in datasets:
        print(dataset)      #print which dataset we're currently analysing

        zs, mu, muerr = read_data(dataset)      #read the data and assign it to arrays
        sizeReduction = 4       #value of 1 corresponds to reading all data. Value of 2 for example, fits the data against 1 in every 2 points (~ 50% time reduction)
        #for the above, sizeReduction = 4 or 7 seems to be about the sweet spot
        #the below line creates reduced size arrays to fit the model to, allowing for faster compute time. 
        zsReduced, muReduced, muerrReduced = zs[range(0, len(zs), sizeReduction)], mu[range(0, len(mu), sizeReduction)], muerr[range(0, len(muerr), sizeReduction)]
        
        bins = 20       #this is the number of bins to be used in the corner plots
        
        if dataset in ["Data00", "Data0", "Data1", "Data2", "Data3"]:
            nwalkers = 50       #this is how many 'walkers' will be used in the MCMC
            initial = [0.3, 0.7]        #an initial guess for the MCMC model
            ndim = len(initial)     #this is the number of dimensions to analyse over
            niter = 1000        #how many iterations of MCMC to do
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
        else:
            #no comments for this section since it's functionally identical to the above section
            nwalkers = 50
            initial = [0.3, 0.7, 0.2, -1, -0.5]
            ndim = len(initial)
            niter = 30000
            data = (zsReduced, muReduced, muerrReduced)
            p0 = [initial + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
            
            sampler, pos, prob, state = mcmc(p0, nwalkers, niter, ndim, lnprob, data)
            samples = sampler.flatchain

            flat_samples = sampler.get_chain(flat=True)

            best_params = flat_samples[np.argmax(sampler.flatlnprobability)]
            
            maxlikeli = np.zeros(ndim)
            oms = []; ols = []; orads = []; w0s = []; was = [];
            for state in flat_samples:
                oms.append(state[0]); ols.append(state[1]); orads.append(state[2]); w0s.append(state[3]); was.append(state[4])
            oms, ols, orads, w0s, was = np.array(oms), np.array(ols), np.array(orads), np.array(w0s), np.array(was)
            for i, parameter in enumerate([oms, ols, orads, w0s, was]):
                y, x, _ = plt.hist(parameter, bins=bins)
                maxlikeli[i] = x[y.argmax()]
            
            
        labels = ["$\Omega_m$", "$\Omega_\Lambda$", "$\Omega_r$", "$w_0$", "$w_a$"]
        fig, ax = plt.subplots()
        mean_params = np.quantile(samples, 0.5, axis=0)     #the mean (median!) of each parameter distribution corresponds to the 50th percentile
        
        #the following plots the corner plot, with a nice blue colour, red lines corresponding to the median values, and 1 sigma uncertainties below and above
        figure = corner.corner(flat_samples, labels=labels, show_titles=True, color='#307CC0', plot_datapoints=False, quantiles=[0.16, 0.84], truths=mean_params, truth_color='r', use_math_text=True)
            
        # Extract the axes of the corner plot
        axes = np.array(figure.axes).reshape((ndim, ndim))
        # mean_params = np.mean(samples, axis=0)
        if dataset in ["Data4", "Data5", "Data6", "Data7"]:
            #if the model is complicated, this will overlay a purple line for the maximum likelihood value of each parameter too
            
            # Loop over the diagonal (distribution) plots
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(maxlikeli[i], color="#8F4F9F", alpha=1)
                
            # Loop over the contour plots 
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(maxlikeli[xi], color="#8F4F9F", alpha=1)     #mean values are in purple
                    ax.axhline(maxlikeli[yi], color="#8F4F9F", alpha=1)
                    ax.plot(maxlikeli[xi], maxlikeli[yi], "s", color="#8F4F9F", alpha=1)    #plot a little square at the max-likelihood intersections
        else:
            mean_params = np.append(mean_params, extra_params)      #this fixes the first few dataset parameters that are missing orad, w0, wa
        plt.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Corner Plot.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        plt.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Corner Plot.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        plt.close(fig)
        
        #plot_params determines which set of fit parameters are to be plotted (and the chi2 calculated)
        plot_params = mean_params       #options are mean_params, best_params, and maxlikeli
        
        #calculate the chi2 of the model, and also its chi2 per degree of freedom
        chi2 = np.sum((model(plot_params, zsReduced, muReduced, muerrReduced) - muReduced)**2 / muerrReduced**2)
        chi2_reduced = chi2 / (len(muReduced) - 2)
        
        #now to create the models with which we'll plot against each other
        mu_om00_ox00 = dist_mod(zs, om=0.0, ol=0.0)  # We're going to use this empty model as a benchmark to compare the others to
        mu_om10_ox00 = model((1, 0, 0, -1, 0), zs, mu, muerr)
        mu_om03_ox00 = model((0.3, 0, 0, -1, 0), zs, mu, muerr)
        mu_om03_ox07 = model((0.3, 0.7, 0, -1, 0), zs, mu, muerr)
        mu_omModel_oxModel = model(plot_params, zs, mu, muerr)
        
        # Now plot a Hubble diagram relative to the empty model (i.e. subtract the empty model from all the data and models)
        fig, ax = plt.subplots()
        ax.errorbar(zs, mu-mu_om00_ox00, yerr=muerr, fmt='.', elinewidth=0.7, markersize=4, alpha=0.5)   #this is the data with associated error
        ax.plot(zs, mu_om10_ox00-mu_om00_ox00, '--',color='#8F4F9F', label='(1.0, 0.0)', alpha=0.5)
        ax.plot(zs, mu_om03_ox00-mu_om00_ox00, '--',color='blue', label='(0.3, 0.0)', alpha=0.5)
        ax.plot(zs, mu_om03_ox07-mu_om00_ox00, '--',color='green', label='(0.3, 0.7)', alpha=0.5)
        ax.plot(zs, mu_omModel_oxModel - mu_om00_ox00, '-', color='red', label=f'({plot_params[0]:.2f}, {plot_params[1]:.2f})')
        ax.axhline(y=0.0, ls=':', color='black')    #horizontal dotted line for the empty universe
        ax.set_xlim(0, 1.0)     
        ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude [Normalised to (0,0)]')
        ax.legend(frameon=False)
        fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Norm Mag vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Norm Mag vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        plt.close(fig)
        
        
        
        # Plot the data with model fits. This magnitude vs redshift plot is called a Hubble diagram
        fig, ax = plt.subplots()
        ax.errorbar(zs, mu, yerr=muerr, fmt='.', elinewidth=0.7, markersize=4, alpha=0.5)
        ax.plot(zs, mu_om10_ox00, '--', color='#8F4F9F', label='(1.0, 0.0)', alpha=0.5)
        ax.plot(zs, mu_om03_ox00, '--', color='blue', label='(0.3, 0.0)', alpha=0.5)
        ax.plot(zs, mu_om03_ox07, '--', color='g', label='(0.3, 0.7)', alpha=0.5)
        ax.plot(zs, mu_omModel_oxModel, '-', color='red', label=f'Model = ({plot_params[0]:.2f}, {plot_params[1]:.2f})')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude')
        ax.legend(frameon=False)
        fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Magnitude vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Magnitude vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
        plt.close(fig)
    
    
    
        #the following text writes to a .txt file with all of the relevant parameters of the best fit model so that we don't need to run the code again to see :)
        text = open(f'Part Two Graphs/{dataset}/{dataset} MCMC details.txt', "w")
        text.write(str('The program ran with ' + str(nwalkers) + ' walkers over ' + str(niter) + ' iterations.'))
        text.write('\nThis corresponds to effectively ' + str(nwalkers * niter) + ' models sampled.')
        text.write('\nThe mean value of each parameter is (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%(mean_params[0], mean_params[1], mean_params[2], mean_params[3], mean_params[4]))
        text.write('\nThe maximum likelihood value of each parameter is (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%(maxlikeli[0], maxlikeli[1], maxlikeli[2], maxlikeli[3], maxlikeli[4]))
        text.write('\nThe plot fit values are (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%(plot_params[0], plot_params[1], plot_params[2], plot_params[3], plot_params[4]))
        text.write('\nand the best fit values are (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%(best_params[0], best_params[1], best_params[2], best_params[3], best_params[4]))
        text.write('\nThe Reduced chi^2 for the plot fit is %0.2f'%chi2_reduced)
        text.write('\nand the Chi^2 value is %0.2f'%chi2)
        text.close()

#for multiprocessing to work, the program must be run with the __name__ == ... method
if __name__ == '__main__':
    main()