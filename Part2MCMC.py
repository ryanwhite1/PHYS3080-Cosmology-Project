# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:19:26 2022

@author: ryanw

https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
"""

# First let's set up our packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import emcee
import corner

plt.rcParams['font.family'] = 'Serif'

# And set some constants
c = 299792.458 # km/s (speed of light)
H0kmsmpc = 70.  # Hubble constant in km/s/Mpc
H0s = H0kmsmpc * 3.2408e-20 # H0 in inverse seconds is H0 in km/s/Mpc * (3.2408e-20 Mpc/km)

# Write a function for the integrand, i.e. $1/E(z)$,
def ezinv(z, om=0.3, ol=0.7, orad=0.0, w0=-1.0, wa=0.0):
    ok = 1 - om - ol - orad
    wDarkEnergy = 3 * (1 + w0 + wa * (1 - (1 / (1 + z))))
    inside = orad * (1 + z)**4 + om * (1 + z)**3 + ok * (1 + z)**2 + ol * (1 + z)**wDarkEnergy
    if inside <= 0:
        return 1
    else:
        ez = np.sqrt(inside) 
        return 1 / ez
    
# The curvature correction function
def Sk(xx, ok):
    if ok < 0.0:
        dk = np.sin(np.sqrt(-ok) * xx) / np.sqrt(-ok)
    elif ok > 0.0:
        dk = np.sinh(np.sqrt(ok) * xx) / np.sqrt(ok)
    else:
        dk = xx
    return dk

# The distance modulus
def dist_mod(zs, om=0.3, ol=0.7, orad=0.0, w0=-1.0, wa=0.0):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ol - orad
    xx = np.array([integrate.quad(ezinv, 0, z, args=(om, ol, orad, w0, wa))[0] for z in zs])
    D = Sk(xx, ok)
    lum_dist = D * (1 + zs) 
    dist_mod = 5 * np.log10(lum_dist) # Distance modulus
    # Add an arbitrary constant that's approximately the log of c on Hubble constant minus absolute magnitude of -19.5
    dist_mod = dist_mod + np.log(c / H0kmsmpc) - (-19.5)  # You can actually skip this step and it won't make a difference to our fitting
    return dist_mod

# Add a new function that reads in the data (data files should be in a directory called data)
def read_data(model_name):
    d = np.genfromtxt('SNe Data/' + model_name+'.txt', delimiter=',')
    zs = d[:, 0]
    mu = d[:, 1]
    muerr=d[:, 2]
    return zs, mu, muerr

#to implement MCMC, we need four 5 main functions
#the first function is the model to which we need to fit to the data
def model(params, zs=None, mu=None, muerr=None):
    if zs is None or mu is None or muerr is None:
        zs=zsReduced; mu=muReduced; muerr=muerrReduced
    if len(params) == 2:
        orad, w0, wa = 0, -1, 0
        om, ol = params
    else:
        om, ol, orad, w0, wa = params
    mu_model = dist_mod(zs, om, ol, orad, w0, wa)     # calculate the distance modulus vs redshift for that model 
    mscr = np.sum((mu_model - mu) / muerr**2) / np.sum(1 / muerr**2) # Calculate the vertical offset to apply
    mu_model_norm = mu_model - mscr         # Apply the vertical offset
    return mu_model_norm

#the second function is the 
def lnlike(params, zs=None, mu=None, muerr=None):
    if zs is None or mu is None or muerr is None:
        zs=zsReduced; mu=muReduced; muerr=muerrReduced
    LnLike = -0.5 * np.sum((model(params) - mu)**2 / muerr**2)
    return LnLike

def lnprior(params):
    if len(params) == 2: #the below priors are for datasets 00 to 3 only
        om, ol = params
        if (0 <= om <= 1) and (-1 <= ol <= 1):
            return 0.0
        else:
            return -np.inf
    else: #the below parameters are for dataset 4-6
        om, ol, orad, w0, wa = params
        if (0 <= om <= 2.5) and (-2 <= ol <= 2) and (0 <= orad <= 2) and (-2 <= w0 <= 1) and (-1 <= wa <= 1.5):
            return 0.0
        else:
            return -np.inf

def lnprob(params, zs=None, mu=None, muerr=None):
    if zs is None or mu is None or muerr is None:
        zs=zsReduced; mu=muReduced; muerr=muerrReduced
    lp = lnprior(params)
    if lp != 0.0:
        return -np.inf
    return lp + lnlike(params, zs, mu, muerr)

def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

def plotter(sampler, zs=None, mu=None):
    if zs is None or mu is None:
        zs=zsReduced; mu=muReduced
    plt.ion()
    plt.scatter(zs, mu, marker='.')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(zs, model(theta, zs), color="r", alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel(r'Redshift ($z$)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()



runall = "Yes"      #change this if you want to run just one dataset

if runall == "Yes":
    datasets = ["Data00", "Data0", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7"]
else:
    datasets = ["Data4", "Data5", "Data6", "Data7"]        #change this to the data set you want to run


for dataset in datasets:
    print(dataset)
    number = int(dataset[-1])
    zs, mu, muerr = read_data(dataset)      #read the data and assign it to arrays
    sizeReduction = 2       #value of 1 corresponds to reading all data. Value of 2 for example, fits the data against 1 in every 2 points (~ 50% time reduction)
    #for the above, sizeReduction = 4 or 7 seems to be about the sweet spot, and gives Chi2 ~ 1
    #the below line creates reduced size arrays to fit the model to, allowing for faster comput time. 
    zsReduced, muReduced, muerrReduced = zs[range(0, len(zs), sizeReduction)], mu[range(0, len(mu), sizeReduction)], muerr[range(0, len(muerr), sizeReduction)]
    
    mu_om00_ox00 = dist_mod(zs, om=0.0, ol=0.0)  # We're going to use this empty model as a benchmark to compare the others to
    mu_om10_ox00 = dist_mod(zs, om=1.0, ol=0.0)
    mu_om03_ox00 = dist_mod(zs, om=0.3, ol=0.0)
    mu_om03_ox07 = dist_mod(zs, om=0.3, ol=0.7)
    if dataset in ["Data00", "Data0", "Data1", "Data2", "Data3"]:
        nwalkers = 20
        initials = [[0.3, 0.7], [0.3, 0.7], [0.2, 0.1], [0.4, -0.4], [0, 0.4]]
        initial = np.array(initials[number])
        ndim = len(initial)
        niter = 500
        data = (zsReduced, muReduced, muerrReduced)
        p0 = [initial + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
        samples = sampler.flatchain
        best_params  = samples[np.argmax(sampler.flatlnprobability)]
        best_params = np.append(best_params, [0, -1, 0])
        plotter(sampler)
        flat_samples = sampler.get_chain(flat=True)
    else:
        nwalkers = 50
        initials = [[0, -0.5, 0.2, -0.8, 0.2], [0.5, 0.2, 0.1, -0.6, 0], [0.1, 0.5, 0.2, -0.5, 0.6], [0.3, -0.5, 0.2, -1, -0.2]]
        initial = np.array(initials[number-4])
        ndim = len(initial)
        niter = 2000
        data = (zsReduced, muReduced, muerrReduced)
        p0 = [initial + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
        samples = sampler.flatchain
        best_params  = samples[np.argmax(sampler.flatlnprobability)]
        plotter(sampler)
        flat_samples = sampler.get_chain(flat=True)
    
    labels = ["$\Omega_m$", "$\Omega_\Lambda$", "$\Omega_r$", "$w_0$", "$w_a$"]
    fig, ax = plt.subplots()
    figure = corner.corner(flat_samples, labels=labels, show_titles=True, color='#307CC0', plot_datapoints=False, quantiles=[0.16, 0.5, 0.84], use_math_text=True)
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
    value2 = np.mean(samples, axis=0)
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(value2[i], color="r", alpha=0.4)
        
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r", alpha=0.4)
            ax.axhline(value2[yi], color="r", alpha=0.4)
            ax.plot(value2[xi], value2[yi], "sr", alpha=0.4)
    plt.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Corner Plot.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Corner Plot.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
        
    mu_omModel_oxModel = dist_mod(zs, best_params[0], best_params[1], best_params[2], best_params[3], best_params[4])
    
    chi2 = np.sum((model(best_params) - muReduced)**2 / muerrReduced**2)
    chi2_reduced = chi2 / (len(muReduced) - 2)
    
    # Calculate mscript for each of the models, which is the thing that determines the vertical normalisation 
    mscr_om10_ox00 = np.sum((mu_om10_ox00 - mu) / muerr**2) / np.sum(1 / muerr**2)
    mscr_om03_ox00 = np.sum((mu_om03_ox00 - mu) / muerr**2) / np.sum(1 / muerr**2)
    mscr_om03_ox07 = np.sum((mu_om03_ox07 - mu) / muerr**2) / np.sum(1 / muerr**2)
    mscr_omModel_oxModel = np.sum((mu_omModel_oxModel - mu) / muerr**2) / np.sum(1 / muerr**2)
    #print(mscr_om10_ox00,mscr_om03_ox00,mscr_om03_ox07)
    mu_om10_ox00 = mu_om10_ox00 - mscr_om10_ox00
    mu_om03_ox00 = mu_om03_ox00 - mscr_om03_ox00
    mu_om03_ox07 = mu_om03_ox07 - mscr_om03_ox07
    mu_omModel_oxModel = mu_omModel_oxModel - mscr_omModel_oxModel
    
    # Now plot a Hubble diagram relative to the empty model (i.e. subtract the empty model from all the data and models)
    fig, ax = plt.subplots()
    ax.errorbar(zs, mu-mu_om00_ox00,yerr=muerr, fmt='.', elinewidth=0.7, markersize=4, alpha=0.5)
    ax.plot(zs, mu_om10_ox00-mu_om00_ox00, '--',color='red', label='(1.0, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox00-mu_om00_ox00, '--',color='blue', label='(0.3, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox07-mu_om00_ox00, '--',color='green', label='(0.3, 0.7)', alpha=0.5)
    ax.plot(zs, mu_omModel_oxModel - mu_om00_ox00, '-', color='#8F4F9F', label=f'({best_params[0]:.2f}, {best_params[1]:.2f})')
    ax.axhline(y=0.0, ls=':', color='black')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude [Normalised to (0,0)]')
    ax.legend(frameon=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Norm Mag vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Norm Mag vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
    
    
    # Plot it to see what it looks like, this is called a Hubble diagram
    fig, ax = plt.subplots()
    ax.errorbar(zs, mu, yerr=muerr, fmt='.', elinewidth=0.7, markersize=4, alpha=0.5)
    ax.plot(zs, mu_om10_ox00, '--', color='red', label='(1.0, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox00, '--', color='blue', label='(0.3, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox07, '--', color='g', label='(0.3, 0.7)', alpha=0.5)
    ax.plot(zs, mu_omModel_oxModel, '-', color='#8F4F9F', label=f'Model = ({best_params[0]:.2f}, {best_params[1]:.2f})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude')
    ax.legend(frameon=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Magnitude vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} MCMC Magnitude vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
    
    
    #the following text writes to a .txt file with all of the relevant parameters of the best fit model so that we don't need to run the code again to see :)
    text = open(f'Part Two Graphs/{dataset}/{dataset} MCMC details.txt', "w")
    text.write(str('The program ran with ' + str(nwalkers) + ' over ' + str(niter) + ' iterations.'))
    text.write('\nThis corresponds to effectively ' + str(nwalkers * niter) + ' models sampled.')
    text.write('\nBest fit values are (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%(best_params[0], best_params[1], best_params[2], best_params[3], best_params[4]))
    text.write('\nReduced chi^2 for the best fit is %0.2f'%chi2_reduced)
    text.write('\n and the Chi^2 value is %0.2f'%chi2)
    text.close()