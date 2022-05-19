# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:19:26 2022

@author: ryanw
"""

# First let's set up our packages
import numpy as np
import corner
from matplotlib import pyplot as plt
from scipy import integrate

plt.rcParams['font.family'] = 'Serif'

# And set some constants
c = 299792.458 # km/s (speed of light)
H0kmsmpc = 70.  # Hubble constant in km/s/Mpc
H0s = H0kmsmpc * 3.2408e-20 # H0 in inverse seconds is H0 in km/s/Mpc * (3.2408e-20 Mpc/km)

# Write a function for the integrand, i.e. $1/E(z)$,
def ezinv(z, om=0.3, ol=0.7, w0=-1.0, wa=0.0, orad=0.0):
    ok = 1 - om - ol - orad
    wDarkEnergy = 3 * (1 + w0 + wa * (1 - (1 / (1 + z))))
    inside = orad * (1 + z)**4 + om * (1 + z)**3 + ok * (1 + z)**2 + ol * (1 + z)**wDarkEnergy
    if inside < 0:
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
def dist_mod(zs, om=0.3, ol=0.7, w0=-1.0, wa=0.0, orad=0.0):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ol - orad
    xx = np.array([integrate.quad(ezinv, 0, z, args=(om, ol, w0, wa, orad))[0] for z in zs])
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

runall = "n"      #change this if you want to run just one dataset

if runall == "Yes":
    datasets = ["Data00", "Data0", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7"]
else:
    datasets = ["Data6"]        #change this to the data set you want to run


for dataset in datasets:
    print(dataset)
    zs, mu, muerr = read_data(dataset)      #read the data and assign it to arrays
    sizeReduction = 4       #value of 1 corresponds to reading all data. Value of 2 for example, fits the data against 1 in every 2 points (~ 50% time reduction)
    #for the above, sizeReduction = 4 or 7 seems to be about the sweet spot, and gives Chi2 ~ 1
    #the below line creates reduced size arrays to fit the model to, allowing for faster comput time. 
    zsReduced, muReduced, muerrReduced = zs[range(0, len(zs), sizeReduction)], mu[range(0, len(mu), sizeReduction)], muerr[range(0, len(muerr), sizeReduction)]
    
    mu_om00_ox00 = dist_mod(zs, om=0.0, ol=0.0)  # We're going to use this empty model as a benchmark to compare the others to
    mu_om10_ox00 = dist_mod(zs, om=1.0, ol=0.0)
    mu_om03_ox00 = dist_mod(zs, om=0.3, ol=0.0)
    mu_om03_ox07 = dist_mod(zs, om=0.3, ol=0.7)
    
    # Set up the arrays for the models you want to test, e.g. a range of Omega_m and Omega_Lambda models:
    #be careful increasing n and nSmall! Compute time increases as n^2 * nSmall^3
    n = 10                  # Increase this for a finer grid.
    nSmall = 6
    nTiny = 4
    ols = np.linspace(0, 1, n)   # Array of cosmological constant values
    #the above datasets are computed according to simple models, and so the compute time can be reduced by reducing the sample space
    if dataset in ["Data00", "Data0", "Data1", "Data2", "Data3"]:
        #the above datasets are computed according to simple models, and so the compute time can be reduced by reducing the sample space
        oms = np.linspace(0, 1, n)   # Array of matter densities
        orads = [0]     #data sets have orad = 0
        w0_array = [-1]     #data sets have w0 = -1
        wa_array = [0]      #data sets have wa = 0
    else:           
        oms = np.linspace(0, 1, n)   # Array of matter densities
        orads = np.linspace(0, 1, nSmall)    # Array of radiation density values
        w0_array = np.linspace(-2, 0, nTiny)    # Array of Dark Energy Equation of state, w0
        wa_array = np.linspace(-1.5, 0.5, nSmall)     # Array of change of Dark energy equation of state, wa
        
    chi2 = np.ones((len(oms), len(ols), len(orads), len(w0_array), len(wa_array))) * np.inf  # Array to hold our chi2 values, set initially to super large values
    print("Number of parameter samples iterated over = ", len(oms) * len(ols) * len(orads) * len(w0_array) * len(wa_array))
    
    # Calculate Chi2 for each model
    for i, om in enumerate(oms):                # loop through matter densities   
        print(i, "/", n, "complete.")                                         
        for j, ol in enumerate(ols):            # loop through cosmological constant densities
            for k, orad in enumerate(orads):                           
                for l, w0 in enumerate(w0_array):
                    for m, wa in enumerate(wa_array):
                        mu_model = dist_mod(zsReduced, om=om, ol=ol, orad=orad, w0=w0, wa=wa)     # calculate the distance modulus vs redshift for that model 
                        mscr = np.sum((mu_model - muReduced) / muerrReduced**2) / np.sum(1 / muerrReduced**2) # Calculate the vertical offset to apply
                        mu_model_norm = mu_model - mscr         # Apply the vertical offset
                        chi2[i, j, k, l, m] = np.sum((mu_model_norm - muReduced)**2 / muerrReduced**2)  # Calculate the chi2 and save it in a matrix

                
    # Convert that to a likelihood and calculate the reduced chi2
    likelihood = np.exp(-0.5 * (chi2 - np.amin(chi2)))        # convert the chi^2 to a likelihood (np.amin(chi2) calculates the minimum of the chi^2 array)
    chi2_reduced = chi2 / (len(muReduced) - 2)                # calculate the reduced chi^2, i.e. chi^2 per degree of freedom, where dof = number of data points minus number of parameters being fitted 
    
    # Calculate the best fit values (where chi2 is minimum)
    indbest = np.argmin(chi2)                 # Gives index of best fit but where the indices are just a single number
    ibest   = np.unravel_index(indbest, [len(oms), len(ols), len(orads), len(w0_array), len(wa_array)]) # Converts the best fit index to the 2d version (i,j)
    print( 'Best fit values are (om, ol, orad)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%( oms[ibest[0]], ols[ibest[1]], orads[ibest[2]], w0_array[ibest[3]], wa_array[ibest[4]]))
    print( 'Reduced chi^2 for the best fit is %0.2f'%chi2_reduced[ibest[0], ibest[1], ibest[2], ibest[3], ibest[4]])
    
    mu_omModel_oxModel = dist_mod(zs, om=oms[ibest[0]], ol=ols[ibest[1]], orad=orads[ibest[2]], w0=w0_array[ibest[3]], wa=wa_array[ibest[4]])
    
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
    ax.plot(zs, mu_omModel_oxModel - mu_om00_ox00, '-', color='#8F4F9F', label=f'({oms[ibest[0]]:.2f}, {ols[ibest[1]]:.2f})')
    ax.axhline(y=0.0, ls=':', color='black')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude [Normalised to (0,0)]')
    ax.legend(frameon=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} Norm Mag vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} Norm Mag vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
    
    
    # Repeat the plot and see how it changes
    # Plot it to see what it looks like, this is called a Hubble diagram
    fig, ax = plt.subplots()
    ax.errorbar(zs, mu, yerr=muerr, fmt='.', elinewidth=0.7, markersize=4, alpha=0.5)
    ax.plot(zs, mu_om10_ox00, '--', color='red', label='(1.0, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox00, '--', color='blue', label='(0.3, 0.0)', alpha=0.5)
    ax.plot(zs, mu_om03_ox07, '--', color='g', label='(0.3, 0.7)', alpha=0.5)
    ax.plot(zs, mu_omModel_oxModel, '-', color='#8F4F9F', label=f'Model = ({oms[ibest[0]]:.2f}, {ols[ibest[1]]:.2f})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Apparent Magnitude')
    ax.legend(frameon=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} Magnitude vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} Magnitude vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
    
    # Plot contours of 1, 2, and 3 sigma
    fig, ax = plt.subplots()
    samples = len(oms) * len(ols) * len(orads) * len(w0_array) * len(wa_array)
    data = chi2.reshape(samples, 5)
    cornerplot = corner.corner(data, labels=["om", "ol", "orad", "w0", "wa"], quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # contourMatrix = np.transpose((chi2 - np.amin(chi2))[:, :, 0, 0, 0]) #removes all but the first two dimensions (om & ol respectively)
    # likelihoodPlot = ax.contour(oms, ols, contourMatrix, cmap="copper", **{'levels':[2.30,6.18,11.83]})
    # #ax.clabel(likelihoodPlot, likelihoodPlot.levels, inline=True, fontsize=4)      #puts the labels on the contour lines
    # labels = ["1 $\sigma$ Likelihood", "2 $\sigma$ Likelihood", "3 $\sigma$ Likelihood"]
    # for i in range(len(labels)):        #this makes a label on the axis for each of the contour lines
    #     likelihoodPlot.collections[i].set_label(labels[i])
    # ax.plot(oms[ibest[0]], ols[ibest[1]], 'x', color='black', label='(om,ol)=(%.3f,%.3f)'%(oms[ibest[0]], ols[ibest[1]]))
    # ax.errorbar(oms[ibest[0]], ols[ibest[1]], xerr=(oms[1]-oms[0]), yerr=(ols[1]-ols[0]), fmt='x', color='#307CC0', elinewidth=1, label='($\Omega_m$, $\Omega_\Lambda$)=(%.3f, %.3f)'%(oms[ibest[0]], ols[ibest[1]]))
    # ax.set_xlabel("$\Omega_m$", fontsize=12); ax.set_ylabel("$\Omega_\Lambda$", fontsize=12)
    # ax.set_xlim(min(oms), max(oms)); ax.set_ylim(min(ols), max(ols))
    # #ax.plot([oms[0], oms[1]], [ols[0], ols[1]], '-', color='black', label='Step size indicator' ) # Delete this line after making step size smaller!
    # ax.legend(loc='best', frameon=True)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} contours.png', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    fig.savefig(f'Part Two Graphs/{dataset}/{dataset} contours.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01, transparent=False)
    plt.close(fig)
    
    #the following text writes to a .txt file with all of the relevant parameters of the best fit model so that we don't need to run the code again to see :)
    text = open(f'Part Two Graphs/{dataset}/{dataset} details.txt', "w")
    text.write('The sample size of parameters tested was (om, ol, orad, w0, wa) = (' + str(n) + ', ' + str(n) + ', ' + str(nSmall) + ', ' + str(nSmall) + ', ' + str(nSmall) + ')')
    text.write('\nThe total number of models compared are ' + str(n**2 * nSmall**2 * nTiny))
    text.write('\nBest fit values are (om, ol, orad, w0, wa)=(%.3f, %.3f, %.3f, %.3f, %.3f)'%( oms[ibest[0]], ols[ibest[1]], orads[ibest[2]], w0_array[ibest[3]], wa_array[ibest[4]]))
    text.write('\nReduced chi^2 for the best fit is %0.2f'%chi2_reduced[ibest[0], ibest[1], ibest[2], ibest[3], ibest[4]])
    text.close()