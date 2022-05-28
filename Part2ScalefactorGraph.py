# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:42:39 2022

@author: ryanw
"""
import numpy as np
import os 
from matplotlib import pyplot as plt
import matplotlib.ticker
from matplotlib import rc
from scipy import integrate

def adotinv(a, om, ol, orad, w0, wa):
    ok = 1 - om - ol - orad
    wDarkEnergy = 3 * (1 + w0 + wa * (1 - a))
    adot = a * np.sqrt(orad * a**-4 + om * a**-3 + ok * a**-2 + ol * a**-wDarkEnergy)
    return 1.0/adot

def nobigbang(om):
    if om < 1/2:
        return 4 * om * (np.cosh(1/3 * np.arccosh((1 - om) / om)))**3
    else:
        return 4 * om * (np.cos(1/3 * np.arccos((1 - om) / om)))**3

#plt.rcParams.update(plt.rcParamsDefault)   #only uncomment this if you want to restore the default font
plt.rcParams['font.family'] = 'Serif'


#def main():
dir_path = os.path.dirname(os.path.realpath(__file__))

### --- Constants --- ###
c = 299792.458 # km/s (speed of light)
H0kmsmpc = 70.  # Hubble constant in km/s/Mpc
H0s = H0kmsmpc * 3.2408e-20 # H0 in inverse seconds is H0 in km/s/Mpc * (3.2408e-20 Mpc/km)
H0y = H0s * 3.154e7 * 1.e9 # H0 in inverse Giga years is H0 in inverse seconds * (3.154e7 seconds/year) * (1e9 years / Giga year)
cH0mpc = c/H0kmsmpc   # c/H0 in Mpc  (the km/s cancel out in the numerator and denominator)
cH0Glyr = cH0mpc * 3.262 / 1000 #c/H0 in billions of light years.  There are 3.262 light year / parsec
    




### --- Scalefactor Graphs --- ### 

# Start by making an array of scalefactors
astart, astop, astep = 0, 3, 0.01
a_arr = np.arange(astart, astop, astep)

# Note that when you integrate something with more than one argument you pass it with args=(arg1,arg2) in the integrate function
# e.g. "integrate.quad(adotinv, lower_limit, uper_limit, args=(om,ol))""

param_arr = (('Data00', 0.3, 0.7, 0, -1, 0, 'black'), 
             ('Data0', 0.185, 0.323, 0, -1, 0, 'slategrey'), 
             ('Data1', 0.251, 0.219, 0, -1, 0, 'slateblue'), 
             ('Data2', 0.595, 0.151, 0, -1, 0, 'mediumorchid'), 
             ('Data3', 0.109, 0.510, 0, -1, 0, 'royalblue'), 
             ('Data4', 0.394, 0.306, 0.198, -0.770, -0.516, 'seagreen'), 
             ('Data5', 0.495, 0.257, 0.496, -0.459, -0.506, 'tab:olive'), 
             ('Data6', 0.115, 0.609, 0.052, -1.202, 0.644, 'lightseagreen'), 
             ('Data7', 0.091, 0.484, 0.039, -0.927, -0.747, 'tab:red'))

bigbang = []
age = []

fig, ax = plt.subplots()
for params in param_arr:
    name, om, ol, orad, w0, wa, c = params
    t_lookback_Gyr = np.array([integrate.quad(adotinv, 1, upper_limit, args=(om, ol, orad, w0, wa))[0] for upper_limit in a_arr]) / H0y  #calculates lookback time for some om 
    ax.plot(t_lookback_Gyr, a_arr, label=f'{name}')     #plots the scale factor vs lookback time
    if ol >= nobigbang(om):
        bigbang.append([name, "no"])
    else:
        bigbang.append([name, "yes"])
    age.append([name, round(integrate.quad(adotinv, 0, 1, args=(om, ol, orad, w0, wa))[0] / H0y, 2)])
    

ax.axvline(x=0,linestyle=':'); ax.axhline(y=1,linestyle=':') #Plot some crosshairs 
ax.set_xlabel('Lookback time (Gyr)'); ax.set_ylabel('Scalefactor ($a$)')  
ax.set_ylim(ymin=0, ymax=3); ax.set_xlim(xmin=-24, xmax=60)
ax.legend(loc='upper left',frameon=False)
ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
fig.savefig(dir_path+'\\Part Two Graphs\\Scalefactor-vs-LookbackTime Datasets.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part Two Graphs\\Scalefactor-vs-LookbackTime Datasets.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


text = open(f'Part Two Graphs/Dataset details MCMC.txt', "w")
text.write(str(bigbang))
text.write("\n" + str(age))
text.close()




# if __name__ == "__main__":
#     main()