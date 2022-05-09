# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:42:39 2022

@author: ryanw
"""
import numpy as np
import os 
from matplotlib import pyplot as plt
from scipy import integrate

def adotinv(a, om, ol, orad):
    ok = 1 - om - ol - orad
    adot = a * np.sqrt(orad * a**-4 + om * a**-3 + ok * a**-2 + ol)
    return 1.0/adot

def aadotinv(a, om, ol, orad):
    ok = 1 - om - ol - orad
    adot = a**2 * np.sqrt(orad * a**-4 + om * a**-3 + ok * a**-2 + ol)
    return 1.0/adot

def Ez(z, om, ol, orad):
    ok = 1 - om - ol - orad
    Ez = np.sqrt(orad * (1 + z)**4 + om * (1 + z)**3 + ok * (1 + z)**2 + ol) # Put your code here!  This is not right until you change it.
    return 1 / Ez

def Sk(xx, om, ol, orad):
    ok = 1 - om - ol - orad
    if ok < 0:
        dk = np.sin(np.sqrt(-ok) * xx) / np.sqrt(-ok)
    elif ok > 0:
        dk = np.sinh(np.sqrt(ok) * xx) / np.sqrt(ok)
    else:
        dk = xx
    return dk






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
astart, astop, astep = 0, 2.1, 0.05
a_arr = np.arange(astart, astop, astep)

# Calculate for the universe we think we live in, with approximately matter density 0.3 and cosmological constant 0.7
om, ol, orad = 0.3, 0.7, 0.00005

# Note that when you integrate something with more than one argument you pass it with args=(arg1,arg2) in the integrate function
# e.g. "integrate.quad(adotinv, lower_limit, uper_limit, args=(om,ol))""
t_lookback_Gyr = np.array([integrate.quad(adotinv, 1, upper_limit, args=(om,ol, orad))[0] for upper_limit in a_arr])/H0y

fig, ax = plt.subplots()
ax.plot(t_lookback_Gyr,a_arr,label='$(\Omega_M,\Omega_\Lambda)$=(%.2f,%.2f)'%(om,ol)) 
ax.axvline(x=0,linestyle=':'); plt.axhline(y=1,linestyle=':') # Plot some crosshairs 
ax.set_xlabel('Lookback time (Gyr)'); ax.set_ylabel('Scalefactor')
ax.legend(loc='lower right',frameon=False)
fig.savefig(dir_path+'\\Part One Graphs\\Scalefactor-vs-LookbackTime for Our Universe.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Scalefactor-vs-LookbackTime for Our Universe.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


om_arr = np.arange(0,2.1,0.4)
fig, ax = plt.subplots()
for om in om_arr:
    t_lookback_Gyr = np.array([integrate.quad(adotinv, 1, upper_limit, args=(om, ol, orad))[0] for upper_limit in a_arr])/H0y
    ax.plot(t_lookback_Gyr, a_arr, label='$(\Omega_M,\Omega_\Lambda)$=(%.1f,%.1f)'%(om,ol))

# Plot this new model (note I've added a label that can be used in the legend)
#plt.plot(t_lookback_Gyr,a_arr,label='$(\Omega_M,\Omega_\Lambda)$=(%.2f,%.2f)'%(om,ol)) 
ax.axvline(x=0,linestyle=':'); ax.axhline(y=1,linestyle=':') #Plot some crosshairs 
ax.set_xlabel('Lookback time (Gyr)'); ax.set_ylabel('Scalefactor ($a$)')  
ax.legend(loc='upper left',frameon=False)
fig.savefig(dir_path+'\\Part One Graphs\\Scalefactor-vs-LookbackTime.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Scalefactor-vs-LookbackTime.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


fig, ax = plt.subplots()
redshift = -1 + (1 / a_arr)
ax.plot(a_arr, redshift)
ax.set_xlabel("Scalefactor ($a$)"); ax.set_ylabel("Redshift ($z$)")
fig.savefig(dir_path+'\\Part One Graphs\\Redshift vs Scalefactor.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Redshift vs Scalefactor.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)



### --- Redshift Graphs --- ###

# Start by making an array of redshifts
zstart, zstop, zstep = 0, 4, 0.01
zarr = np.arange(zstart, zstop + zstep, zstep)

# Now add your code to calculate distance vs redshift and then plot it.  
xarr = np.zeros(len(zarr))
axarr = np.zeros(len(zarr))
for i, z in enumerate(zarr):
    xarr[i] = integrate.quad(Ez, 0, z, args=(om, ol, orad))[0] 
    axarr[i] = (1 / (1 + z)) * integrate.quad(Ez, 0, z, args=(om, ol, orad))[0]
    
# Sub in the required constants to get the comoving distance R_0*X
R0X = xarr * cH0Glyr # Distance in Glyr
aR0X = axarr * cH0Glyr # Distance in Glyr

fig, ax = plt.subplots()
ax.plot(zarr,R0X)
ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('$R_0\chi$ (Glyr)')
ax.set_xlim(xmin=0); ax.set_ylim(ymin=0)
fig.savefig(dir_path+'\\Part One Graphs\\Comoving Distance vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Comoving Distance vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(zarr,aR0X)
ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('$aR_0\chi$ (Glyr)')
ax.set_xlim(xmin=0); ax.set_ylim(ymin=0)
fig.savefig(dir_path+'\\Part One Graphs\\Emission Distance vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Emission Distance vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


om, ol, orad = 0.3, 0.7, 0.00005
DD = R0X                                                     # Proper distance
DL = cH0Glyr * Sk(xarr, om, ol, orad) * (1 + zarr)           # Luminosity distance
DA = cH0Glyr * Sk(xarr, om, ol, orad) / (1 + zarr)           # Angular diameter distance

fig, ax = plt.subplots()
ax.plot(zarr, DD, label='Proper Distance')
ax.plot(zarr, DL, label='Luminosity Distance')
plt.plot(zarr, DA, label='Angular Diameter Distance')
ax.legend()
ax.set_xlabel('Redshift ($z$)'); ax.set_ylabel('Distances (Glyr)')
ax.set_xlim(xmin=0)
fig.savefig(dir_path+'\\Part One Graphs\\DL and DA vs Redshift.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\DL and DA vs Redshift.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


ScaleArr = np.arange(0.001, 2, 0.001)
ParticleHoriz = np.zeros(len(ScaleArr))
EventHoriz = np.zeros(len(ScaleArr))
HubbleHoriz = np.zeros(len(ScaleArr))
for i, a in enumerate(ScaleArr):
    ParticleHoriz[i] = integrate.quad(aadotinv, 0, a, args=(om, ol, orad))[0]
    EventHoriz[i] = integrate.quad(aadotinv, a, np.inf, args=(om, ol, orad))[0]
    HubbleHoriz[i] = adotinv(a, om, ol, orad)
    
# Sub in the required constants to get the comoving distance R_0*X
ParticleHoriz = ParticleHoriz * cH0Glyr # Distance in Glyr
EventHoriz = EventHoriz * cH0Glyr
HubbleHoriz = HubbleHoriz * cH0Glyr

fig, ax = plt.subplots()
ax.plot(ParticleHoriz, ScaleArr, color='g', label="Particle Horizon"); ax.plot(-ParticleHoriz, ScaleArr, color='g')
ax.plot(EventHoriz, ScaleArr, color='b', label="Event Horizon"); ax.plot(-EventHoriz, ScaleArr, color='b')
ax.plot(HubbleHoriz, ScaleArr, color='r', label="Hubble Sphere"); ax.plot(-HubbleHoriz, ScaleArr, color='r')
ax.legend(loc="lower right"); ax.grid(which="major")
ax.set_xlabel('Comoving Distance $R_0 \chi$ (Glyr)'); ax.set_ylabel('Scalefactor ($a$)')
#ax.set_xlim(xmin=0); ax.set_ylim(ymin=0)
fig.savefig(dir_path+'\\Part One Graphs\\Spacetime Diagram.png', dpi=200, bbox_inches='tight', pad_inches = 0.01)
fig.savefig(dir_path+'\\Part One Graphs\\Spacetime Diagram.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
plt.close(fig)


# if __name__ == "__main__":
#     main()