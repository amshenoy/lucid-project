from __future__ import print_function
import numpy as np
from numpy import sin,cos,tan # For convenience
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import scipy.spatial
from matplotlib.colors import LogNorm
import sys
from mpl_toolkits.basemap import Basemap

def distfunc(dist):
    return np.exp(((dist**2)*-100))

def sqdist(p1,p2):
    return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )

def cartesian(lat, lng):
    r = 1 # It doesn't actually matter for our purposes
    lat = np.radians(lat)
    lng = np.radians(lng)

    x = r * cos(lat) * cos(lng)
    y = r * cos(lat) * sin(lng)
    z = r * sin(lat)
    return (x,y,z)
    
def lookup(particle_type, labels, lat, lng):
    if (lat%20 == 0 and lng%60 == 0):
        print(lat, lng)
    
    #if lat > 84 or lat < -84:
    #    return 0
    
    neighbours = 2000
    x,y,z = cartesian(lat,lng)
    dists, indices = tree.query((x,y,z), neighbours)

    # 3 is electrons, 4 is protons ... (electron, proton, gamma, muon, alpha, others)
    ##"Ratio of Negatively to Positively Charged Particles"
    ##particles = [(datapoints[index][3] + datapoints[index][6] + 1)/(datapoints[index][4] + datapoints[index][7] + 1)  for index in indices]
    
    type_list = ["", "", ""] + labels
    particles = [(datapoints[i][type_list.index(particle_type)]) for i in indices]
    val = np.average(particles, weights=[distfunc(dist) for dist in dists])
    return val

## labels in order of data (excluding lat, lng)
def map_make(data, labels = ["Electron", "Proton", "Gamma", "Muon", "Alpha", "Other"] ):
    num_dep_vars = len(data[0])-2
    try:
        assert(len(labels) == num_dep_vars)
    except:
        print("Number of variables/maps to be plotted must match number of labels.")

    # Convert all into cartesian coordinates
    datapoints = [cartesian(lat, lng) + tuple(datum) for lat,lng,*datum in data]
    tree = scipy.spatial.cKDTree([(x,y,z) for x,y,z,*d in datapoints])
    
    for i in range(num_dep_vars):
        label = labels[i]
        grid = [ [lookup(label, labels, lat, lng) for lng in range(-180,180)] for lat in range(-90,90)]
        plt.figure(figsize=(20,14), dpi=300)
        
        a, r, length = 1, 2, 15
        m = Basemap(projection='cyl')
        m.imshow(grid,norm=SymLogNorm(linthresh=a*r*1.14))
        m.drawcoastlines()
        ##m.fillcontinents(color='#00ff00',alpha=0.4)
        #m.shadedrelief()
        #m.etopo()
        #m.bluemarble()
        
        tick_fontsize = 16
        parallels = np.arange(-60,90,30)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[True,False,True,False], fontsize=tick_fontsize)
        meridians = np.arange(-150,180,30)
        m.drawmeridians(meridians,labels=[True,False,False,True], fontsize=tick_fontsize)
        
        horizontal_label = particle_type.title()+" Events ($\mathregular{cm^{-2}}$$\mathregular{s^{-1}}$)"
        ##horizontal_label = "Ratio of Negatively to Positively Charged Particle Events"
        
        ticks = [0]+[a * r ** (n - 1) for n in range(1, length + 1)]
        cbar = plt.colorbar(orientation='horizontal', ticks=ticks)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label(horizontal_label, fontsize=22)
        
        #plt.show()
        try:
            plt.savefig("./maps/"+label.lower()+"_map.png", dpi=300, frameon=False, bbox_inches="tight")
            plt.close()
            print(label.lower()+"_map.png saved successfully!")
        except:
            print("Failed to save plot!")
