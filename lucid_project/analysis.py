import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import euclidean as dist
from copy import deepcopy
try:
    from Queue import Queue
except ImportError:
    from queue import Queue

from .utils.common import Blob
from . import lca

frame_xlabel = "Position [X] (Pixels)"
frame_ylabel = "Position [Y] (Pixels)"
frame_clabel = "Time-Over-Threshold [C] (Arbitrary Units)"
frame_elabel = "Energy per Pixel [E] (keV/px)"

##calib_curve=(2.78, 57.25, 1020, -8.96)

def energy_to_tot(E, calib_curve=(2.086, 82.192, 276.657, -0.389)):
    if E <= 0:
        return 0
    a, b, c, t = calib_curve
    return a*E + b - (c/(E-t))

def tot_to_energy(tot, calib_curve=(2.086, 82.192, 276.657, -0.389)):
    if tot <= 0:
        return 0
    a, b, c, t = calib_curve
    disc = abs((b-a*t-tot)**2 - 4*a*(tot*t-b*t-c))
    return ( (a*t+tot-b) +  disc**0.5 ) / (2*a)

def flip_pix(pix, y_flip, x_flip, coord_min, coord_max):
        if x_flip:
            x_f = coord_max[0] - pix[0]
        else:
            x_f = pix[0] - coord_min[0]
        
        if y_flip:
            y_f = coord_max[1] - pix[1]
        else:
            y_f = pix[1] - coord_min[1]

        return x_f, y_f

####just a fun function I accidentally found (ignore this) 
##def reflect_diag(A):
##    M = A + A.T - np.diag(np.diag(A))
##    return M

def bounding_box_rect(pixels):
    a,b,c,d = 255,255,0,0
    for pix in pixels:
        # Set min coords
        if pix[0] < a:
            a = pix[0]
        if pix[1] < b:
            b = pix[1]
        # Set max coords
        if pix[0] > c:
            c = pix[0]
        if pix[1] > d:
            d = pix[1]
    return (a,b,c,d)

class Figure():
        def __init__(self, figure):
            self.fig = figure

        def show(self):
            try:
                if len(self.fig) == 2:
                    fig = self.fig[0]
                    plt.show()
                    #plt.close(fig)
                elif len(self.fig) == 1:
                    fig = self.fig[0]
                    plt.axis('off')
                    plt.imshow(fig)
                    #plt.close()
            except:
                return "Failed"
            
        def save(self, file_path):
            try:
                if len(self.fig) == 2:
                    fig = self.fig[0]
                    fig.savefig(file_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                elif len(self.fig) == 1:
                    plt.imsave(file_path, self.fig[0])
                    #plt.close()
            except:
                return "Failed"
            else:
                return "Successful"


class Cluster(Blob):
        def __init__(self, cluster, energy=False):
            Blob.__init__(self, cluster)
            self.energy = self.set_energy(energy)

        def set_energy(self, boolean):
            self.energy = boolean
            if self.energy:
                self.energy_sum = self.c_sum()
                self.tot_sum = energy_to_tot(self.energy_sum)
            else:
                self.tot_sum = self.c_sum()
                self.energy_sum = tot_to_energy(self.tot_sum)
            return boolean
            
        def calculate(self, masked=False):
            if masked == True:
                self.density *= 2
                self.width *= 2
            return [self.num_pixels, self.density, self.radius, self.curvature_radius, self.line_residual, self.circle_residual, self.width, self.avg_neighbours]

        def classify(self, algorithm="neural", c_val=False):
            if not c_val:
                pixels = [ (pix[0],pix[1]) for pix in self.pixels ]
            pred = lca.classify(pixels, alg=algorithm, c_val=c_val)
            return pred

        def c_sum(self):
                c_sum = 0
                for pix in self.pixels:   
                        c_sum += pix[2]
                return c_sum
        
        def xyc(self):
                xyc = ""
                for pix in self.pixels:   
                        xyc += str(pix[0])+"\t"+str(pix[1])+"\t"+str(pix[2])+"\n"
                return xyc
           
        def flip(self, y_flip, x_flip, coord_min=[0,0], coord_max=[255,255]):
            cluster = []
            for pix in self.pixels:
                x_f, y_f = flip_pix(pix, y_flip, x_flip, coord_min, coord_max)
                if len(pix) > 2:
                    cluster.append((x_f, y_f, pix[2]))
                else:
                    cluster.append((x_f, y_f, 2))
            return Cluster(cluster, self.energy)

        def plot(self, inv_y=True, y_flip=True, x_flip=False):
            #c is tuple in list
            x,y,*c = zip(*self.pixels)
            pad = 5
            frame = np.zeros((max(x)-min(x)+1+2*pad, max(y)-min(y)+1+2*pad))
            for pix in self.pixels:
                x_f, y_f = flip_pix(pix, y_flip, x_flip, [min(x), min(y)], [max(x), max(y)])
                if len(pix) > 2:
                    frame[ x_f + pad ][ y_f + pad ] = pix[2]
                else:
                    frame[ x_f + pad][ y_f + pad ] = 2
                    
            min_val = 0
            if len(c) != 0 and max(c[0]) > 0:
                max_val = max(c[0])
            else:
                max_val = 256
            
            #a, r, length = 0.75, 4, 8
            a, r, length = 1, 2, 15
            
            fig,ax = plt.subplots(1)
            img = ax.imshow(frame.T, norm=SymLogNorm( linthresh = a*r*1.14), vmin=min_val, vmax=max_val, cmap="jet")
            if inv_y:
                ax.invert_yaxis()
            ax.set_xlabel(frame_xlabel)
            ax.set_ylabel(frame_ylabel)
                
            if (len(self.pixels[0]) > 2):
                ticks = [0]+[a * r ** (n - 1) for n in range(1, length + 1)]
                cbar = fig.colorbar(img, ticks=ticks)
                cbar.set_ticklabels(ticks)
                
                if self.energy:
                    cbar.set_label(frame_elabel)
                else:
                    cbar.set_label(frame_clabel)
            
            return Figure((fig,ax))


class ClusterList():
        def __init__(self, frame, cluster_list, energy):
            self.frame = frame
            self.list = cluster_list
            self.energy = energy
            
        def plot(self, patch_type=0, pad=1.5, y_flip=True, x_flip=False, classify=False):
            col="w"
            co="5"
            colours = ["#"+co*2+"FFFF", "#FFFF"+co*2, "#"+co*4+"FF", "#FF"+co*2+"FF", "#"+co*2+"FF"+co*2, "#FF"+co*4]
            fig, ax = Frame(self.frame, self.energy).plot().fig
            for cluster in self.list:
                if classify == True:
                    pred = cluster.classify()
                    col = colours[["gamma", "beta", "muon", "proton", "alpha", "others"].index(pred)]
                    
                cluster = cluster.flip(y_flip, x_flip)
                pixels = cluster.pixels
                if patch_type == 0:
                    suffix = "circle"
                    c, r = cluster.centroid, cluster.radius
                    r += pad
                    patch = patches.Circle(c,r, linewidth=1, fill=False, edgecolor=col, facecolor='none')

                elif patch_type == 1:
                    suffix = "ellipse"
                    centre = cluster.centroid
                    a,b,c,d = bounding_box_rect(pixels)
                    a -= pad
                    b -= pad
                    c += pad
                    d += pad
                    angle = ( cluster.best_fit_theta * 180/math.pi ) - 90
                    if c-a < d-b:
                        w, h = c-a, d-b
                    else:
                        w, h = d-b, c-a
                    patch = patches.Ellipse(centre,w,h,angle, linewidth=1, fill=False, edgecolor=col, facecolor='none')
                    
                elif patch_type == 2:
                    suffix = "rect"
                    a,b,c,d = bounding_box_rect(pixels)
                    a -= pad
                    b -= pad
                    c += pad
                    d += pad
                    patch = patches.Rectangle((a,b),c-a,d-b, linewidth=1, fill=False, edgecolor=col, facecolor='none')
                    
                elif patch_type == 3:
                    suffix = "square"
                    c, r = cluster.centroid, cluster.radius
                    r += pad
                    patch = patches.Rectangle((c[0]-r,c[1]-r),2*r,2*r, linewidth=1, fill=False, edgecolor=col, facecolor='none')

                ax.add_patch(patch)
            
            return Figure((fig,ax))

        ## FILTER PARTICLES BY CLASS ATTRIBUTES
        ##attributes = {"radius":[0,10], "energy_sum":[500, 10000]}
        def filtrate(self, attributes):
            cl = self.list
            for attr, filt in attributes.items():
                cl = [c for c in cl if filt[0] <= getattr(c, attr) <= filt[1]]
            self.list = cl
            return self
        
###################################################

class Clustering:

    def __init__(self, frame, rad=1.5, c_value=True, energy=False):
        self.aq = Queue()
        self.rad = rad
        self.irad = int(np.ceil(rad))
        self.frame = deepcopy(frame)
        self.mat = deepcopy(frame)
        self.c_value = c_value
        self.energy = energy
        self.blobs = []
        self.blob = None  # Holds currently active blob

    def circle(self, location):
        sqxs = range(max(0, location[0] - self.irad), min(256, location[0] + self.irad + 1))
        sqys = range(max(0, location[1] - self.irad), min(256, location[1] + self.irad + 1))
        square = [(x, y) for x in sqxs for y in sqys]
        return [p for p in square if (dist(p, location) <= self.rad)]

    def process(self, pixel):
        if self.c_value == False:
            pixel = (pixel[0], pixel[1])
        return pixel

    def add(self, pixel):
        (x, y, c) = pixel
        self.blob.append( self.process(pixel) )
        close_region = self.circle((x, y))
        for (x1, y1) in close_region:
            if self.mat[x1][y1] > 0:
                c1 = self.mat[x1][y1]
                self.mat[x1][y1] = 0
                self.aq.put( (x1, y1, c1) )
                                
    def find(self):
        for x in range(256):
            for y in range(256):
                if self.mat[x][y] > 0:
                    ##added energy value
                    c = self.mat[x][y]
                    self.mat[x][y] = 0
                    self.blob = []
                    self.aq.put( (x, y, c) )

                    while not self.aq.empty():
                        pixel = self.aq.get()
                        self.add(pixel)

                    self.blobs.append(Cluster(self.blob, self.energy))
                    
        return ClusterList(self.frame, self.blobs, self.energy)

##########################################################


class Frame():
        def __init__(self, frame, energy=False):
            self.frame_c = frame
            ##private variable, do not change via class attrib, use set_energy
            self.energy = energy
            
        def set_energy(self, boolean):
            self.energy = boolean
            if self.energy:
                e_func = np.vectorize(tot_to_energy)
                self.frame_c = e_func(self.frame_c)
            return boolean
                
        def cluster(self, rad=1.5, c_value=True):
                return Clustering(self.frame_c, rad, c_value, self.energy).find()

        def image(self):
                mat = self.frame_c.T
                min_val = 0
                max_val = np.max(mat) if (np.count_nonzero(mat) > 0) else 2
                #a, r, length = 0.75, 4, 8
                a, r, length = 1, 2, 15
                cmap = plt.cm.jet
                norm = matplotlib.colors.SymLogNorm( linthresh = a*r*1.14, vmin=min_val, vmax=max_val)
                img = np.array([cmap(norm(mat))])
                return Figure(img)

        def plot(self, inv_y=True, y_flip=True, x_flip=False):
            
                mat = self.frame_c.T
                if y_flip:
                    mat = np.flipud(mat)
                if x_flip:
                    mat = np.fliplr(mat)
                    
                min_val = 0
                max_val = np.max(mat) if (np.count_nonzero(mat) > 0) else 2
                
                #a, r, length = 0.75, 4, 8
                a, r, length = 1, 2, 15
                
                fig,ax = plt.subplots(1)
                img = ax.imshow(mat, norm=SymLogNorm( linthresh = a*r*1.14), vmin=min_val, vmax=max_val, cmap="jet")
                if inv_y:
                    ax.invert_yaxis()
                ax.set_xlabel(frame_xlabel)
                ax.set_ylabel(frame_ylabel)

                ticks = [0]+[a * r ** (n - 1) for n in range(1, length + 1)]
                cbar = fig.colorbar(img, ticks=ticks)
                cbar.set_ticklabels(ticks)
                
                if self.energy:
                    cbar.set_label(frame_elabel)
                else:
                    cbar.set_label(frame_clabel)
                
                return Figure((fig,ax))


## Adapt this for url access and direct file path (no extra param)
class XYC():
        def read(file, file_type="frame"):
            if (file_type == "frame"):
                frame = np.zeros((256, 256))
                f = open(file)
                for line in f.readlines():
                        vals = line.split("\t")
                        x = int(float(vals[0].strip()))
                        y = int(float(vals[1].strip()))
                        c = int(float(vals[2].strip()))
                        frame[x][y] = c
                return Frame(frame)
            elif (file_type == "cluster"):
                f = open(file)
                pixels = []
                for line in f.readlines():
                        vals = line.split("\t")
                        x = int(float(vals[0].strip()))
                        y = int(float(vals[1].strip()))
                        c = int(float(vals[2].strip()))
                        pixels.append((x,y,c))
                return Cluster(pixels)
