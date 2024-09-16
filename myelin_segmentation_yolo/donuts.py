# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 08:34:43 2024

@author: simon
"""

import numpy as np
from skimage import draw, measure
from scipy import ndimage, spatial
import random
import math

import matplotlib.pyplot as plt



#donuts maker
def noisy_donuts(img_shape, off_center, thickness, noise_lvl) -> np.array:
    """
    Generate a noisy donut shape using a dual ellipsis substraction.
    This is done by calculating a center region, and selecting one or two center to create ellipsis.
    The rotation is randomly choosen and the same for both ellipsis.
    The minor axis and major axis is also randomly selectionned, with the condition it should fit the image dimension.
    Draw a first ellipsis, filled.
    Negatively draw a second ellipsis smaller by the thickness parameter on both axis.
    Noise is added by a 5 steps noisy dilation process. The noise level parameter control how much of the dilation is kept.
    
    #example
    donut = noisy_donuts((512,512), 50, random.randint(8, 30), .3)
    #visu_check
    plt.imshow(donut, interpolation='None')
    plt.axis('off')
    plt.show()
    
    Parameters
    ----------
    img_shape : tuple of 2 int
        The shape of the img
    off_center : int
        The difference in thickness along the donuts. Higher means more homogenuous. Randomized. 0 to deactivate.
    thickness : int
        The thickness of the donut, in pixel.
    noise_lvl : float
        Amount of noise to add on the donut edge. 0 to deactivate.

    Returns
    -------
    img : numpy array
        Same dimension as the image shape, bool type, contain the donut

    """
    img = np.zeros(img_shape, dtype=bool)
    #random center of both ellipse
    rr, cc = draw.ellipse(img_shape[0]//2, img_shape[1]//2,
                          img_shape[0]//off_center, img_shape[1]//off_center)
    c0 = (random.choice(rr), random.choice(cc))
    if off_center != 0:
        c1 = (random.choice(rr), random.choice(cc))
    else:
        c1 = c0
    
    #minor axis, major axis and rotation
    rotation = random.random() * math.pi * random.choice((-1, 1)) #random rotation
    min_axis_1 = random.randint(min(img_shape)//5, int(min(img_shape)/2.2))
    maj_axis_1 = random.randint(min(img_shape)//5, int(min(img_shape)/2.2))
    
    #assign 1st circle
    rr, cc = draw.ellipse(c0[0], c0[1], min_axis_1, maj_axis_1,
                          rotation=rotation)
    img[rr,cc] = True
    #make the ellips
    rr, cc = draw.ellipse(c1[0], c1[1],
                          min_axis_1-int(thickness), maj_axis_1-int(thickness),
                          rotation=rotation)
    img[rr,cc] = False
    
    #let's add some noisy dilation
    if noise_lvl != 0:
        for x in range(5): #number of time
            img_dil = ndimage.binary_dilation(img) #dilation
            img_dif = img ^ img_dil #only dilated pixel
            rr, cc = np.where(img_dif) #coordinate
            coo = np.random.choice(np.arange(len(rr)),
                                   size=int(len(rr)*noise_lvl),
                                   replace=False) #select random index
            #grab the coordinate
            rr = rr[coo]
            cc = cc[coo]
            img[rr,cc] = True #assign
        #img = ndimage.binary_dilation(img) #last, just to smooth a little
    
    return img

def is_close(donut, size_thr=500) -> bool:
    """
    Simple check to see if a donut is close or not. The size threshold is to exclude single pixel noise.

    Parameters
    ----------
    donut : numpy array
        Should contain a donut
    size_thr : int, optional
        Size threshold. Everything smaller will be consider as noise. The default is 500.

    Returns
    -------
    bool
        The check is the donut is close or not
    """
    donut_f = donut ^ ndimage.binary_fill_holes(donut)
    donut_hole_s = np.sum(donut_f)
    if donut_hole_s < size_thr:
        return False
    else:
        return True



def generate_contour(donut, biggest_only=True, as_hull_coo=True) -> np.array:
    """
    Generate a contour of the donut. If the donut is close, open it at a random location.
    NOT memory savvy

    Parameters
    ----------
    donut : Numpy array
        2D image containing the donut.
    biggest_only : bool, optional
        Only return the biggest contour, otherwise a list for each contour. The default is True.
    as_hull_coo : bool, optional
        first and last coordinate are the same to close the polygon. The default is True.

    Returns
    -------
    contours : Numpy array or list of numpy array
        Contain the coodinate of the contour in a [N, 2] shape

    """
    if is_close(donut): #just to be sure to not take in account single pixel hole due to the noisy dilation
        points = np.array(np.where(donut)).T
        hull = spatial.ConvexHull(points)
        
        vertices = hull.vertices
        #randomly select a point of the hull
        stop_pos = random.choice(vertices)
        #make the line
        rr, cc = draw.line_nd((int(round(np.mean(points[hull.vertices,0]))),
                               int(round(np.mean(points[hull.vertices,1])))),
                              (points[[stop_pos],0][0], points[[stop_pos],1][0]))
        #make an image of it
        img = np.zeros(donut.shape, dtype=bool) #<- memory needed to make the image
        img[rr, cc] = True
        img = ndimage.binary_dilation(img, iterations=3)
        rr, cc = np.where(img) #grab back the coordinate
        
        #assign
        donut[rr, cc] = False
    
    
    #measure the contour
    contours = measure.find_contours(donut, level=0.5)
    if biggest_only:
        contours_size = [len(x) for x in contours]
        contours_max_size_idx = np.argmax(contours_size) #select only the biggest
        contours = contours[contours_max_size_idx]
    
    if not as_hull_coo:
        contours = contours[:-1]
        
    return contours

if __name__ == "__main__":
    __DEBUG = False
    donut = noisy_donuts((512,512), 50, random.randint(8, 20), .2)
    contour = generate_contour(donut)
    
    #visu check
    if __DEBUG:
        plt.imshow(donut)
        plt.plot(contour[:, 1], contour[:, 0], 'c-',linewidth=1)
        plt.axis('off')
        plt.show()










