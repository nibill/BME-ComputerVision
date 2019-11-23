import numpy as np
import os
from PIL import Image


def load_merton_data(folder):
    im1 = np.array(Image.open(os.path.join(folder, 'images/001.jpg')))
    im2 = np.array(Image.open(os.path.join(folder, 'images/002.jpg')))
    
    # load 2D points for each view to a list
    points2D = [np.loadtxt(os.path.join(folder, f'2D/00{i+1:d}.corners')).T for i in range(3)]  

    # load 3D points
    points3D = np.loadtxt(os.path.join(folder, '3D/p3d')).T  

    # load correspondences
    corr = np.genfromtxt(os.path.join(folder, '2D/nview-corners'), dtype=int, missing_values='*') 

    # create cameras
    P = [np.loadtxt(os.path.join(folder, f'2D/00{i+1:d}.P')) for i in range(3)]
    
    return im1, im2, points2D, points3D, corr, P