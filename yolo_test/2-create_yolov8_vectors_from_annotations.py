from pathlib import Path
import os
import glob
from PIL import Image
from skimage import measure
import numpy as np
from pathlib import Path
import scipy.spatial as ssp
import config


path = str(os.getcwd()) + "/dl/"
annot_dl_path = path + "annotations/"
vector_path = path + "vectors/"
annotations = glob.glob(annot_dl_path+"/*.*")
Path(vector_path).mkdir(parents=True, exist_ok=True)

old_vectors = glob.glob(vector_path+"/*.*")
for v in old_vectors:
    os.remove(v)


for ann in annotations:
    im = np.asarray(Image.open(ann)).swapaxes(0,1)

    props = measure.regionprops(label_image=im)
    with open(vector_path + str(Path(ann).stem) + '.txt',"x") as targets_file:
        for prop in props:
            coords = prop.coords
            if prop.area < 100:
                continue
            ch = ssp.ConvexHull(coords)
            points = ch.points[ch.vertices].astype(np.uint16)
            points = np.divide(points,config.IMG_SIZE)
            points = np.append(points,points[0]) #add the first point as the last to make a complete polygon
            targets_file.write("0 ")
            np.savetxt(targets_file,points, newline=' ', fmt='%1.3f')
            targets_file.write("\n")
