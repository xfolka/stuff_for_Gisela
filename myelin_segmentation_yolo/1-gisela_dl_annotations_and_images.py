from PIL import Image
import numpy as np
import webknossos as wk
from webknossos_utils import Pixel_size
from skimage import measure

import dask.array as da

from pathlib import Path
import os
import glob

import config

#SHOW_IMAGES = False
#CLEAR_OUTPUT_DIR = False


#AUTH_TOKEN = "S-QRDIegZYX0IM1lXmyiJg" #2024-07-04
#WK_TIMEOUT="3600" # in seconds
#ORG_ID = "83d574429f8bc523" # gisela's webknossos

id_1 = "6644c04d0100004a01fa11af"
id_2 = "664316880100008a049e890e"
id_3 = "664606440100005102550210"

wk_id_list = [id_1,id_2,id_3]

# img_size = 512
# min_labels_per_image = 2
# min_coverage = 0.2

if config.SHOW_IMAGES:
    import matplotlib.pyplot as plt

path = str(os.getcwd()) + "/dl/"
img_dl_path = path + "images/"
annot_dl_path = path + "annotations/"

Path(path).mkdir(parents=True, exist_ok=True)
Path(img_dl_path).mkdir(parents=True, exist_ok=True)
Path(annot_dl_path).mkdir(parents=True, exist_ok=True)
if config.CLEAR_OUTPUT_DIR:
    files = glob.glob(img_dl_path+"/*.*")
    files += glob.glob(annot_dl_path+"/*.*")
    for f in files:
        os.remove(f)

for wkid in wk_id_list:

    ANNOTATION_ID = wkid

    with wk.webknossos_context(token=config.WK_TOKEN):
        annotations = wk.Annotation.open_as_remote_dataset(annotation_id_or_url=ANNOTATION_ID)
        lbl_layers = annotations.get_segmentation_layers()

        label_indices = {i.name : l for l,i in enumerate(lbl_layers)}

        DATASET_NAME = annotations._properties.id['name']
        ds = wk.Dataset.open_remote(dataset_name_or_url=DATASET_NAME, organization_id=config.ORG_ID)
        img_layer = ds.get_color_layers()
        assert len(img_layer) == 1, "more than an image, this is unexpected for this project"
        img_layer = img_layer[0]    

        voxel_size = ds.voxel_size
        mag_list = list(img_layer.mags.keys())
        
        MAG = mag_list[3]
        pSize = Pixel_size(voxel_size[0] * MAG.x, voxel_size[1] * MAG.y, voxel_size[2] * MAG.z, MAG=MAG, unit="nm")

        img_data = img_layer.get_mag(pSize.MAG).read()
        lbl_data = lbl_layers[label_indices["Myelin"]].get_mag(pSize.MAG).read()

    unique_lbls = np.unique(lbl_data)

    if np.max(unique_lbls) < 512:
        lbl_data = lbl_data.astype(np.uint8)

    lbl_dask = da.from_array(np.swapaxes(lbl_data,-1,-3), chunks=(1,5,512,512))

    segmentation = np.nonzero(lbl_dask[0,0])

    bbox = 0, 0, 0, 0
    #if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    #construct WK bbox from large_bbox
    wk_bbox = wk.BoundingBox(topleft=(x_min,y_min,0), size=(x_max-x_min,y_max-y_min,1))
    wk_bbox = wk_bbox.align_with_mag(pSize.MAG,ceil=True)
    wk_bbox = wk_bbox.from_mag_to_mag1(pSize.MAG)

    with wk.webknossos_context():
        img_data_small = img_layer.get_finest_mag().read(absolute_bounding_box=wk_bbox)
        lbl_data_small = lbl_layers[label_indices["Myelin"]].get_finest_mag().read(absolute_bounding_box=wk_bbox)
    
    bx = wk_bbox.topleft[1]
    by = wk_bbox.topleft[0]
    bw = wk_bbox.size.x
    bh = wk_bbox.size.y

    lbl_dask_small = da.from_array(np.swapaxes(lbl_data_small,-1,-3), chunks=(1,2,512,512))
    img_dask_small = da.from_array(np.swapaxes(img_data_small,-1,-3), chunks=(1,2,512,512))

    if config.SHOW_IMAGES:
        from matplotlib.patches import Rectangle
        ax = plt.gca()
    
    properties = ['label', 'bbox', 'centroid']

    im_size = config.IMG_SIZE
    img_x_div = bw // im_size
    img_y_div = bh // im_size
    lbl_dask_cropped = lbl_dask_small[0,0,0:img_y_div*im_size,0:img_x_div*im_size].compute()
    img_dask_cropped = img_dask_small[0,0,0:img_y_div*im_size,0:img_x_div*im_size].compute()

    idx = 0
    for y in range(img_y_div):
        for x in range(img_x_div):
            print(f"x: {x}, y: {y}")
            start_x = x * im_size
            start_y = y * im_size
            end_x = start_x + im_size
            end_y = start_y + im_size
            active_lbl_chunk = lbl_dask_cropped[start_y:end_y,start_x:end_x]
            active_img_chunk = img_dask_cropped[start_y:end_y,start_x:end_x]

            regions = measure.regionprops(label_image=active_lbl_chunk)
##            reg_table = pd.DataFrame(reg_table)
            tot_elems = len(regions)

            if tot_elems == 0:
                continue 

            im = Image.fromarray(active_img_chunk.astype(np.uint8))
            im.save(img_dl_path + str(ANNOTATION_ID) + "_" + str(idx) + '.png')

            ann = Image.fromarray(active_lbl_chunk.astype(np.uint8))
            ann.save(annot_dl_path + str(ANNOTATION_ID) + "_" + str(idx) + '.png')
            idx+=1

print("Done!")

