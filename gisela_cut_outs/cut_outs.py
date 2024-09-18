import numpy as np
import webknossos as wk
from webknossos_utils import Pixel_size
from pathlib import Path
import os
import glob
import random
from pyometiff import OMETIFFWriter
import json

import config

def intersects(wk_bbox, list_of_bboxes) -> bool:
    intersects = False
    for b in list_of_bboxes:
        intersects = not b.intersected_with(wk_bbox,dont_assert=True).is_empty() or intersects
        
    return intersects

random.seed()

id_1 = "6644c04d0100004a01fa11af"
id_2 = "664316880100008a049e890e"
id_3 = "664606440100005102550210"

wk_id_list = [id_1,id_2,id_3]

if config.SHOW_IMAGES:
    import matplotlib.pyplot as plt

path = str(os.getcwd()) + "/dl/"
img_dl_path = path + "images/"

Path(path).mkdir(parents=True, exist_ok=True)
Path(img_dl_path).mkdir(parents=True, exist_ok=True)
if config.CLEAR_OUTPUT_DIR:
    files = glob.glob(img_dl_path+"/*.*")
    for f in files:
        os.remove(f)

#for wkid in wk_id_list:

#    ANNOTATION_ID = wkid

with wk.webknossos_context(token=config.WK_TOKEN):
#        annotations = wk.Annotation.open_as_remote_dataset(annotation_id_or_url=ANNOTATION_ID)

    rds = wk.RemoteDataset.get_remote_datasets(config.ORG_ID)
    for entry in rds.entries:
        #lbl_layers = annotations.get_segmentation_layers()

        #label_indices = {i.name : l for l,i in enumerate(lbl_layers)}

        #DATASET_NAME = annotations._properties.id['name']
        print(f"*********************** using dataset: {entry} *********************")
        ds = wk.Dataset.open_remote(dataset_name_or_url=entry, organization_id=config.ORG_ID)
        img_layer = ds.get_color_layers()
        assert len(img_layer) == 1, "more than an image, this is unexpected for this project"
        first_layer = img_layer[0]    

        voxel_size = ds.voxel_size_with_unit
        mag_list = list(first_layer.mags.keys())
        # MAG = mag_list[len(mag_list)-2]
        MAG = mag_list[config.MAG_LIST_INDEX]
        MAG1 = first_layer.get_finest_mag().mag
        # pSize = Pixel_size(voxel_size[0] * MAG.x, voxel_size[1] * MAG.y, voxel_size[2] * MAG.z, MAG=MAG, unit="nm")

        # read down image in manageable size/mag
        img_data = first_layer.get_mag(MAG).read()#.swapaxes(0,2)

        # find the area of the image that contains no "black edges"
        binary_img_data = np.logical_and(img_data,True) #make this a binary image

        # construct a bbox, that is IMG_SIZE (see config) in MAG_1 that fits in the convex hull by:
        # - generate all four points, the first one is "random"
        # - check that all of the points are inside the convex hull
        # - check that none of the points are insde another bbox (i.e. they dont intersect)
        #

        bbox_in_mag_1 = wk.BoundingBox((0,0,0),(config.IMG_SIZE,config.IMG_SIZE,1))
        bbox_aligned_mag = bbox_in_mag_1.align_with_mag(MAG,ceil=True)
        bbox_in_selected_mag = bbox_aligned_mag.in_mag(MAG)
        current_bbox_side = bbox_in_selected_mag.bottomright.x

        bboxes = []
        cnt = 0
        while len(bboxes) < config.NR_OF_CROPS_PER_IMAGE and cnt < config.MAX_ITERATIONS:
            cnt += 1
            s_x = current_bbox_side-1
            s_y = current_bbox_side-1
            s_z = 1

            tl_x = random.randint(0,img_data.shape[1]-current_bbox_side)
            tl_y = random.randint(0,img_data.shape[2]-current_bbox_side)

            ll_x = tl_x
            ll_y = tl_y + s_y

            lr_x = ll_x + s_x
            lr_y = ll_y

            tr_x = lr_x
            tr_y = tl_y

            tl_ok = binary_img_data[0,tl_x,tl_y,0]
            ll_ok = binary_img_data[0,ll_x,ll_y,0]
            lr_ok = binary_img_data[0,lr_x,lr_y,0]
            tr_ok = binary_img_data[0,tr_x,tr_y,0]

            tl_z = 0

            if not tl_ok or not ll_ok or not lr_ok or not tr_ok: #this point is not inside the image region
                continue
            top_left = (tl_x,tl_y,tl_z)
            b_size = (s_x,s_y,s_z)
            the_box = wk.BoundingBox(top_left, b_size)

            if intersects(the_box,bboxes):
                continue

            bboxes.append(the_box)

        from matplotlib import pyplot as plt
        plt.imshow(img_data[0,:,:,0])

        bboxes = [x.from_mag_to_mag1(MAG) for x in bboxes]

        for i,bx in enumerate(bboxes):
            img_cut = first_layer.get_finest_mag().read_bbox(bx).swapaxes(1,-1)
            metadata_dict = {
                "PhysicalSizeX": voxel_size.factor[0],
                "PhysicalSizeXUnit": voxel_size.unit.name,
                "PhysicalSizeY": voxel_size.factor[1],
                "PhysicalSizeYUnit": voxel_size.unit.name,
                "PhysicalSizeZ": voxel_size.factor[2],
                "PhysicalSizeZUnit": voxel_size.unit.name,
                "Channels": {
                    "SEM": {
                        "Name": "BSD",
                        "SamplesPerPixel": 1,
                    },
                },
            }
            
            extra_data = {
                "OriginX" : str(bx.topleft_xyz.x),
                "OriginY" : str(bx.topleft_xyz.y),
            }
            
            dimension_order = "ZYX"
            writer = OMETIFFWriter(
                    fpath=img_dl_path + f"{entry}-{i}.ome.tiff",
                    dimension_order=dimension_order,
                    array=img_cut[0,:,:,:],
                    metadata=metadata_dict,
                    
                    explicit_tiffdata=False)
            writer.write()
            f = open(img_dl_path + f"{entry}-{i}-data.json", "x")
            f.write(json.dumps(extra_data))
            f.close()
            # mdata = {
            #     'PhysicalSizeX' : f"{voxel_size.factor[0]}",
            #     'PhysicalSizeY' : f"{voxel_size.factor[1]}",
            #     'PhysicalSizeUnit' : voxel_size.unit,
            #         'Origin' : f"{bx.topleft_xyz.x},{bx.topleft_xyz.y}"
            # }
            # tiff.write(img_cut, description=mdata)
            # imsh =plt.imshow(img_cut[0,:,:,0])
            # plt.show()

        # print(props[0])
        # with open(vector_path + str(Path(ann).stem) + '.txt',"x") as targets_file:

        # read the bbox from the finest_mag of remote datase

        # img_data = first_layer.get_mag(pSize.MAG).read()

            # lbl_data = lbl_layers[label_indices["Myelin"]].get_mag(pSize.MAG).read()

#    unique_lbls = np.unique(lbl_data)

# if np.max(unique_lbls) < 512:
#     lbl_data = lbl_data.astype(np.uint8)

# lbl_dask = da.from_array(np.swapaxes(lbl_data,-1,-3), chunks=(1,5,512,512))

# segmentation = np.nonzero(lbl_dask[0,0])

# bbox = 0, 0, 0, 0
# #if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
# x_min = int(np.min(segmentation[1]))
# x_max = int(np.max(segmentation[1]))
# y_min = int(np.min(segmentation[0]))
# y_max = int(np.max(segmentation[0]))

# #construct WK bbox from large_bbox
# wk_bbox = wk.BoundingBox(topleft=(x_min,y_min,0), size=(x_max-x_min,y_max-y_min,1))
# wk_bbox = wk_bbox.align_with_mag(pSize.MAG,ceil=True)
# wk_bbox = wk_bbox.from_mag_to_mag1(pSize.MAG)

# with wk.webknossos_context():
#     img_data_small = img_layer.get_finest_mag().read(absolute_bounding_box=wk_bbox)
#     lbl_data_small = lbl_layers[label_indices["Myelin"]].get_finest_mag().read(absolute_bounding_box=wk_bbox)

# bx = wk_bbox.topleft[1]
# by = wk_bbox.topleft[0]
# bw = wk_bbox.size.x
# bh = wk_bbox.size.y

# lbl_dask_small = da.from_array(np.swapaxes(lbl_data_small,-1,-3), chunks=(1,2,512,512))
# img_dask_small = da.from_array(np.swapaxes(img_data_small,-1,-3), chunks=(1,2,512,512))
