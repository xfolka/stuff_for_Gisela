from ultralytics import YOLO
from ultralytics.engine.results import Results

from PIL import Image
import numpy as np
import os
import dask as da
from dask.array import image
#import dask.bag as db
from dask import array
from pathlib import Path
import random
from functools import partial
from threading import Lock
from shapely.affinity import translate
import threading
from timeit import default_timer as timer

model_mutex = threading.Lock()
#all_coords_mutex = threading.Lock()
#overlaps_mutex = threading.Lock()

random.seed()

IMG_SIZE = 512
OVERLAP = 2

import threading
class IntGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.cnt : np.uint32 = 100

    def __iter__(self): return self

    def getNext(self):
        self.lock.acquire()
        try:
            self.cnt += 1
            return self.cnt
        finally:
            self.lock.release()

intGen = IntGenerator()

def merge_border_segments(data, block_id, img_size, scan_vertical, border_distance):

    print(f"computing chunk {block_id}")
    local_coords_mod = (0,0,0)
    neighbour_coords_mod = (0,0,0)
    if scan_vertical: 
        if data.shape[0] <= img_size:
            return data
        else:
            local_coords_mod = (-(border_distance + 1),0,0)
            neighbour_coords_mod =  (border_distance,0,0)
            x = img_size
    else:
        if data.shape[1] <= img_size:
            return data
        else:
            local_coords_mod = (0,-1,0)
            neighbour_coords_mod =  (0,border_distance,0)
            y = img_size
    
    for coord in range(img_size):
        if scan_vertical:
            y = coord 
        else:
            x = coord

        local_indices     = (x + local_coords_mod[0],     y + local_coords_mod[1],     0 + local_coords_mod[2])
        neighbour_indices = (x + neighbour_coords_mod[0], y + neighbour_coords_mod[1], 0 + neighbour_coords_mod[2])
        id_local = data[local_indices]
        id_neighbour =  data[neighbour_indices]
        if  id_local != 0 and id_neighbour != 0 and id_neighbour != id_local:
            print(f"merging with id: {id_local} {id_neighbour} {block_id} {scan_vertical}")
            idxs = np.where(data == id_local)
            data[idxs] = id_neighbour

    return data


@da.delayed
def segment_with_yolo(model, data):
    
    results = model.predict(source=np.ascontiguousarray(data), imgsz=512,show_boxes=False,show_labels=False, verbose=False)
    return results

def segment_wrapper(model, out_path, data, block_info=None):
    with model_mutex:
        result = segment_with_yolo(model,data)
        computed_result = result.compute()
                
    if computed_result is None or computed_result[0].masks is None:
        return np.zeros(shape=(512,512,1), dtype=np.uint32)
    
    result_masks = computed_result[0].masks
    masks = result_masks.data.cpu().numpy()
    shape = computed_result[0].masks.shape
    
    sh1 = shape[1]
    sh2 = shape[2]
    all_masks = np.zeros(shape=(512,512,1), dtype=np.uint32)
    tmp_id = intGen.getNext()

    for n in range(shape[0]):
        mask = masks[n,:,:] * intGen.getNext()# random.randint(1,255)
        #mask = masks[n,:,:] * tmp_id# random.randint(1,255)
        mask = np.expand_dims(mask,axis=2)
        mask = mask.astype(np.uint32)
        if shape[1] != 512 or shape[2] != 512: 
            all_masks[:sh1, :sh2,:] = np.where(all_masks[:sh1, :sh2,:] == 0, mask, all_masks[:sh1, :sh2,:])
        else:
            all_masks = np.where(all_masks == 0, mask, all_masks)
        

    return all_masks

base = os.getcwd()
out_data_dir = Path(str(base) + "/output")
Path(out_data_dir).mkdir(parents=True, exist_ok=True)

model = YOLO(str(base) + "/latest_model.pt")

large_image_tmp = da.array.image.imread(str(base) + "/cropped_rgb.png")

s = large_image_tmp.shape
large_image = large_image_tmp.reshape((s[1],s[2],s[3])).rechunk((512,512,3))

bound_f = partial(segment_wrapper, model, str(out_data_dir))
segment_results = large_image.map_blocks(bound_f, dtype=np.uint32,chunks=(512,512,1))

border_distance_to_check = 2

merge_horizontal = partial(merge_border_segments,img_size = IMG_SIZE, scan_vertical = False, border_distance = border_distance_to_check)
horizontal_result = segment_results.map_overlap(merge_horizontal,dtype=np.uint32,depth={0: (0,IMG_SIZE // 2),1: (0,IMG_SIZE // 2)}, boundary=None)

merge_vertical = partial(merge_border_segments,img_size = IMG_SIZE, scan_vertical = True, border_distance = border_distance_to_check)
combined_result = horizontal_result.map_overlap(merge_vertical,dtype=np.uint32,depth={0: (0,IMG_SIZE // 2),1: (0,IMG_SIZE // 2)}, boundary=None)

print("starting...")
start = timer()

#result = segment_results.compute(scheduler='single-threaded')
result = combined_result.compute(scheduler='single-threaded')

end = timer()
print("stopping: ",end - start)
save_im = Image.fromarray(result[:,:,0])
save_im.save("result_mask.png")

#save_im = Image.fromarray(list_of_all_coords.coords)
#save_im.save("result_mask_new.png")
