from ultralytics import YOLO
import config
import os
import glob
from pathlib import Path
import random
random.seed()
#TODO: read path from confg? And if it does not exist, do magic below
file_dir = Path(__file__).parent.resolve()
base = file_dir
#base = os.getcwd()
img_dir = f"{config.IMG_SIZE}x{config.IMG_SIZE}"
test_data_path = str(base) + "/" + config.TEST_DATA_DIR
test_images = glob.glob(test_data_path + "/" + img_dir + "/*.png")

test_image = random.choice(test_images) 

model = YOLO(str(base) + "/" + config.MODEL_SAVE_FILE_NAME)

results = model(test_image,task="segment", show_boxes=False,show_labels=False,imgsz=config.IMG_SIZE)
segments = results[0].masks
results[0].show()
results[0].save(str(base) + "/" + config.TEST_IMAGE_RESULT_FILE_NAME,)
