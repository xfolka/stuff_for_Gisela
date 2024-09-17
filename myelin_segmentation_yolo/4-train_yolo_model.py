from ultralytics import YOLO
import config
import os
from pathlib import Path


#TODO: read path from confg? And if it does not exist, do magic below
file_dir = Path(__file__).parent.resolve()
base = file_dir
#base = os.getcwd()
dataSetFile = str(base) + "/" + config.TRAINING_DATASET_FILE

model = YOLO("yolov8n-seg.pt")
results = model.train(data=dataSetFile,epochs=config.TRAINING_EPOCHS, imgsz=config.IMG_SIZE, show_boxes=False, show_labels=False)
results = model.val()
model.save(str(base) + "/" + config.MODEL_SAVE_FILE_NAME)

