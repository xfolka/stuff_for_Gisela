from pathlib import Path
import os
import glob
import shutil

dataset_path = str(os.getcwd()) + "/datasets/"
img_train_path = dataset_path + "images/train/"
img_val_path = dataset_path + "images/val/"
labels_train_path = dataset_path + "labels/train/"
labels_val_path = dataset_path + "labels/val/"

Path(dataset_path).mkdir(parents=True, exist_ok=True)
Path(img_train_path).mkdir(parents=True, exist_ok=True)
Path(img_val_path).mkdir(parents=True, exist_ok=True)
Path(labels_train_path).mkdir(parents=True, exist_ok=True)
Path(labels_val_path).mkdir(parents=True, exist_ok=True)

files = glob.glob(img_train_path+"/*.png")
files += glob.glob(img_val_path+"/*.png")
files += (glob.glob(labels_train_path+"/*.txt"))
files += (glob.glob(labels_val_path+"/*.txt"))
for f in files:
    os.remove(f)
    
dl_path = str(os.getcwd()) + "/dl/"
img_dl_path = dl_path + "images/"
labels_dl_path = dl_path + "vectors/"

#copy the files
images = glob.glob(img_dl_path+"/*.png")
label_vectors = (glob.glob(labels_dl_path+"/*.txt")) 

for idx,img in enumerate(images):
    if idx % 2 == 0:
        shutil.copy2(img,img_train_path)
    else:
        shutil.copy2(img,img_val_path)

for idx,lbl in enumerate(label_vectors):
    if idx % 2 == 0:
        shutil.copy2(lbl,labels_train_path)
    else:
        shutil.copy2(lbl,labels_val_path)
        
#create the dataset file it it does not exist already
dataset_file = Path(dataset_path + "dataset.yaml")
if dataset_file.is_file():
    import sys
    sys.exit()
    
dataset_file.open(mode="x")
dataset_file.write_text("train: ./images/train/\nval: ./images/val/\n\nnames:\n  0: myelin_sheet")





