

## Requirements
use conda env yolo?

## How to run
### For training
- from cli: `yolo segment train data=datasets/dataset.yaml epochs=50 imgsz=512`
where  "data" is the path to your dataset .yaml file and epochs are the number of epochs you want to train for. "imgsz" is the dimension of the images in the set.


### For segmentation of single image:

 - from cli: `yolo segment predict model=runs/segment/train5/weights/best.pt show_labels=false show_boxes=false source=dl/test_data/t1-15.png`
 where "model" is the patch to a trained model (output from training in previous step), "source" is the path to the image you want to segment. "show_boxes" and "show_labels" determine if you want the segmented image to contain boxes and labels around the segmentations.

### Generate testdata from a large picture
 - use the convert command from imagemagick (linux) to split a large image into many smaller ones like this:
 `convert -crop 1024x1024 664316880100008a049e890e_part_0.png t1.png`
 This will split the file 664316880100008a049e890e_part_0.png into chunks of 1024x1024 and name them t1-0.png, t1-1.png,...,t1-n.png

## Caveats
Absolute path needs to be used in dataset.yaml as path parameter:

```
path: /home/xfolka/Projects/stuff_for_Gisela/myelin_segmentation_yolo/datasets
train: ./images/train/ 
val: ./images/val/ 
 
names: 
  0: myelin_sheet
```