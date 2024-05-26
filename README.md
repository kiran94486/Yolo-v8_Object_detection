## 1) The first step is to convert the data from pascal to yolo
## python pascalVOC_to_yolo.py data yolo_annotations


## 2) Train YOLOv8 for Person Detection:
## Train YOLOv8 for Person Detection:

## 3) Running inferance script
## python inference.py --input_dir data/test/images --output_dir output --person_det_model weights/person_model.pt --ppe_detection_model weights/ppe_model.pt


## the model is trained and the output is present inside data\runs\detect\train with all the graphs and the detections based on the classes.txt i have run 25 epochs 

## First i have tried it in colab due to gpu issue after that i converted it into modular coding format.

## Be sure to check the project
## And please provide me the feedback :)