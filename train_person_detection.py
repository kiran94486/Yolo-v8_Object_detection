import os

def train_person_detection():
   
    os.chdir('E:\Computer Vision Intern\project')

    
    os.system('yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=640 plots=True')

if __name__ == '__main__':
    train_person_detection()
