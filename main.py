import os
from pathlib import Path


#  import super_gradients
#  model = models.get("model-name", pretrained_weights="pretrained-model-name")
#  error when trying to install YOLONAS : Failed building wheel for pycocotools


def inference(chosen_model, data_set):
    if chosen_model == "YOLOv8":
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(data_set, save=True)
        return results
    if chosen_model == "YOLOv5":
        from Models_zoo.yolov5.detect import run
        run(source=data_set)

    if chosen_model == "YOLOv6":
        from Models_zoo.YOLOv6.tools.infer import run
        os.chdir(f'{Path(__file__).parent}/YOLOv6')
        print(os.getcwd())
        run(source=data_set)

    if chosen_model == "YOLOv7":
        from Models_zoo.yolov7.detect import detect
        detect()

    if chosen_model == "YOLOvR":
        os.chdir('C:\object_detection_lib\yolor')
        os.system(f'python detect.py --source {data_set} --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt')
        os.chdir('C:\object_detection_lib')
        #  need a save in custom file
        #  work only in cuda env

inference('YOLOv7','m')



