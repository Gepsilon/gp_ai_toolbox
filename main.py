from ultralytics import YOLO
import os
#  import super_gradients
#  model = models.get("model-name", pretrained_weights="pretrained-model-name")
#  error when trying to install YOLONAS : Failed building wheel for pycocotools


def inference(chosen_model, data_set):
    if chosen_model == "YOLOv8":
        model = YOLO('yolov8n.pt')
        results = model(data_set, save=True)
        return results
    if chosen_model == "YOLOv5":
        os.system(f'python yolov5\detect.py --source={data_set}')
        #  need a save in custom file

    if chosen_model == "YOLOv6":
        os.chdir('C:\object_detection_lib\YOLOv6')
        os.system(f'python tools/infer.py --weights yolov6s.pt --source {data_set}')
        os.chdir('C:\object_detection_lib')
        #  need a save in custom file

    if chosen_model == "YOLOv7":
        os.chdir('C:\object_detection_lib\yolov7')
        os.system(f'python detect.py --weights yolov7.pt --source {data_set}')
        os.chdir('C:\object_detection_lib')
        #  need a save in custom file

    if chosen_model == "YOLOvR":
        os.chdir('C:\object_detection_lib\yolor')
        os.system(f'python detect.py --source {data_set} --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt')
        os.chdir('C:\object_detection_lib')
        #  need a save in custom file
        #  work only in cuda env





