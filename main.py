import sys, os


def is_python_module(path):
    return os.path.isfile(os.path.join(path, '__init__.py'))


def load_python_modules(path):
    [sys.path.append(os.path.abspath(x[0])) for x in os.walk(path) if is_python_module(x[0])]


def inference(chosen_model, data_set):
    if chosen_model == "YOLOv8":
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(data_set, save=True)
        return results
    if chosen_model == "YOLOv5":
        from yolov5.detect import run
        run()

    if chosen_model == "YOLOv6":
        yolo_v6_path = 'YOLOv6'
        sys.path.append(os.path.abspath(yolo_v6_path))
        load_python_modules(yolo_v6_path)
        from YOLOv6.tools import infer
        os.chdir(yolo_v6_path)
        infer.run(weights='yolov6s.pt', yaml='D:\object_detection_lib\YOLOv6\data\dataset.yaml', source=data_set,
                  hide_labels=True)

    if chosen_model == "YOLOv7":
        yolo_v7_path = 'YOLOv7'
        sys.path.append(os.path.abspath(yolo_v7_path))
        load_python_modules(yolo_v7_path)
        os.chdir(yolo_v7_path)
        os.system(f'python detect.py --weights yolov7.pt ')


if __name__ == '__main__':
    inference()

