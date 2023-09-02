from ultralytics import YOLO


def inference(chosen_model, data_set):
    if chosen_model == "YOLO":
        model = YOLO('yolov8n.pt')
        results = model(data_set, save=True)
        return results
