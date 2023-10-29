from typing import Dict, Any

from gp_detection.exceptions import UnsupportedSource
from gp_detection.gepsilon_model import GepsilonModel
from gp_detection.models.detection_output import DetectionOutput, Detection, LocalImageSource, Box, GepsilonSource
from gp_detection.models.gepsilon_model_descriptor import GepsilonModelDescriptor
from gp_detection.models.parameter_descriptor import ParameterDescriptor
from gp_utils.validators import SkipValidation

CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']


class GepsilonYoloV6(GepsilonModel):
    @staticmethod
    def model_name():
        return 'yolo_v6'

    def __init__(self):
        super().__init__(
            GepsilonModelDescriptor(
                name=GepsilonYoloV6.model_name(),
                parameters=[
                    ParameterDescriptor(name='weights', description='', validator=SkipValidation, required=True,
                                        default_value='yolov6n.pt')
                ]
            )
        )

    def predict(self, source: GepsilonSource, params: Dict[str, Any]) -> DetectionOutput:
        import torch
        torch.set_grad_enabled(False)

        weights = self.resolve('weights', params)
        model = torch.hub.load("meituan/YOLOv6", 'custom',
                               ckpt_path=weights,
                               class_names=CLASS_NAMES,
                               force_reload=True,
                               trust_repo=True)  # or yolov5n - yolov5x6, custom

        data_set = ''
        if isinstance(source, LocalImageSource):
            data_set = source.img_path
        else:
            raise UnsupportedSource()

        results = model.predict(data_set)

        return DetectionOutput(
            detections=[Detection(
                source=LocalImageSource(img_path=data_set),
                confidence=confidence,
                label=label,
                box=Box.from_xyxy(box)
            ) for box, confidence, label in zip(results['boxes'], results['scores'], results['classes'])]
        )

    def train(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self

    def eval(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self
