from typing import Dict, Any

from gp_detection.exceptions import UnsupportedSource
from gp_detection.gepsilon_model import GepsilonModel
from gp_detection.models.detection_output import DetectionOutput, Detection, LocalImageSource, Box, GepsilonSource
from gp_detection.models.gepsilon_model_descriptor import GepsilonModelDescriptor
from gp_detection.models.parameter_descriptor import ParameterDescriptor
from gp_utils import flat_map
from gp_utils.validators import SkipValidation


class GepsilonYoloV8(GepsilonModel):
    @staticmethod
    def model_name():
        return 'yolo_v8'

    def __init__(self):
        super().__init__(
            GepsilonModelDescriptor(
                name=GepsilonYoloV8.model_name(),
                parameters=[
                    ParameterDescriptor(name='weights', description='', validator=SkipValidation, required=True,
                                        default_value='yolov5n.pt')
                ]
            )
        )

    def predict(self, source: GepsilonSource, params: Dict[str, Any]) -> DetectionOutput:
        from ultralytics import YOLO
        weights = self.resolve('weights', params)
        model = YOLO(weights)

        data_set = ''
        if isinstance(source, LocalImageSource):
            data_set = source.img_path
        else:
            raise UnsupportedSource()
        results = model(data_set)

        predictions = flat_map(lambda x: x.boxes.data, results)
        return DetectionOutput(
            detections=[Detection(
                source=LocalImageSource(img_path=data_set),
                confidence=prediction.cpu().numpy()[4],
                label=model.names[prediction.cpu().numpy()[5]],
                box=Box.from_xyxy(
                    [prediction.cpu().numpy()[0],
                     prediction.cpu().numpy()[1],
                     prediction.cpu().numpy()[2],
                     prediction.cpu().numpy()[3]])
            ) for prediction in predictions]
        )

    def train(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self

    def eval(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self
