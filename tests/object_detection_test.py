import unittest
from os import path
from unittest.mock import MagicMock

from src.gp_detection.exceptions import UnknownModel
from src.gp_detection.model_repository import ModelRepository
from src.gp_detection.models.detection_output import DetectionOutput, LocalImageSource
from src.gp_models.gepsilon_yolo_v5 import GepsilonYoloV5
from src.gp_models.gepsilon_yolo_v6 import GepsilonYoloV6
from src.gp_models.gepsilon_yolo_v7 import GepsilonYoloV7
from src.gp_models.gepsilon_yolo_v8 import GepsilonYoloV8

MOCK_MODEL_NAME = 'MOCK_MODEL_NAME'
UNKNOWN_MODEL_NAME = 'UNKNOWN_MODEL_NAME'
YOLO_V8 = GepsilonYoloV8.model_name()
YOLO_V7 = GepsilonYoloV7.model_name()
YOLO_V6 = GepsilonYoloV6.model_name()
YOLO_V5 = GepsilonYoloV5.model_name()
TEST_IMAGE = path.abspath('data_sets/image3.jpg')


class TestObjectDetection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestObjectDetection, self).__init__(*args, **kwargs)
        model_resolver = ModelRepository()
        self.gepsilon_model = MagicMock()
        self.gepsilon_model.descriptor.name = MOCK_MODEL_NAME
        self.gepsilon_model.run.return_value = DetectionOutput(
            detections=[]
        )
        model_resolver.register(self.gepsilon_model)
        model_resolver.register(GepsilonYoloV8())
        model_resolver.register(GepsilonYoloV7())
        model_resolver.register(GepsilonYoloV6())
        model_resolver.register(GepsilonYoloV5())
        self.model_resolver = model_resolver

    def test_detect_returns_inference_output(self):
        gepsilon_model = self.model_resolver.resolve(MOCK_MODEL_NAME)

        self.assertEqual(self.gepsilon_model, gepsilon_model)

    def test_detect_call_unknown_model(self):
        with self.assertRaises(UnknownModel):
            self.model_resolver.resolve(UNKNOWN_MODEL_NAME)

    def test_detect_yolov8(self):
        gepsilon_model = self.model_resolver.resolve(YOLO_V8)
        output = gepsilon_model.predict(LocalImageSource(TEST_IMAGE), {
            'weights': path.abspath('../checkpoints/yolov8n.pt')
        })
        self.assertEqual(2, len(output.detections))

    def test_detect_yolov5(self):
        gepsilon_model = self.model_resolver.resolve(YOLO_V5)
        output = gepsilon_model.predict(LocalImageSource(TEST_IMAGE), {
            'weights': path.abspath('../checkpoints/yolov5n.pt')
        })
        print(output.detections)

        self.assertEqual(2, len(output.detections))

    def test_detect_yolov7(self):
        gepsilon_model = self.model_resolver.resolve(YOLO_V7)
        output = gepsilon_model.predict(LocalImageSource(TEST_IMAGE), {
            'weights': path.abspath('../checkpoints/yolov7.pt')
        })

        self.assertEqual(2, len(output.detections))

    def test_detect_yolov6(self):
        gepsilon_model = self.model_resolver.resolve(YOLO_V6)
        output = gepsilon_model.predict(LocalImageSource(TEST_IMAGE), {
            'weights': path.abspath('../checkpoints/yolov6n.pt')
        })

        self.assertEqual(2, len(output.detections))


if __name__ == '__main__':
    unittest.main()
