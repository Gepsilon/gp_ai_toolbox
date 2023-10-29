import argparse
from os import path

from gp_detection.model_repository import ModelRepository
from gp_detection.models.detection_output import LocalImageSource
from gp_models.gepsilon_yolo_v5 import GepsilonYoloV5
from gp_models.gepsilon_yolo_v6 import GepsilonYoloV6
from gp_models.gepsilon_yolo_v7 import GepsilonYoloV7
from gp_models.gepsilon_yolo_v8 import GepsilonYoloV8


def create_model_repository():
    model_resolver = ModelRepository()
    model_resolver.register(GepsilonYoloV8())
    model_resolver.register(GepsilonYoloV7())
    model_resolver.register(GepsilonYoloV6())
    model_resolver.register(GepsilonYoloV5())
    return model_resolver


def main(opt):
    print(opt)
    repository = create_model_repository()

    gepsilon_model = repository.resolve('yolo_v5')
    output = gepsilon_model.predict(LocalImageSource(path.abspath('../tests/data_sets/image3.jpg')), {
        'weights': path.abspath('../checkpoints/yolov5n.pt')
    })

    print(output)
    output.show()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: define cli args and use them
    # suggestion of the CLI: cli.py --mode=predict --model=yolo_v8 --source='.....' --weights=...
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--model', type=str, default='yolo_v8', help='file/dir/URL/glob/screen/0(webcam)')
    opt = parser.parse_args()

    main(opt)
