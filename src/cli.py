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


def main(weights, source, model):
    if True:
        repository = create_model_repository()
        gepsilon_model = repository.resolve(model)
        output = gepsilon_model.predict(LocalImageSource(path.abspath(source)), {
            'weights': path.abspath(f'../checkpoints/{weights}')
        })
        output.show()
        return output


def parser():
    parser = argparse.ArgumentParser()
    #  parser.add_argument('--mode', type=str, default='predict', help='')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='')
    parser.add_argument('--source', type=str, default='gettyimages-1214430325.jpg', help='')
    parser.add_argument('--model', type=str, default='yolo_v7', help='')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parser()
    main(**vars(opt))

