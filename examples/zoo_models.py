# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pytouch import PyTouchZoo, sensors


def main():
    pytouch_zoo = PyTouchZoo()

    # list available pytouch zoo models
    available_models = pytouch_zoo.list_models()
    print(available_models)

    # load DIGIT sensor touch detect model from pytouch zoo
    touch_detect_model = pytouch_zoo.load_model_from_zoo(  # noqa: F841
        "touchdetect_resnet", sensors.DigitSensor
    )

    # load custom PyTorch-Lightning saved model
    custom_model = pytouch_zoo.load_model("/path/to/pl/model")  # noqa: F841

    # create custom onnx session for inference
    custom_onnx = pytouch_zoo.load_onnx_session("/path/to/onnx/model")  # noqa: F841


if __name__ == "__main__":
    main()
