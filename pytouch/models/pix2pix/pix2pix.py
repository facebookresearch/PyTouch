# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from PIL import Image

from pytouch.models.pix2pix.thirdparty.pix2pix.data.base_dataset import (
    get_params,
    get_transform,
)
from pytouch.models.pix2pix.thirdparty.pix2pix.models.pix2pix_model import (
    Pix2PixModel as Pix2PixBaseModel,
)
from pytouch.utils.data_utils import interpolate_img


@dataclass
class Pix2PixModelBaseParams:
    aspect_ratio: float = 1.0
    crop_size: int = 256
    dataroot = None
    display_winsize: int = 256
    epoch: str = "latest"
    eval: bool = False
    gpu_ids: List[int] = field(default_factory=list)
    init_gain: float = 0.02
    init_type: str = "normal"
    input_nc: int = 3
    isTrain: bool = False
    load_iter: int = 0
    load_size: int = 256
    max_dataset_size: float = float("inf")
    n_layers_D: int = 3
    ndf: int = 64
    netD: str = "basic"
    netG: str = "unet_256"
    ngf: int = 64
    no_dropout: bool = False
    norm: str = "batch"
    num_test: int = 50
    output_nc: int = 3
    phase: str = "test"
    preprocess: str = "resize_and_crop"
    results_dir: str = "./results/"
    suffix: str = ""
    verbose: bool = False

    def __post_init__(self):
        self.gpu_ids = [0] if torch.cuda.is_available() else None


@dataclass
class Pix2PixModelParams(Pix2PixModelBaseParams):
    batch_size: int = 1
    checkpoints_dir: str = "local/pix2pix/checkpoints"
    direction: str = "AtoB"
    dataset_mode: str = "aligned"
    model: str = "pix2pix"
    name: str = "experiment_name"
    no_flip: bool = True
    num_threads: int = 0
    serial_batches: bool = True
    display_id: int = -1


class Pix2PixModel:
    def __init__(self, name, model_type, dataset_mode, direction, model_dir=""):
        self.name = name
        self.model_params = Pix2PixModelParams()
        self.model_params.name = name
        self.model_params.model = model_type
        self.model_params.dataset_mode = dataset_mode
        self.model_params.direction = direction
        self.model_params.checkpoints_dir = model_dir
        self.model = Pix2PixBaseModel(self.model_params)
        if model_dir:
            self.model.setup(self.model_params)

    def init_zoo_model(self, state_dict):
        net = self.model.netG
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def color_to_normal(self, img_input):
        # preprocess inputs
        img = self.preprocess_image(img_input)
        model_img_input = self._create_model_input(img)

        # call model
        self.model.set_input(model_img_input)
        self.model.test()
        output = self.model.get_current_visuals()

        # post process model output
        img_normal = ((output["fake_B"]).squeeze(0) + 1) / 2.0
        img_normal = interpolate_img(img=img_normal, rows=160, cols=120)

        return img_normal

    def preprocess_image(self, img_input):
        img = (
            (img_input * 255.0)
            .permute(1, 2, 0)
            .to("cpu")
            .float()
            .numpy()
            .astype(np.uint8)
        )
        img = Image.fromarray(img)
        img = self.transform(img)
        return img

    def _create_model_input(self, img_input):
        img = img_input.unsqueeze(0)  # B x C x H x W
        model_input = {}
        model_input["A"], model_input["B"] = img, img
        model_input["A_paths"], model_input["B_paths"] = "", ""
        return model_input

    def transform(self, img_input):
        transform_params = get_params(self.model_params, img_input.size)
        transforms = get_transform(self.model_params, transform_params, grayscale=False)
        return transforms(img_input)
