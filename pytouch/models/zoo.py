# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import sys
from urllib.parse import urljoin, urlparse

import boto3
import botocore
import onnx
import onnxruntime
import torch
import torch.hub as hub
from botocore.client import Config

from pytouch.utils import model_utils


class PyTouchZooModelNotFound(Exception):
    """Raised when a PyTouch Zoo model cannot be located"""

    pass


class ZooConfig:
    SERVICE_NAME = "s3"
    REGION_NAME = "us-east-2"
    BUCKET_NAME = "pytouch-zoo"
    SIG_VERSION = botocore.UNSIGNED

    @classmethod
    def generate_download_url(cls, model_file):
        base_url = f"http://{cls.BUCKET_NAME}.{cls.SERVICE_NAME}.{cls.REGION_NAME}.amazonaws.com"
        return urljoin(base_url, model_file)


class PyTouchZoo:
    def __init__(
        self,
        service=ZooConfig.SERVICE_NAME,
        region=ZooConfig.REGION_NAME,
        bucket=ZooConfig.BUCKET_NAME,
    ):
        self.service = service
        self.region = region
        self.bucket = bucket
        self.client = boto3.client(
            self.service,
            region_name=self.region,
            config=Config(signature_version=ZooConfig.SIG_VERSION),
        )
        self.objects = self._get_objects()

    def _get_objects(self):
        objects = self.client.list_objects(Bucket=self.bucket)
        return objects["Contents"]

    def _get_zoo_model_url(self, model_name, sensor, version=None):
        model_file = f"{model_name}_{sensor}.{version}"
        if not any(model_file in obj["Key"] for obj in self.objects):
            raise PyTouchZooModelNotFound(f"cannot find model {model_file}")
        return ZooConfig.generate_download_url(model_file)

    def list_models(self):
        return [obj["Key"] for obj in self.objects]

    def load_model_from_zoo(self, model_name, sensor, version="pth"):
        model_url = self._get_zoo_model_url(model_name, sensor.zoo_name(), version)
        model_state_dict = hub.load_state_dict_from_url(model_url)
        model_state_dict = model_utils.convert_state_dict_if_from_pl(model_state_dict)
        return model_state_dict

    def load_onnx_from_zoo(self, model_name, sensor, version="onnx"):
        model_url = self._get_zoo_model_url(model_name, sensor.name, version)
        hub_dir = os.path.join(hub.get_dir(), "checkpoints")

        if not os.path.isfile(hub_dir):
            os.makedirs(hub_dir)

        # from https://github.com/pytorch/pytorch/blob/master/torch/hub.py
        parts = urlparse(model_url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(hub_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(model_url, cached_file))
            hub.download_url_to_file(model_url, cached_file, None, progress=True)

        return self.load_onnx_session(cached_file)

    @staticmethod
    def load_model(model_path):
        saved_model = torch.load(model_path)
        saved_model = model_utils.convert_state_dict_if_from_pl(saved_model)
        return saved_model

    @staticmethod
    def load_onnx_session(model_path):
        saved_model = onnx.load(model_path)
        onnx.checker.check_model(saved_model)
        session = onnxruntime.InferenceSession(saved_model, None)
        return session
