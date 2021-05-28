# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

_log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train")
def main(cfg):
    _log.info("PyTouch training initialized with the following configuration...")
    _log.info(OmegaConf.to_yaml(cfg))

    _log.info(f"Dataset parameters: {OmegaConf.to_yaml(cfg.data)}")
    _log.info(f"Model parameters: {OmegaConf.to_yaml(cfg.model)}")
    _log.info(f"Training paramters: {OmegaConf.to_yaml(cfg.training)}")
    _log.info(f"Optimizer paramters: {OmegaConf.to_yaml(cfg.optimizer)}")

    pl.seed_everything(cfg.training.seed)

    # # Instantiate objects from config file
    task_model = instantiate(cfg.model, cfg)
    task_data_module = instantiate(cfg.data, cfg)

    checkpoint_filename = cfg.experiment + "-{epoch}_{val_loss:.3f}_{val_acc:.3f}"

    _log.info(
        f"Creating model checkpoint monitoring {cfg.checkpoints.monitor}, mode: {cfg.checkpoints.mode}"
    )
    _log.info(f"Saving top {cfg.checkpoints.save_top_k} checkpoints!")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoints.path,
        filename=checkpoint_filename,
        save_top_k=cfg.checkpoints.save_top_k,
        verbose=cfg.general.verbose,
        monitor=cfg.checkpoints.monitor,
        mode=cfg.checkpoints.mode,
        save_weights_only=cfg.checkpoints.save_weights_only,
    )

    logger = TensorBoardLogger(cfg.general.tb_log_path, name=cfg.experiment)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.training.n_epochs,
        callbacks=[checkpoint_callback],
        gpus=1,
        default_root_dir=".",
    )

    trainer.fit(task_model, task_data_module)

    _log.info(
        f"Best Checkpoint: {checkpoint_callback.best_model_score} -- {checkpoint_callback.best_model_path}"
    )
    _log.info(f"Training completed for experiment: {cfg.experiment}")

    if cfg.onnx_export:
        _log.info("Exporting to ONNX")
        best_model = task_model.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        input_sample = torch.randn(1, 3, 64, 64)
        onnx_filename = os.path.basename(checkpoint_callback.best_model_path)
        best_model.to_onnx(onnx_filename, input_sample, export_params=True)


if __name__ == "__main__":
    sys.exit(main())
