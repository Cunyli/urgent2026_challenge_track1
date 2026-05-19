#!/bin/env python

from datetime import datetime
import os
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import  Callback, ModelCheckpoint,LearningRateMonitor
import torch.multiprocessing as mp
from baseline_code.d_model import SEModel
from baseline_code.flow_model import FlowSEModel
from baseline_code.config import Config
from baseline_code.dataset import AudioDataModule
import glob
from baseline_code.config import config_parser


class PruneLatestCheckpoints(Callback):
    def __init__(self, checkpoint_dir=None, keep=3, pattern="latest_*.ckpt"):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.keep = keep
        self.pattern = pattern

    def _prune(self, trainer):
        checkpoint_dir = self.checkpoint_dir or Path(trainer.checkpoint_callback.dirpath)
        checkpoints = sorted(
            checkpoint_dir.glob(self.pattern),
            key=lambda path: path.stat().st_mtime,
        )
        for checkpoint_path in checkpoints[:-self.keep]:
            checkpoint_path.unlink(missing_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._prune(trainer)

    def on_validation_end(self, trainer, pl_module):
        self._prune(trainer)


def prepare_call_backs(cfg, checkpoint_dir=None):

    best_metrics = [
        ('val_loss', 'min'),
        ]
    call_backs = [LearningRateMonitor(logging_interval='epoch')]
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_{epoch:02d}-{step:06d}-{"+ metric + ":.3f}",
            save_top_k=1,
            monitor=metric,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=False,
            save_on_train_epoch_end=False,
        )
        call_backs.append(call_back)

    call_backs.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latest_{epoch:02d}-{step:06d}",
            save_top_k=-1,
            every_n_epochs=getattr(cfg, "checkpoint_every_n_epochs", 1),
            save_weights_only=False,
            save_last=False,
        )
    )
    call_backs.append(PruneLatestCheckpoints(checkpoint_dir, keep=3))

    return call_backs


def _serializable_config(cfg):
    scalar_types = (str, int, float, bool, type(None), list, tuple, dict)
    return {k: v for k, v in vars(cfg).items() if isinstance(v, scalar_types)}


def _dataset_type(cfg, split):
    if split == "valid":
        return getattr(cfg, "valid_dataset_type", getattr(cfg, "val_dataset_type", getattr(cfg, "dataset_type", None)))
    return getattr(cfg, f"{split}_dataset_type", getattr(cfg, "dataset_type", None))


def build_wandb_logger(cfg, config_path, log_dir, checkpoint_dir):
    repo_name = getattr(cfg, "repo_name", Path.cwd().name)
    experiment = getattr(cfg, "experiment", cfg.train_tag)
    train_dataset_type = _dataset_type(cfg, "train")
    if train_dataset_type is None:
        train_dataset_type = "dynamic_mixing" if cfg.train_set_dynamic_mixing else "presimulated"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = getattr(cfg, "wandb_run_name", None) or (
        f"{repo_name}__{experiment}__{train_dataset_type}__{timestamp}"
    )
    default_tags = [repo_name, experiment, train_dataset_type, cfg.se_model]
    tags = list(dict.fromkeys(default_tags + list(getattr(cfg, "wandb_tags", []) or [])))

    wandb_config = _serializable_config(cfg)
    wandb_config.update(
        {
            "repo_name": repo_name,
            "experiment": experiment,
            "config_path": str(config_path),
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "train_dataset_type": train_dataset_type,
            "val_dataset_type": _dataset_type(cfg, "valid"),
            "test_dataset_type": _dataset_type(cfg, "test"),
            "max_epochs": cfg.num_train_epochs,
            "val_check_interval": cfg.val_check_interval,
            "devices": cfg.num_gpu,
        }
    )

    return WandbLogger(
        project=getattr(cfg, "wandb_project", "bsrnn"),
        entity=getattr(cfg, "wandb_entity", None),
        name=run_name,
        group=getattr(cfg, "wandb_group", experiment),
        tags=tags,
        save_dir=str(log_dir),
        config=wandb_config,
    )

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    
    args = config_parser()
    cfg = Config(**vars(args))
    cfg.read_yaml()
    print(cfg)
    L.seed_everything(seed=cfg.seed)

    if cfg.train_set_dynamic_mixing:
        os.environ['OMP_NUM_THREADS'] = "1"

    if cfg.model_type == "flowse":
        model = FlowSEModel(cfg=cfg)
    else:
        model = SEModel(cfg=cfg)

    if cfg.init_from != 'none':
        state_dict = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        print(f"Init param loaded from {cfg.init_from}")

    print(model)

    log_dir = Path(getattr(cfg, "log_dir", f"./exp/{cfg.train_tag}"))
    checkpoint_dir = Path(
        getattr(
            cfg,
            "checkpoint_dir",
            log_dir / cfg.train_name / f"version_{cfg.train_version}" / "checkpoints",
        )
    )

    logger = TensorBoardLogger(save_dir=str(log_dir), version=cfg.train_version, name=cfg.train_name)
    if getattr(cfg, "use_wandb", False):
        wandb_logger = build_wandb_logger(cfg, cfg.config_file, log_dir, checkpoint_dir)
        logger = [logger, wandb_logger]
    call_backs = prepare_call_backs(cfg=cfg, checkpoint_dir=checkpoint_dir)

    ckpts = glob.glob(str(checkpoint_dir / "*-val_loss*.ckpt"))
    ckpts.sort(key=os.path.getmtime, reverse=True)
    last_ckpt = ckpts[0] if ckpts else None
    last_ckpt = last_ckpt if cfg.resume  else None
    if last_ckpt is not None:
        print(f"Resume form {last_ckpt}")

    trainer = L.Trainer(
        max_epochs=cfg.num_train_epochs,
        accelerator=cfg.device,
        devices=cfg.num_gpu,
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        callbacks=call_backs,
        strategy='auto' if cfg.num_gpu == 1 else 'ddp',
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=AudioDataModule(config=cfg), ckpt_path=last_ckpt,)
