import logging
import os
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import PanopticSegmentation
from utils.utils import flatten_dict, RegularCheckpointing
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg.general.ckpt_path = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    model = PanopticSegmentation(cfg)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())
    # torch.use_deterministic_algorithms(True)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        default_root_dir=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def validate(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.validate(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.test(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def main(cfg: DictConfig):
    if cfg["general"]["mode"] == "train":
        train(cfg)
    elif cfg["general"]["mode"] == "validate":
        validate(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
