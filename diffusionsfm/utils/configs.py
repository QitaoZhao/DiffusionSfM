import argparse
import os

from omegaconf import OmegaConf


def load_cfg(config_path):
    """
    Loads a yaml configuration file.

    Follows the chain of yaml configuration files that have a `_BASE` key, and updates
    the new keys accordingly. _BASE configurations can be specified using relative
    paths.
    """
    config_dir = os.path.dirname(config_path)
    config_path = os.path.basename(config_path)
    return load_cfg_recursive(config_dir, config_path)


def load_cfg_recursive(config_dir, config_path):
    """
    Recursively loads config files.

    Follows the chain of yaml configuration files that have a `_BASE` key, and updates
    the new keys accordingly. _BASE configurations can be specified using relative
    paths.
    """
    cfg = OmegaConf.load(os.path.join(config_dir, config_path))
    base_path = OmegaConf.select(cfg, "_BASE", default=None)
    if base_path is not None:
        base_cfg = load_cfg_recursive(config_dir, base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config_path)
    print(OmegaConf.to_yaml(cfg))

    exp_dir = os.path.join(cfg.training.runs_dir, cfg.training.exp_tag)
    os.makedirs(exp_dir, exist_ok=True)
    to_path = os.path.join(exp_dir, os.path.basename(args.config_path))
    if not os.path.exists(to_path):
        OmegaConf.save(config=cfg, f=to_path)
    return cfg


def get_cfg_from_path(config_path):
    """
    args:
        config_path - get config from path
    """
    print("getting config from path")

    cfg = load_cfg(config_path)
    print(OmegaConf.to_yaml(cfg))

    exp_dir = os.path.join(cfg.training.runs_dir, cfg.training.exp_tag)
    os.makedirs(exp_dir, exist_ok=True)
    to_path = os.path.join(exp_dir, os.path.basename(config_path))
    if not os.path.exists(to_path):
        OmegaConf.save(config=cfg, f=to_path)
    return cfg
