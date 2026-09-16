from flucoma_torch.scaler import FluidStandardize
import hydra
import lightning as L
from loguru import logger
from omegaconf import DictConfig
import torch

from drum_gesture.data import (
    ContinuousGestureDataset,
    create_continuous_gesture_datasets,
)
from drum_gesture.utils import to_absolute_dataset_path
from drum_gesture.export import export_model
from drum_gesture.training import fit_model


def load_dataset(cfg: DictConfig):
    """
    Load the continuous gesture train/validation datasets from the configuration.
    """
    audio_files = [to_absolute_dataset_path(f, cfg) for f in cfg.data.audio]
    annotation_files = [to_absolute_dataset_path(f, cfg) for f in cfg.data.annotations]

    cfg.feature.sample_rate = cfg.sample_rate
    feature = hydra.utils.instantiate(cfg.feature)
    logger.info(f"Instantiated feature extractor: {feature}")

    train_dataset, val_dataset = create_continuous_gesture_datasets(
        audio_files=audio_files,
        annotation_files=annotation_files,
        feature=feature,
        sample_rate=cfg.sample_rate,
        num_data=cfg.data.num_data,
        val_fraction=cfg.val_fraction,
        random_gain=cfg.random_gain,
        gain_domain=cfg.gain_domain,
        seed=cfg.seed,
    )

    return train_dataset, val_dataset


def load_model(cfg: DictConfig, input_size: int):
    model_cfg = cfg.model
    model_cfg.input_size = input_size
    model = hydra.utils.instantiate(model_cfg)
    logger.info(f"Instantiated model: {model}")
    return model


def save_standardize(dataset: ContinuousGestureDataset):
    if hasattr(dataset, "mean") and hasattr(dataset, "std"):
        scaler = FluidStandardize()
        scaler.mean = dataset.mean
        scaler.std = dataset.std
        scaler.save("standardize.json")


@hydra.main(version_base=None, config_path="../cfg", config_name="continous")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{cfg}")
    L.seed_everything(cfg.seed, workers=True)

    dataset, val_dataset = load_dataset(cfg)
    features, annotations = dataset[0]
    logger.info(f"Dataset length: {len(dataset)}, input shape: {features.shape}")

    model = load_model(cfg, input_size=features.shape[-1])

    model = fit_model(cfg, model, dataset, val_dataset)

    torch.save(model.state_dict(), "model.pt")
    export_model(model=model, output_path="model.json")

    # Save standardize
    save_standardize(dataset)


if __name__ == "__main__":
    main()
