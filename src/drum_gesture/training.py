"""
Shared training loop for the continuous gesture models.

Used by both the audio-based path (scripts/train_continuous.py) and the feature-frame
path (scripts/train_frames.py) so the two stay comparable -- the frame path exists partly
to validate the C++ trainer against, and that only works if the surrounding training
procedure is identical.
"""

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig
import torch


def fit_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset = None,
) -> torch.nn.Module:
    """
    Train ``model``, returning it with the best-validation weights loaded.
    """
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    loss = hydra.utils.instantiate(cfg.loss)
    task = hydra.utils.instantiate(
        cfg.task, model=model, loss_fn=loss, learning_rate=cfg.learning_rate
    )

    # Without early stopping and best-checkpoint selection, a short recording trained for
    # the full epoch budget just memorises -- which is the normal case for a musician
    # recording a minute of gesture rather than a curated corpus.
    callbacks = []
    checkpoint = None
    if val_dataloader is not None:
        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, filename="best"
        )
        callbacks = [
            checkpoint,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=cfg.min_delta,
            ),
        ]

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        callbacks=callbacks,
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    # Export the best weights, not whatever the last epoch happened to leave behind.
    if checkpoint is not None and checkpoint.best_model_path:
        logger.info(
            f"Restoring best checkpoint (val_loss={checkpoint.best_model_score:.5f}) "
            f"from {checkpoint.best_model_path}"
        )
        best = torch.load(checkpoint.best_model_path, map_location="cpu")
        state_dict = {
            k[len("model.") :]: v
            for k, v in best["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)

    return model
