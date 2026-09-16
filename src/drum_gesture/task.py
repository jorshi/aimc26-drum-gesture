import lightning as L
import torch


class ContinuousGestureTask(L.LightningModule):
    """
    A Lightning module for continuous gesture recognition.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        # Initialize model, loss function, optimizer, etc.
        # self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs, _ = self.model(features)
        loss = self.loss_fn(outputs.squeeze(-1), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs, _ = self.model(features)
        loss = self.loss_fn(outputs.squeeze(-1), labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
