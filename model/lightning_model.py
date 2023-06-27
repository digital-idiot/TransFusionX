import torch
import torch_optimizer
from typing import Any
from torch import optim
from typing import Dict
from typing import Optional
from torchmetrics import MeanMetric
from .registry import LOSS_REGISTRY
from .registry import MODEL_REGISTRY
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from pytorch_lightning import LightningModule
from model.metrics import MultiClassSegmentationMetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT


# noinspection PyShadowingBuiltins
class LightningModel(LightningModule):
    def __init__(
            self,
            model_name: str,
            model_params: Dict[str, Any],
            optmizer_name: str,
            optimizer_params: Dict[str, Any],
            criterion_name: str,
            criterion_params: Dict[str, Any],
            scheduler_name: str,
            scheduler_params: Optional[Dict[str, Any]] = None,
            ignore_index: int = None,
            normalize_cm: str = 'true'
    ) -> None:
        """
        Wrapper to make model lightning compatible
        Args:
            model_params: Model's keyword parameters
            optimizer_params: Optimizer, use partial function for extra args
            criterion_params: Loss function
            scheduler_params: scheduler, optional, use partial function for extra args
            ignore_index: index to ignore
            normalize_cm: method to normalize confusion matrix
        """
        super(LightningModel, self).__init__()
        self.save_hyperparameters()
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.criterion_params = criterion_params
        self.scheduler_params = scheduler_params
        self.optimizer_name = optmizer_name
        self.scheduler_name = scheduler_name
        self.ignore_index = ignore_index
        self.normalization = normalize_cm

        # noinspection PyUnresolvedReferences
        self.model = getattr(
            MODEL_REGISTRY, model_name
        )(**self.model_params)
        self.criterion = getattr(
            LOSS_REGISTRY, criterion_name
        )(**criterion_params)

        self.training_metrics = MultiClassSegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=self.ignore_index,
            average=None,
            prefix='Training_',
        )
        self.validation_metrics = MultiClassSegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=ignore_index,
            average=None,
            prefix='Validation_',
        )
        self.test_metrics = MultiClassSegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=self.ignore_index,
            average=None,
            prefix='Test_',
        )

        self.training_epoch_loss = MeanMetric(nan_strategy='warn')
        self.validation_epoch_loss = MeanMetric(nan_strategy='warn')
        self.test_epoch_loss = MeanMetric(nan_strategy='warn')

    def configure_optimizers(self):
        optimizer_params = self.optimizer_params.copy()
        scheduler_params = self.scheduler_params.copy()
        opt = getattr(
            optim, self.optimizer_name
        ) if hasattr(optim, self.optimizer_name) else getattr(
            torch_optimizer, self.optimizer_name
        )
        optimizer_params.update(params=self.model.parameters())
        optimizer = opt(**optimizer_params)
        if self.scheduler_name is not None:
            lrs = getattr(lr_scheduler, self.scheduler_name)
            scheduler_params.update(optimizer=optimizer)
            if "total_steps" in scheduler_params.keys():
                scheduler_params.update(
                    total_steps=(
                        int(
                            scheduler_params[
                                "total_steps"
                            ] * self.trainer.estimated_stepping_batches
                        )
                    )
                )
            scheduler = lrs(**scheduler_params)
        else:
            scheduler = None
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def update_batch_size(self, batch_size: int):
        self.hparams.batch_size = batch_size

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def loss_function(self, prediction: Any, target: Any) -> Any:
        return self.criterion(prediction, target)

    def log_multi(self, tag: str, metric: torch.Tensor) -> None:
        if isinstance(self.logger, TensorBoardLogger):
            log_dict = {
                f"C_{i}": v
                for i, v in enumerate(metric)
                if i != self.ignore_index
            }
            log_dict["C̅"] = metric.mean()
            # noinspection PyUnresolvedReferences
            self.logger.experiment.add_scalars(
                tag,
                log_dict,
                global_step=self.global_step
            )

    def log_collection(self, collection: MetricCollection) -> None:
        metric_dict = collection.compute()
        for tag, metric in metric_dict.items():
            self.log_multi(tag=tag, metric=metric)

    def training_step(
            self,
            data: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
            optimizer_idx: int = 0
    ) -> STEP_OUTPUT:
        data = data.batch
        _ = data.pop("query_mask", None)
        target = data["label"].squeeze(dim=1)
        prediction = self.forward(data)

        # target[mask] = self.ignore_index
        current_loss = self.loss_function(
            prediction=prediction, target=target
        )

        self.training_metrics.update(preds=prediction, target=target)
        self.training_epoch_loss.update(
            value=current_loss.detach().clone().squeeze(),
            weight=1
        )
        self.log(
            name="training_epoch_loss",
            value=self.training_epoch_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True
        )
        return {
            'loss': current_loss
        }

    def validation_step(
            self,
            data: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        data = data.batch
        _ = data.pop("query_mask", None)
        target = data["label"].squeeze(dim=1)
        prediction = self.forward(data)

        current_loss = self.loss_function(
            prediction=prediction, target=target
        )
        self.validation_metrics.update(preds=prediction, target=target)
        self.validation_epoch_loss.update(
            value=current_loss.detach().clone().squeeze(),
            weight=1
        )
        self.log(
            name="validation_epoch_loss",
            value=self.validation_epoch_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            name="val_loss",
            value=current_loss.detach().clone().squeeze(),
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )
        return {
            'val_loss': current_loss
        }

    def test_step(
            self,
            data: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        data = data.batch
        _ = data.pop("query_mask", None)
        target = data["label"].squeeze(dim=1)
        prediction = self.forward(data)

        current_loss = self.loss_function(
            prediction=prediction, target=target
        )
        self.test_metrics.update(preds=prediction, target=target)
        self.test_epoch_loss.update(
            value=current_loss.detach().clone().squeeze(),
            weight=1
        )
        self.log(
            name="test_epoch_loss",
            value=self.test_epoch_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log(
            name="test_loss",
            value=current_loss.detach().clone().squeeze(),
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False
        )
        return {
            'test_loss': current_loss
        }

    def predict_step(
            self,
            data: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        data = data.batch
        _ = data.pop("query_mask", None)
        pred = self.forward(data)
        return {
            "image": torch.unbind(input=data["image"], dim=0),
            "image_transform_list": data["image_transfoms_list"],
            "image_crs_list": data["image_crs_list"],
            "prediction": torch.unbind(
                input=torch.argmax(
                    input=pred.log_softmax(dim=1).exp(),
                    dim=1,
                    keepdim=True
                ),
                dim=0
            ),
            "label": [None] * len(
                data["label_transfoms_list"]
            ) if data["label"] is None else torch.unbind(
                input=data["label"], dim=0
            ),
            "label_transfoms_list": data["label_transfoms_list"],
            "label_crs_list": data["label_crs_list"],
        }

    def on_train_epoch_end(self) -> None:
        self.log_collection(self.training_metrics)
        self.training_metrics.reset()
        self.training_epoch_loss.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_collection(self.validation_metrics)
        self.validation_metrics.reset()
        self.validation_epoch_loss.reset()

    def on_test_epoch_end(self) -> None:
        self.log_collection(self.test_metrics)
        self.test_metrics.reset()
        self.test_epoch_loss.reset()
