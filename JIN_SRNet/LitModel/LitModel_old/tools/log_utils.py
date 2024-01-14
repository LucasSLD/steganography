import sys
import os

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from typing import Any

WORK_DIR = '/home/lucas/Documents/Master Data Science/S1/Research_project/'

class CustomLearningRateMonitor(LearningRateMonitor):
    def on_train_epoch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        assert trainer.logger is not None
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.current_epoch)
                


def setup_callbacks_loggers(args):

    log_path = os.path.join(WORK_DIR, 'LogFiles', args.alg, args.payload, 
                            args.backbone, args.version)

    tb_logger = TensorBoardLogger(log_path, name="", version="", default_hp_metric=False)
    lr_logger = CustomLearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(dirpath=os.path.join(log_path, 'checkpoints'),
                                    filename='epoch={epoch:02d}_val_acc={val_acc:.4f}',
                                    auto_insert_metric_name=False,
                                    save_top_k=5, save_last=True, monitor='val_acc', mode='max')

    return ckpt_callback, [tb_logger], lr_logger
