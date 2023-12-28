"""
Runs a model on a single node across multiple gpus.
"""
import warnings
import sys

sys.path.append("/linkhome/rech/gencrl01/una46ym/LitModel/")
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
import torch
from pytorch_lightning import Trainer, seed_everything
from tools.log_utils import setup_callbacks_loggers

seed_everything(666)

WORK_DIR = "/gpfswork/rech/ohz/una46ym/"


def main(args):
    """Main training routine specific for this project."""
    model = LitModel(**vars(args))
    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)
    trainer = Trainer(
        logger=loggers,
        callbacks=[ckpt_callback, lr_logger],
        gpus=args.gpus,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        precision=16,
        amp_backend="native",
        amp_level="O1",
        log_every_n_steps=100,
        flush_logs_every_n_steps=100,
        accelerator="ddp" if len(args.gpus or "") > 1 else None,
        benchmark=True,
        sync_batchnorm=len(args.gpus or "") > 1,
        progress_bar_refresh_rate=0,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    if args.seed is not None:
        print("Seeding from", args.seed)
        model.net = model.load_from_checkpoint(args.seed).net
        # Do some if statement here
    #         model.net._fc = torch.nn.Linear(in_features=2560, out_features=4, bias=True)

    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model)

    # trainer.test(dataloaders=datamodule, ckpt_path=model.best_ckpt_path)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)

    parser.add_argument(
        "--version",
        default=None,
        type=str,
        metavar="V",
        help="version or id of the net",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        type=str,
        metavar="RFC",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--seed", default=None, type=str, help="path to seeding checkpoint"
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
