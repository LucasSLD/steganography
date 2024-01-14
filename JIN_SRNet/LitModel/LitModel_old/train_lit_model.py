"""
Runs a model on a single node across multiple gpus.
"""
import warnings
import sys

# sys.path.append("/linkhome/rech/gencrl01/una46ym/LitModel/")
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
import torch
from pytorch_lightning import Trainer, seed_everything
from tools.log_utils import setup_callbacks_loggers

seed_everything(666)

def main(args):
    """Main training routine specific for this project."""
    model = LitModel(**vars(args))
    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)
    trainer = Trainer(logger=loggers,
                     callbacks=[ckpt_callback, lr_logger],
                     accelerator='gpu',
                     strategy='ddp' if len(args.gpus or '') > 1 else None,
                    #  devices=int(args.gpus),
                     devices="auto",
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     precision=16,
                     #amp_backend='apex',
                     #amp_level='O1',
                     log_every_n_steps=100,
                     #flush_logs_every_n_steps=100,
                     benchmark=True,
                     sync_batchnorm=len(args.gpus or '') > 1,
                     #progress_bar_refresh_rate=0,
                     enable_progress_bar=False,
                     resume_from_checkpoint=args.resume_from_checkpoint)
    
    print("======================================================")
    print(Trainer.devices)

    if args.seed is not None:
        print("Seeding from", args.seed)
        # model.net = model.load_from_checkpoint(args.seed).net
        checkpoint = torch.load(args.seed, map_location='cuda:'+str(torch.cuda.current_device()))
        # if checkpoint['hyper_parameters']['surgery']:
        #     if checkpoint['hyper_parameters']['sca_layers']:
        #         model.net.sca_layers = checkpoint['hyper_parameters']['sca_layers']
        #     model.net = getattr(models, checkpoint['hyper_parameters']['surgery'])(model.net)
        # model.net.load_state_dict({k.split('net.')[-1]: v for k,v in checkpoint['state_dict'].items()})
        model.net.load_state_dict(checkpoint)

    # if args.surgery is not None:
    #     model.net.in_channels = args.in_channels
    #     model.net.sca_layers = args.sca_layers
    #     model.net = getattr(models, args.surgery)(model.net)

    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model)

    # trainer.test(dataloaders=datamodule, ckpt_path=model.best_ckpt_path)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)

    parser.add_argument("--version",
                        default=None,
                        type=str,
                        metavar="V",
                        help="version or id of the net",)
    parser.add_argument("--resume-from-checkpoint",
                        default=None,
                        type=str,
                        metavar="RFC",
                        help="path to checkpoint",)
    parser.add_argument("--seed",
                        default=None,
                        type=str,
                        help="path to seeding checkpoint")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
