"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
from retriever import *
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer, seed_everything
from tools.log_utils import setup_callbacks_loggers

seed_everything(1994)

WORK_DIR = '/gpfswork/rech/ohz/una46ym/'

def main(args):
    """ Main training routine specific for this project. """
    
    model = LitModel(**vars(args)).load_from_checkpoint(args.resume_from_checkpoint)
    
    ckpt_callback, tb_logger, lr_logger = setup_callbacks_loggers(args)
    
    trainer = Trainer(checkpoint_callback=ckpt_callback,
                     logger=tb_logger,
                     callbacks=[lr_logger],
                     gpus=args.gpus,
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     precision=16,
                     amp_backend='native',
                     amp_level='O1',
                     log_every_n_steps=100,
                     flush_logs_every_n_steps=100,
                     accelerator='ddp' if len(args.gpus or '') > 1 else None,
                     benchmark=True,
                     sync_batchnorm=len(args.gpus or '') > 1,
                     resume_from_checkpoint=args.resume_from_checkpoint)
    
    sizes = ['512']
    if args.size != '':
        sizes = [args.size]
        
    classes = ['Cover', 'JUNI']
    #classes = ['JPG_trunc/QF_100/Cover', 'JPG_libjpeg6b/QF_100/Cover', 'JPG_libjpeg7/QF_100/Cover', 'JPG_libjpeg6b_full/QF_100/Cover']
    if args.payload != '':
        classes = ['Cover', args.alg + '/' + args.payload]
    elif args.alg != '':
        classes = ['Cover', args.alg]
        
    IL_test = []
    with open(WORK_DIR + 'DataBase/BOSSBase512/IL_test_n.p', 'rb') as handle:
            IL_test.extend(pickle.load(handle))
            
    # if (not IL_test[0].endswith('jpg')) and args.qf !='':
    #         for i,name in enumerate(IL_test):
    #             IL_test[i] = name[:-3] + 'jpg'
#     with open(WORK_DIR + 'DataBase/ALASKAv2/IL_test.p', 'rb') as handle:
#             IL_test.extend(pickle.load(handle))

#     for size in sizes:
#         with open('/home/jbutora/QF100/IL_test_'+size+'.p', 'rb') as handle:
#             IL_test.extend(pickle.load(handle))
            
    dataset = []
    for label, kind in enumerate(classes):
        for path in IL_test:
            dataset.append({
                'kind': kind,
                'image_name': path,
                'label': label,
                'fold':2,
            })
    dataset = pd.DataFrame(dataset)
    test_dataset = TrainRetriever_hdf5(
            data_path=model.data_path,
            kinds=dataset[dataset['fold'] == 2].kind.values,
            image_names=dataset[dataset['fold'] == 2].image_name.values,
            labels=dataset[dataset['fold'] == 2].label.values,
            transforms=get_valid_transforms(),
            decoder=args.decoder,
            return_name=True,
        )
    test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    trainer.test(model, test_dataloader, args.resume_from_checkpoint)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)
    
    parser.add_argument('--version',
                         default=None,
                         type=str,
                         metavar='V',
                         help='version or id of the net')
    parser.add_argument('--resume-from-checkpoint',
                         default=None,
                         type=str,
                         metavar='RFC',
                         help='path to checkpoint')
    
    
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_cli()