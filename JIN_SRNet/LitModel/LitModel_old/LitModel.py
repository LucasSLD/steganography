WORK_DIR = '/home/lucas/Documents/Master Data Science/S1/Research_project/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np 
import pickle
import argparse
#import wandb
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generator, Union
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import random
from optimizers import *
import models
from retriever import *
from custom_metrics import *

from torchmetrics.classification.accuracy import Accuracy

class LitModel(pl.LightningModule):
    """Transfer Learning
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 cover_folder_name: str,
                 stego_folder_name: str,
                 backbone: str = 'mixnet_s',
                 batch_size: int = 32,
                 pair_constraint: int = 0,
                 lr: float = 1e-3,
                 eps: float = 1e-8,
                 lr_scheduler_name: str = 'cos',
                 surgery: str = '',
                 qf: str = '',
                 size: str = '',
                 optimizer_name: str = 'adamw',
                 num_workers: int = 6, 
                 epochs: int = 50, 
                 gpus: str = '0', 
                 weight_decay: float = 1e-2,
                 decoder: str = 'NR',
                 payload: str = '',
                 alg: str ='',
                 seed: str ='',
                 in_channels: int = 1,
                 save: int = 0
                 ,**kwargs) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.cover_folder_name = cover_folder_name
        self.stego_folder_name = stego_folder_name
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.pair_constraint = pair_constraint
        self.qf = qf
        self.size = size
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_name = optimizer_name
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.surgery = surgery
        self.decoder = decoder
        self.payload = payload
        self.alg = alg
        self.seed = seed
        self.in_channels = in_channels
        self.save = save
        if self.qf != '':
            self.data_path = self.data_path+'QF_'+self.qf+'/'
        if self.pair_constraint:
            self.effective_batch_size = 2*self.batch_size
        if not self.data_path.endswith("/"):
            self.data_path += "/"
        if not self.cover_folder_name.endswith("/"):
            self.cover_folder_name += "/"
        if not self.stego_folder_name.endswith("/"):
            self.stego_folder_name += "/"
        
        self.save_hyperparameters()

        self.train_metrics = {'train_acc': Accuracy(),'train_PE': PE()}
        self.val_metrics = {'val_acc': Accuracy(), 'val_wAUC': wAUC(), 'val_PE': PE(), 'val_MD5': MD5()}
        self.test_metrics = {'test_acc': Accuracy(), 'test_wAUC': wAUC(), 'test_PE': PE(), 'test_MD5': MD5()}
        
        self.__set_attributes(self.train_metrics)
        self.__set_attributes(self.val_metrics)
        self.__set_attributes(self.test_metrics)
        
        self.__build_model()
        
    def __set_attributes(self, attributes_dict):
        for k,v in attributes_dict.items():
            setattr(self, k, v) 

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.net = models.get_net(self.backbone)
        

        self.net.in_channels = self.in_channels
            
        if self.decoder == 'eY':
            self.net.image_c *= 2
        
        if self.surgery != '':
            self.net = getattr(models, self.surgery)(self.net)

        # 2. Loss:
        self.loss_func = F.cross_entropy
        
        self.test_y = []
        self.test_y_hat = []

    def forward(self, x):
        """Forward pass. Returns logits."""

        x = self.net(x)
        
        return x

    def loss(self, logits, labels):
        return self.loss_func(logits, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        
        # 2. Compute loss:
        train_loss = self.loss(y_logits, y)
            
        # 3. Compute metrics and log:
        self.log("train_loss", train_loss, on_step=False, on_epoch=True,  prog_bar=True, logger=True, sync_dist=False)
        
        for metric_name in self.train_metrics.keys():
            self.log(metric_name, getattr(self, metric_name)(y_logits, y), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        return train_loss

    def training_epoch_end(self, outputs):
        self.log("step", self.current_epoch, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.train_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss:
        val_loss = self.loss(y_logits, y)
        
        # 3. Compute metrics and log:
        self.log('val_loss', val_loss, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.val_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)
            
    def validation_epoch_end(self, outputs):
        self.log("step", self.current_epoch, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.val_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

            
    #def on_test_epoch_start(self, *args, **kwargs):
    #    super().on_test_epoch_start(*args, **kwargs)
    #    self.test_table = wandb.Table(columns=['label', 'preds'])
        
    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y, name = batch
        y_logits = self.forward(x)

        # 2. Compute loss:
        test_loss = self.loss(y_logits, y)
     #   for i in range(len(name)):
     #       self.test_table.add_data(y[i], y_logits[i])
        
        # 3. Compute metrics and log:
        self.log('test_loss', test_loss, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.test_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)

    def test_epoch_end(self, outputs):
        test_summary = {'best_ckpt_path': self.trainer.resume_from_checkpoint}
        for metric_name in self.test_metrics.keys():
            test_summary[metric_name] = getattr(self, metric_name).compute()
            self.log(metric_name, test_summary[metric_name], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()
            
        return test_summary


    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_name)
        
        optimizer_kwargs = {'momentum': 0.9} if self.optimizer_name == 'sgd' else {'eps': self.eps}
        
        optimizer = optimizer(self.parameters(), 
                              lr=self.lr, 
                              weight_decay=self.weight_decay, 
                              **optimizer_kwargs)
        num_gpus = (len(self.gpus)//2)+1
        print('Number of gpus', num_gpus)
        
        steps_per_epochs = len(self.train_dataset)//num_gpus//self.batch_size
        
        if self.lr_scheduler_name == 'cos':
            scheduler_kwargs = {'T_max': self.epochs*steps_per_epochs, 
                                'eta_min':self.lr/50} 
            
            
        elif self.lr_scheduler_name == 'onecycle':
            scheduler_kwargs = {'max_lr': self.lr, 'epochs': self.epochs, 'steps_per_epoch':steps_per_epochs,
                                'pct_start':3/self.epochs,'div_factor':25,
                                'final_div_factor':2}
        
        elif self.lr_scheduler_name == 'multistep':
             scheduler_kwargs = {'milestones':[50], 'gamma':0.5}
        
        scheduler = get_lr_scheduler(self.lr_scheduler_name)
                
        scheduler_params, interval = get_lr_scheduler_params(self.lr_scheduler_name, **scheduler_kwargs)
        
        scheduler = scheduler(optimizer, **scheduler_params)

        return [optimizer], [{'scheduler':scheduler, 'interval': interval, 'name': 'lr'}]
    
    


    def prepare_data(self):
        """Download images and prepare images datasets."""
        
        print('data downloaded')

    def setup(self, stage: str): 
                
        
        sizes = ['512']
        if self.size != '':
            sizes = [self.size]
        
        classes = ['Cover', 'Stego']
        # if self.payload != '':
        #     classes = ['Cover', self.alg + '/' + self.payload]
        # elif self.alg != '':
        #     classes = ['Cover', self.alg]
        IL_train = []
        IL_val = []
        
        with open(WORK_DIR + 'JIN_SRNet/IL_train.p', 'rb') as handle:
            IL_train.extend(pickle.load(handle))
        with open(WORK_DIR + 'JIN_SRNet/IL_val.p', 'rb') as handle:
            IL_val.extend(pickle.load(handle))
            
        if not IL_train[0].endswith("pt"):
            for i, name in enumerate(IL_train):
                IL_train[i] = name[:-3] + "pt"
            for i, name in enumerate(IL_val):
                IL_val[i] = name[:-3] + "pt"

        dataset = []
        
        if self.pair_constraint == True:
            retriever = TrainRetrieverPaired
            
            for path in IL_train:
                dataset.append({
                    'kind': classes,
                    'image_name': [len(classes)*path],
                    'label': range(len(classes)),
                    'fold':1,
                    })
                
            for path in IL_val:
                dataset.append({
                    'kind': classes,
                    'image_name': [len(classes)*path],
                    'label': range(len(classes)),
                    'fold':0,
                    })
        
        if self.pair_constraint == False:
            retriever = TrainRetriever_pt
            for label, kind in enumerate(classes):
                for path in IL_train:
                    if kind == "Cover":
                        image_name = self.cover_folder_name + path
                    elif kind == "Stego":
                        image_name = self.stego_folder_name + path
                    else:
                        raise ValueError(f"Unsupported kind -> {kind}")
                    dataset.append({
                        'kind': kind,
                        'image_name': image_name,
                        'label': label,
                        'fold':1,
                    })
            for label, kind in enumerate(classes):
                for path in IL_val:
                    if kind == "Cover":
                        image_name = self.cover_folder_name + path
                    elif kind == "Stego":
                        image_name = self.stego_folder_name + path
                    else:
                        raise ValueError(f"Unsupported kind -> {kind}")
                    dataset.append({
                        'kind': kind,
                        'image_name': image_name,
                        'label': label,
                        'fold':0,
                    })
            
        random.shuffle(dataset)
        dataset = pd.DataFrame(dataset)
        
        self.train_dataset = retriever(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] != 0].kind.values,
            image_names=dataset[dataset['fold'] != 0].image_name.values,
            labels=dataset[dataset['fold'] != 0].label.values,
            transforms=get_train_transforms(),
            decoder=self.decoder
        )
        # print("==================================================")
        # print(self.train_dataset)
        self.valid_dataset = retriever(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] == 0].kind.values,
            image_names=dataset[dataset['fold'] == 0].image_name.values,
            labels=dataset[dataset['fold'] == 0].label.values,
            transforms=get_valid_transforms(),
            decoder=self.decoder
        )
    
    
    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn if self.pair_constraint else None,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='mixnet_s',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--data-path',
                            default='/home/lucas/Documents/Master Data Science/S1/Research_project/JIN_SRNet/',
                            type=str,
                            metavar='dp',
                            help='data path')
        parser.add_argument('--cover-folder-name',
                            default='BossBase-1.01-cover/',
                            type=str,
                            metavar='cfn',
                            help="cover folder's name")
        parser.add_argument('--stego-folder-name',
                            default='stego_1_5bit/',
                            type=str,
                            metavar='sfn',
                            help="stego folder's name")
        parser.add_argument('--epochs',
                            default=50,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--batch-size',
                            default=32,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--pair-constraint',
                            type=int,
                            default=0,
                            help='Use pair constraint?')
        parser.add_argument('--gpus',
                            type=str,
                            default='0',
                            help='which gpus to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--eps',
                            default=1e-8,
                            type=float,
                            help='eps for adaptive optimizers',
                            dest='eps')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--lr-scheduler-name',
                            default='cos',
                            type=str,
                            metavar='LRS',
                            help='Name of LR scheduler')
        parser.add_argument('--optimizer-name',
                            default='adamw',
                            type=str,
                            metavar='OPTI',
                            help='Name of optimizer')
        parser.add_argument('--surgery',
                            default='',
                            type=str,
                            help='name of surgery function')
        parser.add_argument('--qf',
                            default='',
                            type=str,
                            help='quality factor')
        parser.add_argument('--qf2',
                            default='',
                            type=str,
                            help='quality factor')
        parser.add_argument('--size',
                            default='',
                            type=str,
                            help='image size')
        parser.add_argument('--weight-decay',
                            default=1e-2,
                            type=float,
                            metavar='wd',
                            help='Optimizer weight decay')
        parser.add_argument('--decoder',
                            default='NR',
                            type=str,
                            help='decoder')
        parser.add_argument('--payload',
                            default='',
                            type=str,
                            help='payload')
        parser.add_argument('--alg',
                            default='',
                            type=str,
                            help='algorithm')
        parser.add_argument('--in-channels',
                            default=0,
                            type=int,
                            help='number of input channels')
        parser.add_argument('--save',
                            default=0,
                            type=int,
                            help='save outputs?')

        return parser
