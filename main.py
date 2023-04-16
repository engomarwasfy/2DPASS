#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: main.py
@time: 2021/12/7 22:21
'''

import os
import yaml
import torch
import datetime
import importlib
import numpy as np
import pytorch_lightning as pl

from easydict import EasyDict
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from pytorch_lightning.callbacks import LearningRateMonitor

import warnings
warnings.filterwarnings("ignore")


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')

    # general
    # debug

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    if config['hyper_parameters']['test']:
        config['dataset_params']['val_data_loader']['batch_size'] = config['hyper_parameters']['num_vote']
    if config['hyper_parameters']['num_vote'] > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if  config['hyper_parameters']['debug']:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)


def build_loader(config):
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    train_config = config['dataset_params']['train_data_loader']
    val_config = config['dataset_params']['val_data_loader']
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None

    if not config['hyper_parameters']['test']:
        train_pt_dataset = pc_dataset(config, data_path=train_config['data_path'], imageset='train')
        val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val')
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(train_pt_dataset, config, train_config),
            batch_size=train_config["batch_size"],
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),
            shuffle=train_config["shuffle"],
            num_workers=train_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        # config['dataset_params']['training_size'] = len(train_dataset_loader) * len(configs.gpu)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=1),
            batch_size=val_config["batch_size"],
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),
            shuffle=val_config["shuffle"],
            pin_memory=True,
            num_workers=val_config["num_workers"]
        )
    else:
        if config['submit_to_server']:
            test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test', num_vote=val_config["batch_size"])
            test_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )
        else:
            val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val', num_vote=val_config["batch_size"])
            val_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )

    return train_dataset_loader, val_dataset_loader, test_dataset_loader


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    print(configs)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs['hyper_parameters']['gpu']))
    num_gpu = len(configs['hyper_parameters']['gpu'])

    # output path
    log_dir =configs['hyper_parameters']['log_dir']
    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=log_dir, default_hp_metric=False)
    os.makedirs(f'{log_folder}/{log_dir}', exist_ok=True)
    profiler = SimpleProfiler(filename=f'{log_folder}/{log_dir}/profiler.txt')
    np.set_printoptions(precision=4, suppress=True)

    # save the backup files
    backup_dir = os.path.join(log_folder,log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    if not configs['hyper_parameters']['test']:
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp main.py {}'.format(backup_dir))
        os.system('cp dataloader/dataset.py {}'.format(backup_dir))
        os.system('cp dataloader/pc_dataset.py {}'.format(backup_dir))
        os.system('cp {} {}'.format(configs.config_path, backup_dir))
        os.system('cp network/base_model.py {}'.format(backup_dir))
        os.system('cp network/baseline.py {}'.format(backup_dir))
        os.system('cp {}.py {}'.format('network/' + configs['model_params']['model_architecture'], backup_dir))

    # reproducibility
    torch.manual_seed(configs['hyper_parameters']['seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(configs['hyper_parameters']['seed'])
    config_path = configs.config_path

    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)

    pl.seed_everything(configs['hyper_parameters']['seed'])
    checkpoint_callback = ModelCheckpoint(
        monitor=configs['hyper_parameters']['monitor'],
        mode='max',
        save_last=True,
        save_top_k=configs['hyper_parameters']['save_top_k'])

    if configs['hyper_parameters']['checkpoint'] is not None and configs['hyper_parameters']['checkpoint']!='None' :
        print('load pre-trained model...')
        if configs['hyper_parameters']['fine-tune'] or configs['hyper_parameters']['test'] or configs['hyper_parameters']['pretrain2d']:
            my_model = my_model.load_from_checkpoint(configs['hyper_parameters']['checkpoint'], config=configs, strict=(not configs['hyper_parameters']['pretrain2d']))
        else:
            # continue last training
            my_model = my_model.load_from_checkpoint(configs['hyper_parameters']['checkpoint'])

    if configs['hyper_parameters']['SWA']:
        swa = [StochasticWeightAveraging(swa_epoch_start=configs.train_params.swa_epoch_start,swa_lrs=configs.train_params.swa_lrs ,annealing_epochs=configs.train_params.annealing_epochs)]
    else:
        swa = []

    if not configs['hyper_parameters']['test']:
        # init trainer
        print('Start training...')
        trainer = pl.Trainer(accelerator='cuda',
                             max_epochs=configs['train_params']['max_num_epochs'],
                             callbacks=[checkpoint_callback,
                                        LearningRateMonitor(logging_interval='step'),
                                        EarlyStopping(monitor=configs['hyper_parameters']['monitor'],
                                                      patience=configs['hyper_parameters']['stop_patience'],
                                                      mode='max',
                                                      verbose=True),
                                        ] + swa,
                             logger=tb_logger,
                             profiler=profiler,
                             check_val_every_n_epoch=configs['hyper_parameters']['check_val_every_n_epoch'],
                             #gradient_clip_val= configs['hyper_parameters']['gradient_clip_val'],
                             accumulate_grad_batches=configs['hyper_parameters']['accumulate_grad_batches'],
                             log_every_n_steps=configs['hyper_parameters']['log_every_n_steps'],
                             enable_checkpointing=configs['hyper_parameters']['enable_checkpointing'],
                             val_check_interval=configs['hyper_parameters']['val_check_interval'],
                             limit_val_batches=configs['hyper_parameters']['limit_val_batches'],
                             limit_train_batches=configs['hyper_parameters']['limit_train_batches'],
                             benchmark=configs['hyper_parameters']['benchmark'],
                             #precision=configs['hyper_parameters']['precision'],
                             num_sanity_val_steps=configs['hyper_parameters']['num_sanity_val_steps'],
                             #detect_anomaly=True

                             )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)

    else:
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(accelerator='gpu',
                             resume_from_checkpoint=configs['hyper_parameters']['checkpoint'],
                             logger=tb_logger,
                             profiler=profiler)
        trainer.test(my_model, test_dataset_loader if configs['hyper_parameters']['submit_to_server'] else val_dataset_loader)