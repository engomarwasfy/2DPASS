###
# Disclaimer: This is not our central method, we recommend the greedy soup which is found in main.py.
# This method is described in appendix I and, compared to main.py, this code is much less tested.
# For instance, we don't know how stable the results are under optimization noise. However, we expect
# this method to outperform greedy soup. Still, we recommend using greedy soup and not this.
# As mentioned in the paper, this code is computationally expernsive as it requires loading models in memory.
# We run this on a node with 490GB RAM and use 1 GPU with 40GB of memory.
# It also looks like PyTorch released a very helpful utility which we recommend if re-implementing:
# https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html?utm_source=twitter&utm_medium=organic_social&utm_campaign=docs&utm_content=functional-api-for-modules
# When running with lr = 0.05 and epochs = 5 we get 81.38%.
###

import argparse
import os
import torch
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn

import pytorch_lightning as pl

import argparse
import copy
import importlib
import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from easydict import EasyDict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler

from checkpointSort import check_points_sort
from dataloader.dataset import get_collate_class, get_model_class
from dataloader.pc_dataset import get_pc_model_class


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(1,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')
    parser.add_argument('--save_top_k', type=int, default=10,
                        help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--SWA', action='store_true', default=False, help='StochasticWeightAveraging')
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')
    # testing
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')
    parser.add_argument('--num_vote', type=int, default=1, help='number of voting in the test')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    if args.test:
        config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote
    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)


def build_loader(config):
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    train_config = config['dataset_params']['train_data_loader']
    val_config = config['dataset_params']['val_data_loader']
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None

    if not config['test']:
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
            test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test',
                                         num_vote=val_config["batch_size"])
            test_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )
        else:
            val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val',
                                        num_vote=val_config["batch_size"])
            val_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )

    return train_dataset_loader, val_dataset_loader, test_dataset_loader


class AlphaWrapper(pl.LightningModule):
    def __init__(self, model, checkpoints , total_number_of_models):
        super(AlphaWrapper, self).__init__()
        self.my_model = model
        self.checkpoints = checkpoints
        self.soup = self.checkpoints[0]
        self.alpha_raw = nn.Parameter(torch.ones(total_number_of_models))
        self.beta = nn.Parameter(torch.tensor(1.))

    def alpha(self):
        return nn.functional.softmax(self.alpha_raw, dim=0)

    def forward(self, inp):
        alph = self.alpha()
        # generate the model
        for i, checkpoint in enumerate(self.checkpoints['checkpoints']):
            print("iteration number ", i, " out of ", len(self.checkpoints['checkpoints']))
            if i == 0:
                continue
            new_params = torch.load(checkpoint['path'], map_location='cpu')['state_dict']
            checkpoints_dicts.append(new_params)

        trained_checkpoint = {
            k: best_checkpoint['state_dict'][k].clone() * self.alpha_raw[0]
            for k in best_checkpoint['state_dict']
        }
        torch.cuda.empty_cache()
        for i, checkpoint in enumerate(sorted_dict['checkpoints']):
            new_ingredient_params = torch.load(checkpoint['path'], map_location='cpu')['state_dict']
            if (i == 0):
                continue
            trained_checkpoint = {
                k: trained_checkpoint[k].clone() +
                   new_ingredient_params[k].clone() * self.alpha_raw[i]
                for k in new_ingredient_params
            }
        self.soup['state_dict'] = trained_checkpoint
        torch.save(best_checkpoint, greedy_soup_temp_checkpoint_path)
        my_model = self.my_model.load_from_checkpoint(greedy_soup_temp_checkpoint_path, config=configs,
                                                 strict=(not configs.pretrain2d))
        out = self.model(inp)
        return self.beta * out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, input, groundTruth):
        output = self.forward(input)
        loss = nn.CrossEntropyLoss()(output, groundTruth)
        return loss

    def validation_step(self, input):
        pass


if __name__ == '__main__':
    SOUPS_CHECKPOINT_DIR = 'batchSize=8_2'
    SOUPS_RESULTS_DIR = 'soups/uniform_soup'
    configs = parse_config()
    print(configs)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    num_gpu = len(configs.gpu)
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(configs.seed)
    config_path = configs.config_path
    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)
    soups = os.getcwd() + '/' + SOUPS_CHECKPOINT_DIR
    greedy_soup_temp_checkpoint_path = os.getcwd() + '/' + SOUPS_RESULTS_DIR + '/' + 'greedy_soup_temp.ckpt'
    greedy_soup_checkpoint_path = os.getcwd() + '/' + SOUPS_RESULTS_DIR + '/' + 'greedy_soup.ckpt'

    sorted_dict = check_points_sort()
    best_checkpoint = None
    results = {'model_name': f'uniform_soup'}
    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)
    os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)
    profiler = SimpleProfiler(filename='profiler.txt')
    best_checkpoint = torch.load(sorted_dict['checkpoints'][0]['path'], map_location='cpu')
    checkpoints_dicts = [best_checkpoint['state_dict']]
    accuracy_list = [sorted_dict['checkpoints'][0]['miou']]
    alpha_model = AlphaWrapper(my_model, sorted_dict, len(sorted_dict['checkpoints']))

    trainer = pl.Trainer(accelerator='gpu',
                         logger=tb_logger,
                         profiler=profiler)



    results = trainer.train(alpha_model, val_dataset_loader)


