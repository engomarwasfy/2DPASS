import argparse
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
    parser.add_argument('--save_top_k', type=int, default=10, help='save top k checkpoints, use -1 to checkpoint every epoch')
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
        NUM_MODELS = 5
        SOUPS_CHECKPOINT_DIR = '../default'
        SOUPS_RESULTS_DIR = 'soups/uniform_soup'
        SOUPS_RESULTS_FILE_NAME = 'soup.ckpt'
        # parameters
        configs = parse_config()
        print(configs)

        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
        num_gpu = len(configs.gpu)

        # reproducibility
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        np.random.seed(configs.seed)
        config_path = configs.config_path

        train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)
        model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
        my_model = model_file.get_model(configs)

        # create the uniform soup sequentially to not overload memory
        soups = os.getcwd() + '/' + SOUPS_CHECKPOINT_DIR
        soup_result_file = os.getcwd() + '/' + SOUPS_RESULTS_DIR + '/' + SOUPS_RESULTS_FILE_NAME
        greedy_soup_temp_checkpoint_path = os.getcwd() + '/' + SOUPS_RESULTS_DIR + '/' + 'greedy_soup_temp.ckpt'
        uniform_soup = 'uniform_soup'
        sorted_dict = check_points_sort()
        best_checkpoint = None
        for i, checkpoint in enumerate(sorted_dict['checkpoints']):
            if i == NUM_MODELS:
                break
            modelWeight = torch.load(checkpoint['path'])
            state_dict = modelWeight.get('state_dict')
            if i == 0:
                uniform_soup = {k: v * (1. / NUM_MODELS) for k, v in state_dict.items()}
                best_checkpoint = modelWeight
            else:
                uniform_soup = {k: v * (1. / NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        best_checkpoint['state_dict'] = uniform_soup
        torch.save(best_checkpoint, soup_result_file)

        results = {'model_name': f'uniform_soup'}
        log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
        tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)
        os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)
        profiler = SimpleProfiler(filename='profiler.txt')
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        my_model = my_model.load_from_checkpoint(soup_result_file, config=configs, strict=(not configs.pretrain2d))
        trainer = pl.Trainer(accelerator='gpu',
                             # resume_from_checkpoint=soup_result_file,
                             logger=tb_logger,
                             profiler=profiler)



        best_miou_so_far=0
        num_ingredients = 0
        greedy_soup_initial = torch.load(sorted_dict['checkpoints'][0]['path'])
        greedy_soup_params = greedy_soup_initial['state_dict']
        greedy_soup_ingredients = [greedy_soup_params]
        # Now, iterate through all models and consider adding them to the greedy soup.
        for i, checkpoint in enumerate(sorted_dict['checkpoints']):
            if i == 0:
                best_checkpoint= torch.load(checkpoint['path'])
            # Get the potential greedy soup, which consists of the greedy soup with the new model added.
            new_ingredient_params = torch.load(checkpoint['path'])['state_dict']
            num_ingredients = num_ingredients + 1
            potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }
            best_checkpoint['state_dict'] = potential_greedy_soup_params

            # Save the potential greedy soup to a temporary checkpoint file.
            torch.save(best_checkpoint, greedy_soup_temp_checkpoint_path)
            my_model = my_model.load_from_checkpoint(greedy_soup_temp_checkpoint_path, config=configs,
                                                     strict=(not configs.pretrain2d))
            trainer = pl.Trainer(accelerator='gpu',
                                 # resume_from_checkpoint=soup_result_file,
                                 logger=tb_logger,
                                 profiler=profiler)
            results = trainer.test(my_model, val_dataset_loader)
            miou = results[0]['val/mIoU']

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            if miou > best_miou_so_far:
                greedy_soup_ingredients.append(new_ingredient_params)
                best_miou_so_far = miou
                greedy_soup_params = potential_greedy_soup_params


