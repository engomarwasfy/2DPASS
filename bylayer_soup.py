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
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torchmetrics import Accuracy
from checkpointSort import check_points_sort
from dataloader.dataset import get_collate_class, get_model_class
from dataloader.pc_dataset import get_pc_model_class
from utils.metric_util import IoU
from utils.schedulers import cosine_schedule_with_warmup


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
    def __init__(self, model, checkpoints , total_number_of_models,configs):
        super(AlphaWrapper, self).__init__()
        self.my_model = model
        self.checkpoints = checkpoints
        self.soup = torch.load(sorted_dict['checkpoints'][0]['path'], map_location='cuda')
        self.alpha_raw = nn.Parameter(torch.ones(total_number_of_models), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.train_acc = Accuracy(task="multiclass", num_classes=20)
        self.val_acc = Accuracy(task="multiclass", num_classes=20, compute_on_step=False)
        self.val_iou = IoU(configs['dataset_params'], compute_on_step=False)
        self.num_classes = 20
        self.ignore_label = 0
        self.configs= configs
    def alpha(self):
        return nn.functional.softmax(self.alpha_raw, dim=0)
    def forward(self, inp):
        alph = self.alpha()
        alphacpu = alph.to('cpu')
        # generate the model
        best_checkpoint = torch.load(sorted_dict['checkpoints'][0]['path'], map_location='cpu')
        trained_checkpoint = {
                k: best_checkpoint['state_dict'][k].clone() * alphacpu[0]
                for k in best_checkpoint['state_dict']
            }

        my_model_x=self.my_model.load_from_checkpoint(sorted_dict['checkpoints'][0]['path'], config=configs,
                                           strict=(not configs.pretrain2d))
        my_model_xCuda = my_model_x.cuda()
        outputs = []
        outputs.append(my_model_xCuda(inp))
        torch.cuda.empty_cache()
        for i, checkpoint in enumerate(sorted_dict['checkpoints']):
            new_ingredient_params = torch.load(checkpoint['path'], map_location='cpu')['state_dict']
            if (i == 0):
                continue
            if (i==1):
                break
            my_model_x = self.my_model.load_from_checkpoint(checkpoint['path'] , config=configs,
                                                            strict=(not configs.pretrain2d))
            my_model_xCuda = my_model_x.cuda()
            outputs.append(my_model_xCuda(inp))

            trained_checkpoint = {
                k: trained_checkpoint[k].clone() +
                    new_ingredient_params[k].clone() * alphacpu[i]
                for k in new_ingredient_params
            }
        self.soup['state_dict'] = trained_checkpoint
        torch.save(self.soup, greedy_soup_temp_checkpoint_path)
        my_model_n = self.my_model.load_from_checkpoint(greedy_soup_temp_checkpoint_path, config=configs,
                                                 strict=(not configs.pretrain2d))
        my_model_m=my_model_n.cuda()
        out = my_model_m(inp)
        outputs.append(out)
        outputs_alpha_beta = []
        for i in range(len(outputs)):
            if(i == len(outputs) - 1):
                outputs_alpha_beta.append({k: self.beta.to(self.my_model.device) * alphacpu.to(self.my_model.device)[i] * v if torch.is_tensor(v) and torch.is_floating_point(v) else v
                    for k, v in outputs[i].items()})
        # Multiply numeric tensors with self.beta
        outputs_alpha_beta.append({k: self.beta.to(self.my_model.device) * v if torch.is_tensor(v) and torch.is_floating_point(v) else v
                    for k, v in outputs[i].items()})

        return outputs_alpha_beta

    def configure_optimizers(self):
        alpha_beta_params =  [
            {"params": self.alpha_raw},
            {"params": self.beta}
        ]
        optimizer = torch.optim.SGD(alpha_beta_params,
                                        lr=self.configs['train_params']["learning_rate"],
                                        momentum=self.configs['train_params']["momentum"],
                                        weight_decay=self.configs['train_params']["weight_decay"],
                                        nesterov=self.configs['train_params']["nesterov"])
        from functools import partial
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.configs['train_params']['max_num_epochs'],
                    batch_size=self.configs['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.configs['dataset_params']['training_size'],
                    num_gpu=len(self.configs.gpu)
                ))

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' ,
            'frequency': 1
         }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.configs.monitor,
            }
    def training_step(self, input, batch_idx):
        print(self.alpha_raw)
        outputs = self.forward(input)
        #print(self.alpha_raw)
        #print(list(self.parameters()))
        self.log('train/acc', self.train_acc, on_epoch=True)
        #self.log('train/loss_main_ce', outputs['loss_main_ce'])
        #self.log('train/loss_main_lovasz', outputs['loss_main_lovasz'])

        #output avg loss of all models
        individual_losses = [output['loss'] for output in outputs]
        total_loss = sum(individual_losses)
        return total_loss
    def validation_step(self, data_dict , batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        my_model_n = self.my_model.load_from_checkpoint(greedy_soup_temp_checkpoint_path, config=configs,
                                                        strict=(not configs.pretrain2d))
        my_model_m=my_model_n.cuda()
        data_dict = my_model_m(data_dict)
        vote_logits = data_dict['logits'].cpu()
        raw_labels = data_dict['labels'].squeeze(0).cpu()
        prediction = vote_logits.argmax(1)
        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1
        self.val_acc(prediction, raw_labels)
        self.log('val/acc', self.val_acc, on_epoch=True)
        self.val_iou(
            prediction.cpu().detach().numpy(),
            raw_labels.cpu().detach().numpy(),
        )
        return data_dict['loss']
    def on_validation_epoch_end(self):
        iou, best_miou = self.val_iou.compute()
        mIoU = np.nanmean(iou)
        str_print = ''
        self.log('val/mIoU', mIoU, on_epoch=True)
        self.log('val/best_miou', best_miou, on_epoch=True)
        str_print += 'Validation per class iou: '
        for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)
        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
        self.print(str_print)
        self.val_iou.hist_list = []
        def on_after_backward(self) -> None:
            """
            Skipping updates in case of unstable gradients
            https://github.com/Lightning-AI/lightning/issues/4956
            """
            print("hello world")
            valid_gradients = True
            for name, param in self.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                print(f'detected inf or nan values in gradients. not updating model parameters')
                self.zero_grad()
if __name__ == '__main__':
    SOUPS_CHECKPOINT_DIR = 'default3'
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
    alpha_model = AlphaWrapper(my_model, sorted_dict,1,configs)
    torch.cuda.empty_cache()
    trainer = pl.Trainer(accelerator='cuda',
                         devices=[0],
                         # fast_dev_run = True,
                         strategy='auto',
                         max_epochs=configs['train_params']['max_num_epochs'],
                         # resume_from_checkpoint=configs.checkpoint if not configs.fine_tune and not configs.pretrain2d else None,
                         callbacks=[LearningRateMonitor(logging_interval='step'),
                                    EarlyStopping(monitor=configs.monitor,
                                                  patience=configs.stop_patience,
                                                  mode='max',
                                                  verbose=True),
                                    ] ,
                         logger=[tb_logger],
                         profiler=profiler,
                         gradient_clip_val=1,
                         accumulate_grad_batches=1,
                         # log_every_n_steps = 10 ,
                         limit_val_batches = 0.01,
                         limit_train_batches = 0.01,
                         # benchmark = True,
                         # precision=configs['hyper_parameters']['precision'],
                         # num_sapnity_val_steps = 2 ,
                         # detect_anomaly=True
                         sync_batchnorm=True,
                         )
    results = trainer.fit(alpha_model, train_dataloaders =train_dataset_loader,val_dataloaders= val_dataset_loader)


