#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: spvcnn.py
@time: 2021/12/16 22:41
'''
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.basic_block import Lovasz_loss
from network.base_model import LightningBaseModel
from network.basic_block import SparseBasicBlock
from network.voxel_fea_generator import voxel_3d_generator, voxelization
import pytorch_lightning as pl

class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale,indx):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))
        self.indx = indx
    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        output, inv = self.downsample(data_dict[f'coors{self.indx}'], features)
        identity = self.layer_in(features)
        output = self.PPmodel(output)[inv]
        output = torch.cat([identity, output], dim=1)

        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict[f'coors_inv{self.indx}']]),
            data_dict['scale_{}'.format(str(self.scale)+str(self.indx))][f'coors_inv{self.indx}'],
            dim=0
        )
        data_dict[f'coors{self.indx}'] = data_dict['scale_{}'.format(str(self.scale)+str(self.indx))][f'coors{self.indx}']
        data_dict[f'coors_inv{self.indx}'] = data_dict['scale_{}'.format(str(self.scale)+str(self.indx))][f'coors_inv{self.indx}']
        data_dict[f'full_coors{self.indx}'] = data_dict['scale_{}'.format(str(self.scale)+str(self.indx))][f'full_coors{self.indx}']

        return v_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape,indx):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder(in_channels, out_channels, scale,indx)
        self.indx = indx

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(str(self.last_scale)+str(self.indx) )][f'coors_inv{self.indx}']
        coors_inv = data_dict['scale_{}'.format(str(self.scale)+str(self.indx))][f'coors_inv{self.indx}']

        # voxel encoder
        v_fea = self.v_enc(data_dict[f'sparse_tensor{self.indx}'])
        data_dict['layer_{}'.format(str(self.layer_id)+str(self.indx))] = {}
        data_dict['layer_{}'.format(str(self.layer_id)+str(self.indx))]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(str(self.layer_id)+str(self.indx))]['full_coors'] = data_dict[f'full_coors{self.indx}']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        p_fea = self.p_enc(
            features=data_dict[f'sparse_tensor{self.indx}'].features+v_fea.features,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict[f'sparse_tensor{self.indx}'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,
            indices=data_dict[f'coors{self.indx}'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv]
class criterion(nn.Module):
    def __init__(self,config,indx):
        super(criterion, self).__init__()
        self.lambda_lovasz = config['train_params'].get('lambda_lovasz', 0.1)
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )
        self.indx = indx
    def forward(self, data_dict):
        loss_main_ce = self.ce_loss(data_dict[f'logits{self.indx}'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict[f'logits{self.indx}'], dim=1), data_dict['labels'].long())
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz
        data_dict[f'loss_main_ce{self.indx}'] = loss_main_ce
        data_dict[f'loss_main_lovasz{self.indx}'] = loss_main_lovasz
        data_dict[f'loss{self.indx}'] += loss_main

        return data_dict

class RSU(nn.Module):
    def __init__(self, config,levelIndex=1,decoderLevelIndex=7,decoder= False,):
        super(RSU,self).__init__()
        self.input_dims = config[f'model_params{levelIndex}']['input_dims']
        self.hiden_size = config[f'model_params{levelIndex}']['hiden_size']
        self.num_classes = config[f'model_params{levelIndex}']['num_classes']
        self.scale_list = config[f'model_params{levelIndex}']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config[f'dataset_params']['min_volume_space']
        max_volume_space = config[f'dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config[f'model_params{levelIndex}']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]
        self.levelIndex = levelIndex
        self.decoderLevelIndex = decoderLevelIndex
        self.decoder = decoder
        self.config = config
        self.indx = self.levelIndex
        if (self.decoder):
            self.indx = self.decoderLevelIndex
        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list,
            indx=self.indx
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            indx=self.indx
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist(),indx=self.indx)
            )

        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # loss
        self.criterion = criterion(self.config,self.indx)

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)
        enc_feats = []
        '''
        if(self.indx<=4 and self.indx!=1):
            
            for j in range (1,self.indx):
                enc_feats.append(data_dict['layer_{}'.format(str(0)+str(j))]['pts_feat'])
        if (self.indx > 4 and self.indx != 1):

            for j in range(1, 4):
                enc_feats.append(data_dict['layer_{}'.format(str(0) + str(j))]['pts_feat'])
        '''
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))


        output = torch.cat(enc_feats, dim=1)


        data_dict[f'logits{self.indx}'] = self.classifier(output)
        data_dict[f'loss{self.indx}'] = 0.
        data_dict = self.criterion(data_dict)
        return data_dict

### RSU-7 ###
class RSU7(RSU) :#UNet07DRES(nn.Module):
    def __init__(self,config):
        super(RSU7, self).__init__(config,levelIndex=1,decoderLevelIndex=7,decoder= True)
    def forward(self,x):
        return super(RSU7, self).forward(x)

### RSU-6 ###
class RSU6(RSU):#UNet06DRES(nn.Module):
    def __init__(self,config):
        super(RSU6, self).__init__(config,levelIndex=2, decoderLevelIndex=6,decoder= True)
    def forward(self,x):
        return super(RSU6, self).forward(x)
### RSU-5 ###
class RSU5(RSU):#UNet05DRES(nn.Module):
    def __init__(self,config):
       super(RSU5, self).__init__(config,levelIndex=3, decoderLevelIndex=5,decoder= True)
    def forward(self,x):
        return super(RSU5, self).forward(x)
### RSU-4 ###
class RSU4(RSU):#UNet04DRES(nn.Module):
    def __init__(self,config):
       super(RSU4, self).__init__(config,levelIndex=4,decoderLevelIndex=4,decoder= False)
    def forward(self,x):
        return super(RSU4, self).forward(x)
### RSU-3 ###
class RSU3(RSU):#UNet03DRES(nn.Module):
    def __init__(self,config):
       super(RSU3, self).__init__(config,levelIndex=3,decoderLevelIndex=3,decoder= False)
    def forward(self,x):
        return super(RSU3, self).forward(x)
### RSU-2 ###
class RSU2(RSU):#UNet03DRES(nn.Module):
    def __init__(self,config):
       super(RSU2, self).__init__(config,levelIndex=2,decoderLevelIndex=2,decoder= False)
    def forward(self,x):
        return super(RSU2, self).forward(x)
### RSU-1 ###
class RSU1(RSU):#UNet03DRES(nn.Module):
    def __init__(self,config):
        super(RSU1, self).__init__(config,levelIndex=1,decoderLevelIndex=1,decoder= False)
    def forward(self,x):
        return super(RSU1, self).forward(x)
##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,config):
        super(U2NET,self).__init__()

        self.stage1 = RSU1(config)
        self.pool12 = spconv.SparseMaxPool3d(2,stride=2)

        self.stage2 = RSU2(config)
        self.pool23 = spconv.SparseMaxPool3d(2,stride=2)

        self.stage3 = RSU3(config)
        self.pool34 = spconv.SparseMaxPool3d(2, stride=2)

        self.stage4 = RSU4(config)
        self.pool43 = spconv.SparseMaxPool3d(2, stride=2)
        # decoder
        self.stage3d = RSU5(config)
        self.pool32 = nn.MaxUnpool3d(2, stride=2)
        self.stage2d = RSU6(config)
        self.pool21 = nn.MaxUnpool3d(2, stride=2)
        self.stage1d = RSU7(config)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        #hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx1)
        #hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx2)
        #hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx3)
        #hx = self.pool43(hx4)
        #-------------------- decoder --------------------

        #stage 5
        hx3d = self.stage3d(hx4)
        #hx = self.pool32(hx3d)

        #stage 6
        hx2d = self.stage2d(hx3d)
        #hx = self.pool21(hx2d)
        # stage 7
        hx1d = self.stage1d(hx2d)

        return hx1d
