
import pytorch_lightning as pl
import torch
import sys
import os
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.treeDataloader import SyntheticTreeDataset, SyntheticTwigsTreeDataset
from utils.visualize import save_ply
from utils.utils import cosine_similarity, cd_loss_L2, loadply, draw_tsne, draw_closest_figs, chamfer_distance_numpy
from IPython.terminal.embed import embed
import numpy as np
from torch.autograd import Variable


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2
                
class BranchReconstruction(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        torch.set_float32_matmul_precision('medium')
        self.save_hyperparameters(hparams)
        self.sample_num = self.hparams['sample_num']
        self._build_PN_model()
        self.train_step_outputs = []
        self.validation_step_outputs = []


    def _FC_decoder(self, input_channel, output_channel, bn=True, insn=True, bias=True, activation_fn=True):
        layers = []
        layers.append(nn.Linear(input_channel, output_channel, bias=bias))
        if bn:
            layers.append(nn.BatchNorm1d(output_channel))
        elif insn:
            layers.append(nn.InstanceNorm1d(output_channel))
            
        if activation_fn:
            layers.append(nn.ReLU(True))
        
        return nn.Sequential(*layers)

        
    def _build_PN_model(self):
        print("Building PointNet Model")
        self.SA_modules = nn.ModuleList()
        input_channels = 0
        use_xyz = True
        relation_prior = 1
        
        c_out_1 = 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.05],
                nsamples=[48],
                mlps=[[input_channels, c_out_1]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_2 = 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=96,
                radii=[0.1],
                nsamples=[64],
                mlps=[[c_out_1, c_out_2]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        c_out_3 = 256
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1],
                nsamples=[128],
                mlps=[[c_out_2, c_out_3]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        
        c_out_4 = 1024
        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample = 256,
                mlp=[c_out_3, c_out_4], 
                use_xyz=use_xyz
            )
        )

        self.final_npoint = 256
        self.latent_layer = nn.Sequential(
            nn.Linear(c_out_4 + 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.final_npoint),
            nn.BatchNorm1d(self.final_npoint),
            nn.ReLU(True),
            nn.Linear(self.final_npoint, self.final_npoint),
            nn.BatchNorm1d(self.final_npoint),
            nn.ReLU(True),
        )
        
        self.fc_decoder1 = self._FC_decoder(self.final_npoint, c_out_3, bn=True, insn=False)
        self.fc_decoder2 = self._FC_decoder(c_out_3, c_out_4, bn=True, insn=False)
        self.fc_decoder3 = self._FC_decoder(c_out_4, c_out_4, bn=True, insn=False)
        self.fc_decoder4 = self._FC_decoder(c_out_4, self.sample_num*3, bn=False, insn=False, activation_fn=False)
        
        self.classification_sequence = nn.Sequential(
            nn.Linear(self.final_npoint, self.final_npoint),
            nn.BatchNorm1d(self.final_npoint),
            nn.ReLU(True),
            nn.Linear(self.final_npoint, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
        
        self.dir_decoder1 = self._FC_decoder(self.final_npoint, 32, bn=True)
        self.dir_decoder2 = self._FC_decoder(32, 16, bn=True)
        self.dir_decoder3 = self._FC_decoder(16, 3, bn=False, insn=False, activation_fn=False)
        
        self.radius_decoder1 = self._FC_decoder(self.final_npoint, 32, bn=True)
        self.radius_decoder2 = self._FC_decoder(32, 16, bn=True)
        self.radius_decoder3 = self._FC_decoder(16, 1, bn=False, insn=False, activation_fn=False)

       
    def forward_PointNet(self, pointcloud, normalized_offsets):
        
        batch_size, _, _ = pointcloud.shape
        num_point = pointcloud.shape[1]
        
        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [pointcloud], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)
        
        embedding =l_features[-1].reshape(batch_size, -1).contiguous()
        embedding = torch.cat((embedding, normalized_offsets), axis=1)
        embedding = self.latent_layer(embedding)
        
        decoding_feat = self.fc_decoder1(embedding)
        decoding_feat = self.fc_decoder2(decoding_feat)
        decoding_feat = self.fc_decoder3(decoding_feat)
        decoding_feat = self.fc_decoder4(decoding_feat)
        
        reconstruction_res = decoding_feat.reshape(batch_size, -1, 3)
        cls_pred = self.classification_sequence(embedding)
        
        dir_feat = None
        if self.hparams['dir_weight'] != 0:
            dir_feat = self.dir_decoder1(embedding)
            dir_feat = self.dir_decoder2(dir_feat)
            dir_pred = self.dir_decoder3(dir_feat)
        radius_feat = None
        if self.hparams['radius_weight'] != 0:
            radius_feat = self.radius_decoder1(embedding)
            radius_feat = self.radius_decoder2(radius_feat)
            radius_feat = self.radius_decoder3(radius_feat)
            
        return embedding, reconstruction_res, cls_pred, dir_pred, radius_feat
        # return embedding, reconstruction_res, dir_pred, radius_feat
    
    
    def forward(self, pointcloud, normalized_offsets):
        return self.forward_PointNet(pointcloud, normalized_offsets)


    def training_step(self, batch, batch_idx):

        # pc, image, key, length, position = batch['pc'], batch['image'], batch['key'], batch['length'], batch['position']
        pc, foliage_cls, filepath, main_dirs, normalized_offsets, radius = batch['pc'], batch['foliage'], batch['filepath'], batch['main_dir'], batch['normalized_offset'], batch['radius']
        
        chamfer_loss = cd_loss_L2
        cos_error_term = nn.CosineEmbeddingLoss()
        
        output = self.forward(pc, normalized_offsets)
        _, recon_res, pred_cls, pred_dir, pred_radius = output
        
        cls_loss = F.cross_entropy(pred_cls, foliage_cls) * self.hparams['cls_weight']
        cd = chamfer_loss(recon_res, pc) * self.hparams['cd_weight']
        radius_loss = F.mse_loss(pred_radius, radius) * self.hparams['radius_weight']
        cos_loss = cos_error_term(main_dirs, pred_dir, Variable(torch.Tensor(pred_dir.size(0)).to(device=pred_dir.get_device()).fill_(1.0))) * self.hparams['dir_weight']
        
        loss = cd + cos_loss + radius_loss  + cls_loss

        with torch.no_grad():
            onehot = torch.argmax(pred_cls, dim=1)
            acc = (onehot == foliage_cls).float().mean()
            tp = torch.sum((onehot==1)&(foliage_cls==1))
            pp = onehot.sum()
            ap = foliage_cls.sum()
            precision = tp.float() / pp.float()
            recall = tp.float() / ap.float()
            
            
        log = dict(train_loss=loss, train_cos_loss=cos_loss, train_radius_loss=radius_loss)
        self.log_dict({'train_loss': loss.detach(), 
                    'train_cd': cd.detach(), 
                    'train_cos': cos_loss.detach(), 
                    'train_cls': cls_loss.detach(), 
                    'train_radius': radius_loss.detach(),
                    'train_acc': acc.detach(), 
                    'train_recall': recall.detach()}, prog_bar=True, on_step=False, on_epoch=True)

        # return_dict = None
        return_dict = dict(loss=loss, 
                           log=log, 
                           embedding=None,) 
                        #    filepath=filepath)
        
        # self.train_step_outputs.append(return_dict)

        return return_dict


    def validation_step(self, batch, batch_idx):
        # pc, image, key, length, position = batch['pc'], batch['image'], batch['key'], batch['length'], batch['position']
        pc, key, length, position, padding_num, foliage_cls, filepath, mian_dirs, normalized_offsets, radius = batch['pc'], batch['key'], batch['length'], batch['position'], batch['padding_num'], batch['foliage'], batch['filepath'], batch['main_dir'], batch['normalized_offset'], batch['radius']

        chamfer_loss = cd_loss_L2
        cos_error_term = nn.CosineEmbeddingLoss()
        
        _, recon_res, pred_cls, pred_dir, pred_radius = self.forward(pc, normalized_offsets)

        cd = chamfer_loss(recon_res, pc) * self.hparams['cd_weight']
        cls_loss = F.cross_entropy(pred_cls, foliage_cls) * self.hparams['cls_weight']
        cos_loss = cos_error_term(mian_dirs, pred_dir, Variable(torch.Tensor(pred_dir.size(0)).to(device=pred_dir.get_device()).fill_(1.0))) * self.hparams['dir_weight']
        radius_loss = F.mse_loss(pred_radius, radius) * self.hparams['radius_weight']
        
        loss = cd + cos_loss + radius_loss  + cls_loss

        with torch.no_grad():
            onehot = torch.argmax(pred_cls, dim=1)
            acc = (onehot == foliage_cls).float().mean()
            tp = torch.sum((onehot==1)&(foliage_cls==1))
            pp = onehot.sum()
            ap = foliage_cls.sum()
            precision = tp.float() / pp.float()
            recall = tp.float() / ap.float()
            
        log = dict(val_loss=loss, val_acc=acc)
        self.log_dict({'val_loss': loss.detach(), 
                    'val_cd': cd.detach(), 
                    'val_cos': cos_loss.detach(), 
                    'val_cls': cls_loss.detach(), 
                    'val_radius': radius_loss.detach(),
                    'val_acc': acc.detach()}, prog_bar=True, on_step=False, on_epoch=True)

        # pc = pc.detach().cpu().numpy()
        # recon_res = recon_res.detach().cpu().numpy()
        # if self.current_epoch % 10 == 0 and batch_idx == 0:
        #     for i in range(10):
        #         save_ply("./test_results/%s_%d_%i_gt.ply"%(self.hparams['task'], self.current_epoch, i), pc[i])
        #         save_ply("./test_results/%s_%d_%i.ply"%(self.hparams['task'], self.current_epoch, i), recon_res[i])

        return_dict = dict(val_loss=loss, 
                           log=log, 
                           embedding=None,)
        
        self.validation_step_outputs.append(return_dict)

        return return_dict
     
     
    def on_train_epoch_end(self):
        
        outputs = self.train_step_outputs
        if not self.hparams['no_validation']:
            if self.current_epoch % 10 == 0:
                training_embedding = torch.cat([x['embedding'] for x in outputs]).detach().cpu().numpy()
                training_filepath = []
                for x in outputs:
                    training_filepath += x['filepath']
                training_filepath = np.array(training_filepath)
                closest_neighbour_image = draw_closest_figs(self.eval_embeddings, 
                                                            training_embedding,
                                                            self.eval_filepath,
                                                            training_filepath,
                                                            os.path.join(self.hparams['data_src'], self.hparams['data_projector']), 
                                                            self.current_epoch, 
                                                            self.hparams['task'],
                                                            self.hparams['add_noise']
                                                            )
                
                self.logger.experiment.add_image("closest_neighbour", closest_neighbour_image, global_step=self.current_epoch, dataformats='HWC')
                os.system("rm -r *.png")
        
        self.train_step_outputs.clear()
        
        return
    
    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        # val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        if not self.hparams['no_validation']:
            output_num = torch.cat([x['key'] for x in outputs], dim=0).reshape(-1,).detach().cpu().numpy().shape[0]
            index_list = np.arange(output_num)
            index_list = np.concatenate([index_list[:500], index_list[-500:]])
            
            keys = torch.cat([x['key'] for x in outputs], dim=0).reshape(-1,).detach().cpu().numpy()[index_list]
            length = torch.cat([x['length'] for x in outputs], dim=0).detach().cpu().numpy()[index_list]
            position = torch.cat([x['position'] for x in outputs], dim=0).detach().cpu().numpy()[index_list]
            padding_num = torch.cat([x['padding_num'] for x in outputs], dim=0).detach().cpu().numpy()[index_list]
            cls = torch.cat([x['cls'] for x in outputs], dim=0).detach().cpu().numpy()[index_list]
            self.eval_embeddings = torch.cat([x['embedding'] for x in outputs], dim=0).detach().cpu().numpy()[index_list]
            self.eval_filepath = []
            for x in outputs:
                self.eval_filepath += x['filepath']
            self.eval_filepath= np.array(self.eval_filepath)[index_list]
            
            if self.current_epoch % 10 == 0:
                visualize_image, category = draw_tsne(self.eval_embeddings, 
                                                    keys, 
                                                    self.eval_filepath, 
                                                    os.path.join(self.hparams['data_src'], self.hparams['data_projector']), 
                                                    self.current_epoch, 
                                                    self.hparams['task'], 
                                                    length, 
                                                    position, 
                                                    padding_num,
                                                    cls)

                for (cls, image) in zip(category, visualize_image):
                    self.logger.experiment.add_image(cls, image, global_step=self.current_epoch, dataformats='HWC')
            
        self.validation_step_outputs.clear()

        return
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay, 
        )
        # lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        # bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        # return [optimizer], [lr_scheduler, bnm_scheduler]
        # return [optimizer], [lr_scheduler]
        return optimizer


    def _build_dataloader(self, task, data_src, data_folder, sample_num, mode, padding, show_image, add_noise, add_position):
        
        dset = SyntheticTreeDataset(task, data_src, data_folder, sample_num, train=mode=="train", padding=padding, show_image=show_image, add_noise=add_noise, add_position=add_position)
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size, 
            shuffle=mode == "train",
            num_workers=0,
            pin_memory=False,
            drop_last=mode == "train",
        )
 
    def train_dataloader(self):
        return self._build_dataloader(task=self.hparams['task'], data_src=self.hparams['data_src'], data_folder=self.hparams['data_folder'], sample_num=self.hparams['sample_num'], mode="train", padding=self.hparams['padding'], show_image=False, add_noise=self.hparams['add_noise'], add_position=self.hparams['add_position'])

    def val_dataloader(self):
        return self._build_dataloader(task=self.hparams['task'], data_src=self.hparams['data_src'], data_folder=self.hparams['data_folder'], sample_num=self.hparams['sample_num'], mode="val", padding=self.hparams['padding'], show_image=self.hparams['show_image'], add_noise=self.hparams['add_noise'], add_position=self.hparams['add_position'])