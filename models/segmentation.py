import pytorch_lightning as pl
import torch
import sys
import numpy as np
import os
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.treeDataloader import SyntheticSegmentationDataset
from utils.utils import cosine_similarity
from torchvision.ops.focal_loss import sigmoid_focal_loss
from utils.utils import cosine_similarity, vector_normalization
from IPython import embed
from utils.visualize import save_segmentation

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


class FocalLoss(nn.Module):
    '''
    Focal Loss
        FL=alpha*(1-p)^gamma*log(p) where p is the probability of ground truth class
    Parameters:
        alpha(1D tensor): weight for positive
        gamma(1D tensor):
    '''
    
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=torch.tensor(alpha)
        self.gamma=gamma
        self.reduce=reduce

    def forward(self, input, target):
        BCE_Loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_Loss)
        Focal_Loss = torch.pow((1-pt), self.gamma) * F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.alpha, reduction='none')

        if self.reduce=='none':
            return Focal_Loss
        elif self.reduce=='sum':
            return torch.sum(Focal_Loss)
        else:
            return torch.mean(Focal_Loss)


class Segmentation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # for key in hparams.keys():
        #     self.hparams[key]=hparams[key]
        torch.set_float32_matmul_precision("medium")
        self.save_hyperparameters(hparams)
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0

        self.SA_modules.append(     # 0
            PointnetSAModuleMSG(
                npoint=self.hparams['lc_count'],
                radii=[0.075, 0.1, 0.125],
                nsamples=[16, 32, 48],
                mlps=[[c_in, 64], [c_in, 64], [c_in, 64]],
                first_layer=True,
                use_xyz=self.hparams['use_xyz'],
                relation_prior=1
            )
        )
        c_out_0 = 64*3

        c_in = c_out_0
        self.SA_modules.append(    # 1
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.15, 0.2],
                nsamples=[16, 48, 64],
                mlps=[[c_in, 128], [c_in, 128], [c_in, 128]],
                use_xyz=self.hparams['use_xyz'],
                relation_prior=1
            )
        )
        c_out_1 = 128*3

        c_in = c_out_1
        self.SA_modules.append(    # 2
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.3, 0.4],
                nsamples=[16, 32, 48],
                mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                use_xyz=self.hparams['use_xyz'],
                relation_prior=1
            )
        )
        c_out_2 = 256*3

        c_in = c_out_2
        self.SA_modules.append(    # 3
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.4, 0.6, 0.8],
                nsamples=[16, 24, 32],
                mlps=[[c_in, 512], [c_in, 512], [c_in, 512]],
                use_xyz=self.hparams['use_xyz'],
                relation_prior=1
            )
        )
        c_out_3 = 512*3
        
        self.SA_modules.append(   # 4   global pooling
            PointnetSAModule(
                nsample = 16,
                mlp=[c_out_3, 128], use_xyz=self.hparams['use_xyz']
            )
        )
        
        self.SA_modules.append(   # 5   global pooling
            PointnetSAModule(
                nsample = 64,
                mlp=[c_out_2, 128], use_xyz=self.hparams['use_xyz']
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[512, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.juncion_segmentation_layer = nn.Sequential(

            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 2, kernel_size=1),
        )


    def forward(self, pointcloud):

        batch_size, _, _ = pointcloud.shape
        l_xyz, l_features = [pointcloud], [None]
        for i in range(len(self.SA_modules)):
            if i < 5:
                li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
                if li_xyz is not None:
                    random_index = np.arange(li_xyz.size()[1])
                    np.random.shuffle(random_index)
                    li_xyz = li_xyz[:, random_index, :]
                    li_features = li_features[:, :, random_index]
                l_xyz.append(li_xyz)
                l_features.append(li_features)
        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1 - 1] = self.FP_modules[i](
                l_xyz[i - 1 - 1], l_xyz[i - 1], l_features[i - 1 - 1], l_features[i - 1]
            )

        return self.juncion_segmentation_layer(l_features[0]).transpose(1, 2).contiguous()


    def training_step(self, batch, batch_idx):
        
        pc, gt_isfork = batch['pc'], batch['is_fork']
        seg_pred = self.forward(pc)
        
        seg_loss = F.cross_entropy(seg_pred.reshape(-1, 2), gt_isfork.reshape(-1), torch.Tensor([1, self.hparams['FL_alpha']]).to(device=pc.get_device()))
        # seg_loss = F.cross_entropy(seg_pred, gt_isfork)
        loss = seg_loss

        with torch.no_grad():
            onehot = torch.argmax(seg_pred, dim=2)
            acc = (onehot == gt_isfork).float().mean()
            tp = torch.sum((onehot==1)&(gt_isfork==1))
            pp = onehot.sum()
            ap = gt_isfork.sum()
            precision = tp.float() / pp.float()
            recall = tp.float() / ap.float()
            f1_score = 2 * precision * recall / (precision + recall)

        log = dict(train_loss=loss, train_acc=acc, train_recall=recall)
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)

        return dict(loss=loss, log=log)


    def validation_step(self, batch, batch_idx):
        
        pc, gt_isfork = batch['pc'], batch['is_fork']
        seg_pred = self.forward(pc)

        # print(torch.sum(gt_isfork[0]) / gt_isfork[0].shape[0])
        seg_loss = F.cross_entropy(seg_pred.reshape(-1, 2), gt_isfork.reshape(-1), torch.Tensor([1, self.hparams['FL_alpha']]).to(device=pc.get_device()))
        # seg_loss = F.cross_entropy(seg_pred, gt_isfork)

        loss = seg_loss
        
        with torch.no_grad():
            onehot = torch.argmax(seg_pred, dim=2)
            acc = (onehot == gt_isfork).float().mean()
            tp = torch.sum((onehot==1)&(gt_isfork==1))
            pp = onehot.sum()
            ap = gt_isfork.sum()
            precision = tp.float() / pp.float()
            recall = tp.float() / ap.float()
            f1_score = 2 * precision * recall / (precision + recall)

        log = dict(val_loss=loss, val_acc=acc, val_recall=recall)
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)
        
        # visualization
        if batch_idx % 20 == 0:
            for idx in range(5):
                vis_pc = pc[idx].detach().cpu().numpy()
                vis_is_forks = torch.argmax(seg_pred, dim=2)[idx].detach().cpu().numpy()
                save_segmentation("test_data/%s_%d_%d.ply"%(self.hparams['task'], batch_idx, idx), vis_pc, vis_is_forks)
                gt_vis_is_forks = gt_isfork[idx].detach().cpu().numpy()
                save_segmentation("test_data/%s_%d_%d_gt.ply"%(self.hparams['task'], batch_idx, idx), vis_pc, gt_vis_is_forks)
                
        
        return dict(val_loss=loss, val_acc=acc, val_recall=recall, log=log)


    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams.lr_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            lr_clip / self.hparams.lr,
        )
        bn_lbmd = lambda _: max(
            self.hparams.bn_momentum
            * self.hparams.bnm_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        # bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        # return [optimizer], [lr_scheduler, bnm_scheduler]
        return [optimizer], [lr_scheduler]


    def _build_dataloader(self, data_src, data_selection, train_mode):
        dset = SyntheticSegmentationDataset(data_src, data_selection, train_mode)
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=train_mode,
            num_workers=8,
            pin_memory=True,
            drop_last=train_mode,
        )


    def train_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], self.hparams['data_selection'], train_mode=True)


    def val_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], self.hparams['data_selection'], train_mode=False)
