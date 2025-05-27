import torch
import sys
import os

# from models.dir_model_no_cls import BranchReconstruction
from models.dir_model import BranchReconstruction
# from models.dir_point_transformer import BranchReconstruction
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.plugins import DDPPlugin
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='real_sync_lookup_shuffle', type=str)
    parser.add_argument('--padding', default='constant', type=str)
    parser.add_argument('--sample_num', default=500, type=int)
    parser.add_argument('--emd_weight', default=0, type=float)
    parser.add_argument('--cd_weight', default=1000.0, type=float)
    parser.add_argument('--cls_weight', default=0.02, type=float)
    parser.add_argument('--dir_weight', default=1, type=float)
    parser.add_argument('--radius_weight', default=1000, type=float)
    parser.add_argument('--backbone', default="RSCNN", type=str)
    parser.add_argument('--show_image', action='store_true')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--add_position', action='store_true')
    parser.add_argument('--dir_to_top', action='store_true')
    parser.add_argument('--no_validation', action='store_true')
    parser.add_argument('--tree_num', default=1500)
    
    args = parser.parse_args()

    # path = '/data/zhou1178/shapenet/PartAnnotation'
    path = '/data/zhou1178/TreeStructorData'
    tree_num = int(args.tree_num)
    data_folder = 'TreePart%d'%(tree_num)
    data_projector = 'TreePartProjector%d'%(tree_num)
    # path = '/media/dummy1/zhou1178/PointCloudTreePart'
    hparams = {'task': args.task+"%d_%d_%.1f_%.1f_%s_%s"%(tree_num, args.cd_weight, args.cls_weight, args.dir_weight, args.backbone, args.padding), 
               'batch_size':200,
               'input_channels' : 0,
               'use_xyz' : True,
               'lr': 0.001,
               'weight_decay': 0.0,
               'lr_decay': 0.5,
               'decay_step': 3e5,
               'bn_momentum': 0.5,
               'bnm_decay': 0.5,
               'FL_alpha': 253/192,
               'FL_gamma': 2,
               'data_src': path,
               'data_folder': data_folder,
               'data_projector': data_projector,
               'sample_num': args.sample_num,
               'padding': args.padding,
               'emd_weight': args.emd_weight,
               'cd_weight': args.cd_weight,
               'cls_weight': args.cls_weight,
               'dir_weight': args.dir_weight,
               'radius_weight': args.radius_weight,
               "show_image": args.show_image,
               "backbone": args.backbone,
               "add_noise": args.add_noise,
               'add_position': args.add_position,
               'no_validation': args.no_validation,
               }

    logger = TensorBoardLogger("logs", name=hparams['task'])

    model = BranchReconstruction(hparams)


    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), 'checkpoints', hparams['task']),
        save_top_k=5,
        verbose=True,
        every_n_epochs=5,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        # devices=[0],
        devices=[0,1,2,3],
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, RichProgressBar()],
        max_epochs=5000,
        logger=logger,
        num_sanity_val_steps=1,
        log_every_n_steps=1
    )
    trainer.fit(model)
    # trainer.fit(model, ckpt_path='./checkpoints/treestructor12000_1000_0.0_1.0_RSCNN_constant/epoch=34-step=111440.ckpt')
