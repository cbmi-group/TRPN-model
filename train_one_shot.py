
import os
import sys
import datetime
import argparse
import subprocess
import shutil
import torch

# from trainers.trainer_FS2_one_shot import trainer_segmentation_oneshot
# from trainers.trainer_FS2_five_shot import trainer_segmentation_fiveshot
# from trainers.trainer_RANet import trainer_segmentation_ranet

from trainers import *
from utils.utils import print_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--port', type=int, default=8008)

    parser.add_argument('--network', type=str, default='ranet_model') # "one_model", "five_model", "ranet_model" 
    parser.add_argument('--n_classes', type=int, default=1)

    parser.add_argument('--train_support_list', type=str, default='mito_widefield_64/train_oneshot_support_for_er.txt')
    parser.add_argument('--train_query_list', type=str, default='mito_widefield_64/train_oneshot_query_for_er.txt')
    parser.add_argument('--val_support_list', type=str, default='er_tubule_confocal_64/test_oneshot_support.txt')
    parser.add_argument('--val_query_list', type=str, default='er_tubule_confocal_64/test.txt')
    parser.add_argument('--pretrained', type=str, default='train_seg_unet-mito-train.txt-2021325_1312/checkpoints/checkpoints_epoch_30.pth')
    parser.add_argument('--init_lr', type=float, default=0.001) 
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr_decay_every_x_epochs', type=int, default=10)
    parser.add_argument('--save_every_x_epochs', type=int, default=10)
    parser.add_argument('--show_results_every_x_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_bz', type=int, default=4)
    parser.add_argument('--init_lr_decay', type=float, default=1)

    parser.add_argument('--gamma', type=float, default=0.1) 
    parser.add_argument('--train_skl', action='store_true', default=False)
    parser.add_argument('--val_skl', action='store_true', default=False)
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--equalize', action='store_true', default=True)
    parser.add_argument('--std', action='store_true', default=False)
    parser.add_argument('--base_dim', type=int, default=64) 
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--semi', action='store_true', default=False)
 
    args = parser.parse_args()

    data_type, txt_name = os.path.split(args.train_support_list)
    data_type = data_type.split("_")[0]
    args.train_support_list = os.path.join('./data/data_list', args.train_support_list)
    args.train_query_list = os.path.join('./data/data_list', args.train_query_list)
    args.val_support_list = os.path.join('./data/data_list', args.val_support_list)
    args.val_query_list = os.path.join('./data/data_list', args.val_query_list)

    # parse train log directory 
    hour = datetime.datetime.now().hour
    if hour < 10:
        hour = "0"+str(hour)
    else:
        hour = str(hour)
    minute = datetime.datetime.now().minute
    if minute < 10: 
        minute = "0"+str(minute)
    else:
        minute = str(minute)     
           
    train_log_dir = "train_" + args.network + "-" + data_type + "-" + txt_name + "-"\
         + str(datetime.datetime.now().year) \
         + str(datetime.datetime.now().month) \
         + str(datetime.datetime.now().day)\
         + "_" + hour + minute 
    args.train_log_dir = os.path.join('data/train_logs/experiments_20210906', train_log_dir)
    
    # parse checkpoints directory
    ckpt_dir = os.path.join(args.train_log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    args.ckpt_dir = ckpt_dir
    args.pretrained = os.path.join('data/train_logs', args.pretrained)

    # parse code backup directory
    code_backup_dir = os.path.join(args.train_log_dir, 'codes')
    os.makedirs(code_backup_dir, exist_ok=True)
    subprocess.call('cp -r ./models ./trainers ./datasets ./utils {}'.format(code_backup_dir), shell=True)
    subprocess.call('cp -r ./{} {}'.format(__file__, code_backup_dir), shell=True)
    
    # parse gpus 
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpus
    gpu_list = []
    for str_id in args.gpus.split(','):
        id = int(str_id)
        gpu_list.append(id)
    args.gpu_list = gpu_list

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    for key, value in vars(args).items():
        table_key.append(key)
        table_value.append(str(value))
    print_table([table_key, ["="] * len(vars(args).items()), table_value])

    # configure trainer and start training
    if args.network == 'ranet_model':
        trainer = trainer_segmentation_ranet(args)
    elif args.network == 'five_model':
        trainer = trainer_segmentation_fiveshot(args)
    elif args.network == 'one_model':
        trainer = trainer_segmentation_oneshot(args) 

    trainer.train()
    