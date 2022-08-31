
import os
import sys
import time
import random
import numpy as np
import copy
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from visdom import Visdom

from models import model_dict_segmentation
from datasets.data_loader import ER_DataLoader, ER_DataLoader_Pair
from datasets.metrics import evaluate, evaluate_single_class
from utils.AvgMeter import averageMeter
from utils.optimize import configure_optimizers
from utils.utils import init_weights, decay_threshold
                           

class trainer_segmentation_oneshot(nn.Module):
    def __init__(self, params=None):
        super(trainer_segmentation_oneshot, self).__init__()
        self.args = params
        self.global_step = 0
        self.current_step = 0
        if self.args.n_classes > 1:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.BCELoss()

        self.viz = Visdom(env="train_model", server="http://127.0.0.1", port=self.args.port)
        assert self.viz.check_connection()
        print("PyTorch Version: ", torch.__version__)
        
    def _dataloader(self, support_list, query_list, skl=False, split='train', bz=2):
        dataset = ER_DataLoader_Pair(support_list=support_list,
                                     query_list=query_list,
                                     split=split,
                                     equalize=self.args.equalize,
                                     standardization=self.args.std,
                                     skl=skl,
                                     img_size=self.args.img_size)
        shuffle = (split=='train')
        dataloader = DataLoader(dataset, batch_size=bz, num_workers=2, shuffle=shuffle, drop_last=shuffle)

        return dataloader

    def train_one_epoch(self, epoch):
        t0 = 0.0
        self.model.train()
        for inputs in self.train_data_loader:
            self.global_step += 1
            self.current_step +=1
            
            t1 = time.time()
            data_dict = {"support image": inputs["support image"].cuda(),
                         "support mask":  inputs["support mask"].cuda(),
                         'query image':   inputs['query image'].cuda(),
                         'query mask':    inputs['query mask'].cuda()}
            if self.args.train_skl:
                data_dict['support skl'] = inputs['support skl'].cuda()
                data_dict['query skl'] = inputs['query skl'].cuda()

            logits, _, _ = self.model(data_dict, skl=self.args.train_skl)
            if self.args.n_classes > 1:
                loss_val = self.loss_func(logits, data_dict['query mask'].long())
            else:
                if self.args.semi:
                    ##### adjust threshold
                    if epoch < 4:
                        T = 0.4
                    else: 
                        T = 0.6
                    # T = decay_threshold(0.3, epoch, self.args.epochs)
                    predict_mask = logits > T
                    #####
                    loss_val = self.loss_func(logits, predict_mask.float())
                else:
                    loss_val = self.loss_func(logits, data_dict['query mask'].float())
            
            if loss_val < 0:
                breakpoint()

            self.optimizer_backbone.zero_grad()
            self.optimizer_cls.zero_grad()

            loss_val.backward()

            self.optimizer_backbone.step()
            self.optimizer_cls.step()

            t0 += (time.time() - t1)

            if self.global_step % self.args.show_results_every_x_steps == 0:
                message = "Epoch: %d Step: %d LR1: %.6f LR2: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, self.lr_scheduler_backbone.get_lr()[-1], self.lr_scheduler_cls.get_lr()[-1], loss_val, t0, self.current_step)
                print("==> %s" % (message))
                self.current_step = 0
                t0 = 0.0

                self.viz.line(Y=[loss_val.data.cpu().numpy()], X=[self.global_step], win='train loss', update='append', name='total loss',
                        opts=dict(title="Training Loss", xlabel="Step", ylabel="Total Loss"))
               
                support_image = data_dict['support image'].data.cpu().numpy()
                query_image = data_dict['query image'].data.cpu().numpy()
                if self.args.train_skl:
                    support_mask = data_dict['support skl'].data.cpu().numpy()
                else:
                    support_mask = data_dict['support mask'].data.cpu().numpy()

                if self.args.semi:
                    query_mask = predict_mask.long().data.cpu().numpy()
                else:
                    query_mask = data_dict['query mask'].data.cpu().numpy()

                if self.args.n_classes > 1:
                    predicts = torch.argmax(logits, dim=1).squeeze().data.cpu().numpy()
                else:
                    predicts = logits.squeeze(dim=1).data.cpu().numpy()

                self.viz.heatmap(support_image[0,0,::-1,:], win="support image", opts={"title": "support image"})
                self.viz.heatmap(support_mask[0,0,::-1,:], win="support_mask", opts={"title": "support_mask"})
                self.viz.heatmap(query_image[0,0,::-1,:], win="query_image", opts={"title": "query_image"})
                self.viz.heatmap(query_mask[0,0,::-1,:], win="query_mask", opts={"title": "query_mask"})
                self.viz.heatmap(predicts[0,::-1,:], win="predicts", opts={"title": "predicts"})
                
                with open(os.path.join(self.args.train_log_dir, "train_loss.txt"), "a+") as fid:
                    fid.write("%.10f\n" % (loss_val.data.cpu().numpy()))


    def val_one_epoch(self, epoch):
        with torch.no_grad():
            self.model.eval()

            for i, inputs in enumerate(self.val_data_loader):
                data_dict = {"support image": inputs["support image"].cuda(),
                            "support mask":  inputs["support mask"].cuda(),
                            'query image':   inputs['query image'].cuda(),
                            'query mask':    inputs['query mask'].cuda()}
                
                if self.args.val_skl:
                    data_dict['support skl'] = data_dict['support skl'].cuda()
                
                logits, _, _ = self.model(data_dict, skl=self.args.val_skl)
                
                if self.args.n_classes > 1:
                    mask_out = torch.softmax(logits, dim=1)[:,1,...]
                else:
                    mask_out = logits

                if i == 0: 
                    predictions = mask_out
                    groundtruths = data_dict['query mask']
                else:
                    predictions = torch.cat((predictions, mask_out), dim=0)
                    groundtruths = torch.cat((groundtruths, data_dict['query mask']), dim=0)

            support_image = data_dict['support image'].data.cpu().numpy()
            query_image = data_dict['query image'].data.cpu().numpy()
            query_mask = data_dict["query mask"].data.cpu().numpy()
            if self.args.val_skl:
                support_mask = data_dict["support skl"].data.cpu().numpy()
            else:
                support_mask = data_dict["support mask"].data.cpu().numpy()
                
            mask_out = mask_out.squeeze(dim=1).data.cpu().numpy()

            self.viz.heatmap(support_image[0,0,::-1,:], win="test support image", opts={"title": "test support image"})
            self.viz.heatmap(support_mask[0,0,::-1,:], win="test support_mask", opts={"title": "test support_mask"})
            self.viz.heatmap(query_image[0,0,::-1,:], win="test query_image", opts={"title": "test query_image"})
            self.viz.heatmap(query_mask[0,0,::-1,:], win="test query_mask", opts={"title": "test query_mask"})
            self.viz.heatmap(mask_out[0,::-1,:], win="test predicts", opts={"title": "test predicts"})

            predictions = predictions.data.cpu().numpy().flatten()
            groundtruths = groundtruths.data.cpu().numpy().flatten()
            groundtruths[groundtruths==255] = 1

            threshold, f1, iou, _, _, _ = evaluate(predictions, groundtruths, interval=0.01, mode='iou')
            print("==> Epoch: %d IOU %.4f." % (epoch+1, iou))

        self.viz.line(Y=torch.Tensor([iou]),
                 X=torch.Tensor([epoch + 1]),
                 win='val iou',
                 update='append',
                 opts=dict(title="Validation IOU", xlabel="Epoch", ylabel="IOU"))

        torch.cuda.empty_cache()
        
        return iou, f1
    
    
    def train(self):
        print("==> Create model.")
        if self.args.network in model_dict_segmentation.keys():
            self.model = model_dict_segmentation[self.args.network](n_classes=self.args.n_classes, in_channels=1)
        else:
            print("No support model type.")
            sys.exit(1)     

        print("==> Initialize model from random state.")
        init_weights(self.model)    

        if os.path.exists(self.args.pretrained) and os.path.isfile(self.args.pretrained):
            print("==> Initialize model from: %s." % (self.args.pretrained))
            pre_trained = torch.load(self.args.pretrained)['model_state_dict']
            model_dict = self.model.state_dict()

            words = self.args.pretrained.split('/')
            if words[2].startswith('oneshot_seg'):       
                pretrained_dict = {k: v for k, v in pre_trained.items() if
                                   k in model_dict and v.size() == model_dict[k].size()}
            else:
                pretrained_dict = {'feature.'+k: v for k, v in pre_trained.items() if 'feature.'+k in model_dict}
                for k in model_dict: 
                    if k.startswith('feature.up5'):
                        pretrained_dict[k] = pre_trained[k[8:].replace('up5', 'up1')]
                    if k.startswith('feature.up6'):
                        pretrained_dict[k] = pre_trained[k[8:].replace('up6', 'up2')]
                    if k.startswith('feature.up7'):
                        pretrained_dict[k] = pre_trained[k[8:].replace('up7', 'up3')]
                    if k.startswith('feature.up8'):
                        pretrained_dict[k] = pre_trained[k[8:].replace('up8', 'up4')]                                                                     
            
            print(pretrained_dict.keys())

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)      

        feature_extractor = []
        few_seg_head = []
        for name, param in self.model.named_parameters():
            if not name.startswith('feature'):
                param.requires_grad = True
                few_seg_head.append(param)
            else:
                feature_extractor.append(param)
                if self.args.freeze == True:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        print("==> List learnable parameters")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print("\t{}".format(name))
        
        if len(self.args.gpu_list) == 1:
            torch.cuda.set_device(self.args.gpu_list[0])
        else:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_list)
        
        self.model.cuda()
        
        print("==> Load data.")
        self.train_data_loader = self._dataloader(self.args.train_support_list, 
                                                  self.args.train_query_list,
                                                  split='train', 
                                                  bz=self.args.batch_size,
                                                  skl=self.args.train_skl)
        self.val_data_loader = self._dataloader(self.args.val_support_list, 
                                                self.args.val_query_list,
                                                split='val', 
                                                bz=self.args.val_bz,
                                                skl=self.args.val_skl)

        print("==> Configure optimizer.")
        self.optimizer_backbone, self.lr_scheduler_backbone = configure_optimizers(feature_extractor, self.args.init_lr * self.args.init_lr_decay, self.args.weight_decay, self.args.gamma, self.args.lr_decay_every_x_epochs)
        self.optimizer_cls, self.lr_scheduler_cls = configure_optimizers(few_seg_head, self.args.init_lr, self.args.weight_decay, self.args.gamma, self.args.lr_decay_every_x_epochs)
        
        print("==> Start training")
        
        self.log_txt = os.path.join(self.args.train_log_dir, "train_val.txt")

        since = time.time()
        best_iou = 0.0
        for epoch in range(self.args.epochs):
            
            self.train_one_epoch(epoch)
            epoch_iou, epoch_f1 = self.val_one_epoch(epoch)
            
            with open(os.path.join(self.args.train_log_dir, "train_iou.txt"), "a+") as fid:
                fid.write("%.6f\n" % (epoch_iou))
            with open(os.path.join(self.args.train_log_dir, "train_f1.txt"), "a+") as fid:
                fid.write("%.6f\n" % (epoch_f1))
                
            if epoch_iou > best_iou:
                best_iou = epoch_iou
                if len(self.args.gpu_list) > 1:
                    torch.save({'epoch': epoch+1,
                            'model_state_dict': self.model.module.state_dict()},
                            os.path.join(self.args.ckpt_dir, "checkpoints_best.pth"))
                else:
                    torch.save({'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict()},
                            os.path.join(self.args.ckpt_dir, "checkpoints_best.pth"))

            if (epoch+1) % self.args.save_every_x_epochs == 0:
                if len(self.args.gpu_list) > 1:
                    torch.save({'epoch': epoch+1,
                                'model_state_dict': self.model.module.state_dict()},
                                os.path.join(self.args.ckpt_dir, "checkpoints_epoch_" + str(epoch + 1) + ".pth"))
                else:
                    torch.save({'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict()},
                            os.path.join(self.args.ckpt_dir, "checkpoints_epoch_" + str(epoch + 1) + ".pth"))
            
            self.lr_scheduler_backbone.step()
            self.lr_scheduler_cls.step()
            
        print("==> Runtime: %.2f minutes." % ((time.time()-since)/60.0))
            
        
