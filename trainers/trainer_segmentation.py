
import os
import sys
import time
import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from visdom import Visdom

from models.unet import UNet

from datasets.data_loader import ER_DataLoader
from datasets.metrics import evaluate
from utils.optimize import configure_optimizers, soft_iou_loss, weighted_edge_loss, DiceLoss, FocalLoss2d
from utils.utils import init_weights


class trainer_segmentation(nn.Module):
    def __init__(self, params=None):
        super(trainer_segmentation, self).__init__()
        self.args = params
        self.global_step = 0
        self.current_step = 0
        self.data_type = os.path.splitext(os.path.split(self.args.train_data_list)[-1])[0]
        self.viz = Visdom(env="train_model", server="http://127.0.0.1", port=self.args.port)
        assert self.viz.check_connection()

        print("PyTorch Version: ", torch.__version__)
        
    def _dataloader(self, datalist, skl, split='train'):
        dataset = ER_DataLoader(img_list=datalist, split=split, equalize=self.args.equalize, standardization=self.args.std, skl=skl)
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=16, shuffle=shuffle)
        return dataloader

    def train_one_epoch(self, epoch):
        t0 = 0.0
        self.model.train()
        for inputs in self.train_data_loader:
            self.global_step += 1
            self.current_step +=1
            
            t1 = time.time()

            if self.args.skl:
                images, masks = inputs["image"].cuda(), inputs["skl"].cuda()
            else:
                images, masks = inputs["image"].cuda(), inputs["mask"].cuda()

            mask_out = self.model(images)

            if self.args.semi == True:
                masks = (mask_out > 0.3).long()

            if self.args.segmentation_loss == "iou":
                total_loss = soft_iou_loss(mask_out, masks)
            elif self.args.segmentation_loss == "bce":
                total_loss = F.binary_cross_entropy(mask_out, masks)
            elif self.args.segmentation_loss == "dice":
                total_loss = DiceLoss()(mask_out, masks)
            elif self.args.segmentation_loss == "focal":
                total_loss = FocalLoss2d()(mask_out, masks)
            else:
                print("Loss function is not supported.")
                sys.exit(1)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            t0 += (time.time() - t1)

            if self.global_step % self.args.show_results_every_x_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, self.lr_scheduler.get_lr()[-1], total_loss, t0, self.current_step)
                print("==> %s" % (message))
                self.current_step = 0
                t0 = 0.0

                self.viz.line(Y=[total_loss.data.cpu().numpy()], X=[self.global_step], win='train loss', update='append', name='total loss',
                        opts=dict(title="Training Loss", xlabel="Step", ylabel="Total Loss"))

                orig_images = inputs["orig_image"]
                input_img = orig_images[0,...].data.cpu().numpy()
                self.viz.heatmap(input_img, win="train_image", opts={"title": "Train Image"})
                    
                raw_mask = masks[0,0,...].data.cpu().numpy() * 255
                raw_mask = raw_mask.astype(np.uint8)
                self.viz.image(raw_mask[::-1,:], win="train_mask", opts={"title": "Train Mask"})

                prd_seg_score = mask_out[0, 0, ...].data.cpu().numpy()
                self.viz.heatmap(prd_seg_score, win="train_pred_score", opts={'title': "Train Predict Score", "colormap": "viridis"})
                
                with open(os.path.join(self.args.train_log_dir, "train_loss.txt"), "a+") as fid:
                    fid.write("%.10f\n" % (total_loss.data.cpu().numpy()))



    def val_one_epoch(self, epoch):
        with torch.no_grad():
            self.model.eval()

            for i, inputs in enumerate(self.val_data_loader):
                images, masks = inputs["image"].cuda(), inputs["mask"].cuda()
                
                train_file = os.path.split(self.args.train_data_list)[-1]
                val_file = os.path.split(self.args.val_data_list)[-1]
                if train_file.startswith("tubule") and val_file.startswith("sheet"):
                    masks[masks==2] = 1 # convert sheet into tubule
                elif train_file.startswith("sheet") and val_file.startswith("sheet"):
                    masks[masks==2] = 0

                mask_out = self.model(images)

                if i == 0: 
                    predictions = mask_out
                    groundtruths = masks 
                else:
                    predictions = torch.cat((predictions, mask_out), dim=0)
                    groundtruths = torch.cat((groundtruths, masks), dim=0)

            predictions = predictions.data.cpu().numpy().flatten()
            groundtruths = groundtruths.data.cpu().numpy().flatten()

            orig_images = inputs["orig_image"]
            input_img = orig_images[0,...].data.cpu().numpy()
            self.viz.heatmap(input_img, win="val_image", opts={"title": "Val Image"})
                
            raw_mask = masks[0,0,...].data.cpu().numpy() * 255
            raw_mask = raw_mask.astype(np.uint8)
            self.viz.image(raw_mask[::-1,:], win="val_mask", opts={"title": "Val Mask"})

            prd_seg_score = mask_out[0, 0, ...].data.cpu().numpy()
            self.viz.heatmap(prd_seg_score, win="val_pred_score", opts={'title': "Val Predict Score", "colormap": "viridis"})
            
            best_threshold, best_f1, best_iou, _, _, _ = evaluate(predictions, groundtruths, interval=0.02, mode='iou')

        self.viz.line(Y=torch.Tensor([best_iou]),
                 X=torch.Tensor([epoch + 1]),
                 win='val iou',
                 update='append',
                 opts=dict(title="Validation IOU", xlabel="Epoch", ylabel="IOU"))

        print("==> Epoch: %d Evaluation Threshold %.2f IOU %.4f." % (epoch+1, best_threshold, best_iou))
        torch.cuda.empty_cache()
        return best_iou, best_f1


    def train(self):
        print("==> Create model.")

        if self.args.network in model_dict_segmentation.keys():
            self.model = model_dict_segmentation[self.args.network](base_dim=self.args.base_dim, n_classes=1, input_channels=1)
        else:
            print("No support model type.")
            sys.exit(1)            
            
        print("==> Initialize model from random state.")
        init_weights(self.model)

        if os.path.exists(self.args.pretrained) and os.path.isfile(self.args.pretrained):
            print("==> Initialize model from: %s." % (self.args.pretrained))
            pre_trained = torch.load(self.args.pretrained)['model_state_dict']
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pre_trained.items() if
                               k in model_dict and v.size() == model_dict[k].size()}
            print(pretrained_dict)
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        for name, param in self.model.named_parameters():
            if self.args.freeze == True:
                if name.startswith('out'): 
                    param.requires_grad = True  
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        print("==> List learnable parameters")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print("\t{}".format(name))        

        if len(self.args.gpu_list) == 1:
            torch.cuda.set_device(self.args.gpu_list[0])
            self.model.cuda()
        else:
            self.model.cuda()     
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_list)

        print("==> Load data.")
        self.train_data_loader = self._dataloader(self.args.train_data_list, skl=self.args.skl, split='train')
        self.val_data_loader = self._dataloader(self.args.val_data_list, skl=False, split='val')

        print("==> Configure optimizer.")
        self.optimizer, self.lr_scheduler = configure_optimizers(self.model.parameters(), self.args.init_lr, self.args.weight_decay, self.args.gamma, self.args.lr_decay_every_x_epochs)
        
        print("==> Start training")
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
            self.lr_scheduler.step()
            
        print("==> Runtime: %.2f minutes." % ((time.time()-since)/60.0))
            
        