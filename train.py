# -*- coding: utf-8 -*-
"""
Created on 2021/7/9

@author: Zhang Wenhao @galwayzhang
"""
import argparse
import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import eval_model
from utils.dataset import BasicDataset
from utils.utils.utils import get_time_str
from nets.Pnet import PN
#import imageio

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by x every y epochs"""
    lr = args.lr * (0.1 ** (epoch// 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(
        model: nn.Module,
        device: torch.device,
        epochs=5,
        batch_size=1,
        lr=0.00001,
        val_percent=0.1,
        save_checkpoint=True,
        checkpoint_interval=1,
        checkpoint_dir='checkpoints/',
        val_interval=1
):
    #基础设置
    #设置训练集和测试集！
    dataset = BasicDataset(root_dir='',good_only=True,is_train=True,crop_to=(128,128))

    dataset_another=BasicDataset(root_dir='',good_only=True,is_train=False,crop_to=(256,256))
    #val_set=BasicDataset(root_dir='',good_only=True,is_train=False,crop_to=(256,256))
    n_val = int(len(dataset_another) * val_percent)
    n_train = len(dataset_another) - n_val
    train_set, val_set = random_split(dataset_another, [n_train, n_val],generator=torch.Generator().manual_seed(42))

    #x% 的数据进行训练
    n_val = int(len(dataset) * 0.25)
    n_train = len(dataset) - n_val
    real_train_set,false_val_set=random_split(dataset, [n_train, n_val],generator=torch.Generator().manual_seed(42))

    #!!!全部或者部分！
    train_loader = DataLoader(real_train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    #记得这里改tensorboard的文件名
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_Based_linear_B_0.75')
    global_step = 0


    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4,momentum=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.1,patience=50)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=len(real_train_set), desc=f'Epoch {epoch+1}/{epochs}', unit='slice') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                masks_true = batch['mask']
   
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks_true = masks_true.to(device=device, dtype=torch.float32)
               
                masks_pred=model(imgs)
                

                loss = criterion(masks_pred, torch.squeeze(masks_true.long(),dim=1))
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                adjust_learning_rate(optimizer, epoch, args)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                #break

        if (epoch + 1) % val_interval == 0:
            dice_score = eval_model(model, val_loader, device)
            print(dice_score)
            #scheduler.step(dice_score)

            writer.add_scalar('dice', dice_score, global_step)
            #writer.add_scalar('dice1', dice_score1, global_step)
            #writer.add_scalar('dice1', dice_score1, global_step)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)


            # writer.add_images('images', imgs.cpu(), global_step)
            writer.add_images('masks/true', masks_true.cpu(), global_step)

            sf=nn.Softmax(dim=1)
            masks_pred=sf(masks_pred)
            masks_pred=((masks_pred>0.5).float()).cpu().numpy()

            masks_pred=torch.as_tensor(masks_pred)
            writer.add_images('masks/pred0',torch.unsqueeze(masks_pred[:,0,:,:],dim=1) , global_step)

        if save_checkpoint and (epoch+1) % checkpoint_interval == 0:
            if not os.path.exists(checkpoint_dir):
                try:
                    os.makedirs(checkpoint_dir)
                    logging.info('Checkpoint directory is created.')
                except OSError:
                    pass
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'CP_{get_time_str()}_epoch{epoch+1}.pth'))
            logging.info(f'Checkpoint {epoch+1} is saved.')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train models')

    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        default=False, help='Use cpu')
    parser.add_argument('-b', '--batch-size', dest='batchsize', type=int, nargs='?',
                        default=16, help='Batch size')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int,
                        default=100, help='Number of epochs')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('-l', '--lr', dest='lr', type=float, nargs='?',
                        default=0.001, help='Learning rate')
    parser.add_argument('-m', '--mode', dest='mode', type=str, choices=['be', 'qa', 'both'],
                        default='be', help='Choose train step')
    parser.add_argument('-s', '--save', dest='save', type=str,
                        default='', help='Dir path to save .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info(f'Device: {device}')

    model = PN()

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model is loaded from {args.load}.')

    model.to(device=device)

    train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batchsize,
        lr=args.lr,
        checkpoint_dir=os.path.join('checkpoints/', args.save)
    )
