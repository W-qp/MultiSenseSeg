# TODO:Train model
import argparse
import json
import logging
import math
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import BasicDataset
from utils.eval import eval_net
from utils.logger import logger
from utils.loss_function import ClsLoss
from utils.plot_train import *
from model.build_model.Build_MultiSenseSeg import Build_MultiSenseSeg

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start_time = time.time()


def get_args():
    parser = argparse.ArgumentParser(description='Train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_chans', type=str2tuple, default="3, 1", help='Channels of input images')
    parser.add_argument('--img_size', type=int, default=512, help='Images size')
    parser.add_argument('--json_path', type=str, default=r"./json/train.json", help='.json path')
    parser.add_argument('--net_type', type=str, default='MultiSenseSeg', help='Net type')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--warm_epochs', type=int, default=5, help='Number of warm epochs')
    parser.add_argument('--ref_idx', type=str, default='mIoU', help='Reference index type, using mIoU, mF1 or OA')
    parser.add_argument('--background', default=False, help='Whether to consider background classes')
    parser.add_argument('--batchsize', type=int, default=2, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=0.00016, help='learning rate starting value')
    parser.add_argument('--weight_decay', type=float, default=0.002, help='L2 regular term')
    parser.add_argument('--gpu_id', type=int, default=0, help='Number of gpu')
    parser.add_argument('--if_apex', default=False, help='Whether to enable Apex mixed precision training')
    parser.add_argument('--load', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--loss_weight', default=None, help='Loss weight for each class')
    return parser.parse_args()


def str2tuple(string):
    try:
        string = string.strip().strip('()[]')
        numbers = [int(x.strip()) for x in string.split(',')]
        return tuple(numbers)
    except:
        raise argparse.ArgumentTypeError('Must be a tuple of integers, e.g., "3, 1" or "(3, 1)"')


def create_path(path):
    try:
        os.mkdir(path)
        print(f'Created directory: {path}')
    except OSError:
        pass


def train_net(net_type, net, device, epochs, batch_size, warm_epochs, lr, weight_decay, img_shape, apex, ref, w):
    global best_miou_last, i
    train_dataset = BasicDataset(n_img, json_path, img_shape, query='train', if_transform=True)
    val_dataset = BasicDataset(n_img, json_path, img_shape, query='val', if_transform=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    global_step = 0
    logging.info(f'''Using device {device}
        Starting training:
                     Net type:           {net_type}
                     Epochs:             {epochs}
                     Batch size:         {batch_size}
                     Learning rate:      {lr}
                     L2:                 {weight_decay}
                     Reference index:    {ref}
                     With background:    {w}
                     Input img number:   {n_img}
                     Input channels:     {in_chans}         
                     Dataset size:       {len(train_dataset) + len(val_dataset)}
                     Training size:      {len(train_dataset)}
                     Validation size:    {len(val_dataset)}
                     Device:             {device.type}
                     Apex:               {apex}\n''')

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if apex:
        model, optimizer = amp.initialize(net, optimizer, opt_level="O0")
    e = warm_epochs
    lf = lambda x: (((1 + math.cos((x - e + 1) * math.pi / (epochs - e))) / 2) ** 1.0) * 0.95 + 0.02 if x >= e else 0.95 / (e - 1) * x + 0.02
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)
    scheduler.last_epoch = global_step
    lrs, losses, miou, mf1, oa = [], [], [], [], []
    best_ref_idx = 0
    a_begin, a_end = 0.5, 0.4
    factor = lambda x: a_begin + (a_end - a_begin) / epochs * x

    for epoch in range(epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch= {epoch+1},  lr= {cur_lr}')
        net.train()
        epoch_loss = 0
        a = factor(epoch)

        for i, batch in enumerate(train_loader):
            train_imgs = []
            for n in range(1, 1+n_img):
                imgs = batch[f'image{n}']
                imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
                train_imgs.append(imgs)

            true_masks = batch['mask']
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type, non_blocking=True)

            optimizer.zero_grad()
            masks_pred = net(train_imgs)
            train_imgs.clear() if isinstance(train_imgs, list) else None

            if net.n_classes > 1:
                loss_weight = torch.FloatTensor(args.loss_weight).cuda() if args.loss_weight is not None else None
                if net.aux:
                    criterion_main = ClsLoss(weight=loss_weight)
                    criterion_aux = ClsLoss(weight=loss_weight)
                    loss_main = criterion_main(masks_pred[0], torch.squeeze(true_masks, dim=1))
                    loss_aux = criterion_aux(masks_pred[1], torch.squeeze(true_masks, dim=1))
                    loss = loss_main + loss_aux * a
                else:
                    criterion = ClsLoss(weight=loss_weight)
                    loss = criterion(masks_pred, torch.squeeze(true_masks, dim=1))
                    a = 0
            else:
                if net.aux:
                    criterion_main = nn.BCEWithLogitsLoss()
                    criterion_aux = nn.BCEWithLogitsLoss()
                    loss_main = criterion_main(masks_pred[0], true_masks)
                    loss_aux = criterion_aux(masks_pred[1], true_masks)
                    loss = loss_main + loss_aux * a
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(masks_pred, true_masks)
                    a = 0
            epoch_loss += loss.item()

            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            global_step += 1

        torch.cuda.empty_cache()
        logging.info('Val set:')
        mIoU, mF1, OA, confusion, nc = eval_net(net, val_loader, device, class_json, n_img, w)
        torch.cuda.empty_cache()
        miou.append(mIoU)
        mf1.append(mF1)
        oa.append(OA)

        epoch_loss_mean = epoch_loss / (i + 1)
        logging.info(f'Epoch loss: {epoch_loss_mean}, \talpha= {a}')
        losses.append(epoch_loss_mean)

        scheduler.step()
        lrs.append(cur_lr)

        if ref == 'mIoU':
            ref_idx = mIoU
        elif ref == 'mF1':
            ref_idx = mF1
        elif ref == 'OA':
            ref_idx = OA
        else:
            raise NotImplementedError(f"Reference index type:'{ref}' does not exist!")

        if ref_idx > best_ref_idx:
            torch.save(net.state_dict(), runs + f"ckpts/{net_type}_best.pth")
            best_ref_idx = ref_idx
            logging.info(f'\tEpoch {epoch + 1} weight saved!  {ref} result(val) = {ref_idx}')
            confusion.plot(runs + "Confusion matrix", n_classes=nc) if w else None
        logging.info('The run time is: {} hours\n'.format(round((time.time() - start_time) / 3600, 3)))

        # evaluate training set accuracy
        if epoch + 1 == epochs:
            logging.info('Computing training set accuracy...\nTrain set:')
            torch.cuda.empty_cache()
            eval_net(net, train_loader, device, class_json, n_img, w)
            torch.cuda.empty_cache()
            logging.info('The run time is: {} hours\n'.format(round((time.time() - start_time) / 3600, 3)))

    plot_miou(miou, runs)
    plot_mf1(mf1, runs)
    plot_oa(oa, runs)
    plot_loss(losses, runs)
    plot_lr(lrs, runs)


if __name__ == '__main__':
    args = get_args()
    create_path("./runs/")
    json_path = args.json_path
    in_chans = args.in_chans
    n_img = len(in_chans) if isinstance(in_chans, tuple) else 1
    gpu_id = args.gpu_id
    net_type = args.net_type
    if_apex = args.if_apex
    if if_apex:
        try:
            from apex import amp
        except ImportError:
            print("Missing apex package, defaulting to not using apex!!")
            if_apex = False

    img_size = args.img_size
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert os.path.exists(json_path), f"Cannot find {json_path} file!"
    json_dict = json.load(open(json_path, 'r'))
    runs = json_dict['runs']
    create_path(runs)
    class_json = json_dict['classes']
    n_classes = len(json.load(open(class_json, 'r')))
    logger(net_type, runs)
    logging.info(f'Run path: {runs}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if net_type == 'MultiSenseSeg':
        net = Build_MultiSenseSeg(n_classes=n_classes, in_chans=in_chans)
    else:
        raise NotImplementedError(f"net type:'{net_type}' does not exist, please check the 'net_type' arg!")

    logging.info(f'Network: {net.n_classes} output channels (classes)')
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device=device, non_blocking=True)
    if args.load is not None:
        net.eval()
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    create_path(runs + "ckpts")

    # faster convolutions, but more memory
    from torch.backends import cudnn
    cudnn.benchmark = True

    create_path(runs + "ckpts")
    train_net(net_type=net_type, net=net, epochs=args.epochs, batch_size=args.batchsize, warm_epochs=args.warm_epochs, lr=args.lr,
              weight_decay=args.weight_decay, device=device, img_shape=img_size, apex=if_apex, ref=args.ref_idx, w=args.background)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info('The entire {} model run time is: {} hours.'.format(net_type, round(run_time / 3600, 2)))
