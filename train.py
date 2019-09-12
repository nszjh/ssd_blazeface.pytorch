from data.celeba import *
# from utils.augmentations import SSDAugmentation
from layers.modules.multibox_loss_blaze import BlazeMultiBoxLoss
# from ssd import build_ssd
from blazeface import build_blazeface
from data.config import celeba

import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'CELEBA'],
                    type=str, help='VOC or COCO or CELEBA')
parser.add_argument('--dataset_root', default=CELEBA_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'CELEBA':
        # if args.dataset_root == CELEBA_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')
        cfg = celeba
        dataset = CelebaDetection(root=args.dataset_root)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    # ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # net = ssd_net
    blaze_net = build_blazeface('train')
    net = blaze_net

    if args.cuda:
        net = torch.nn.DataParallel(blaze_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        blaze_net.load_weights(args.resume)
    # else:
    #     vgg_weights = torch.load(args.save_folder + args.basenet)
    #     print('Loading base network...')
    #     blaze_net.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        # blaze_net.extras.apply(weights_init)
        # blaze_net.loc.apply(weights_init)
        # blaze_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = BlazeMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) # args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)

    with_landmark = ""
    save_count = 0
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)
        # print ("dataset: ", images.shape, targets.shape)
        
        zmask = targets[:, :, 10] == 0
        if torch.sum(zmask).data > 0:
            continue

        # print ("zmask", torch.sum(zmask))

        if args.cuda:
            images = Variable(images.cuda())
            # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            targets = Variable(targets.cuda())
        else:
            images = Variable(images)     
            # targets = [Variable(ann, volatile=True) for ann in targets]
            targets = Variable(targets.cuda())

        # forward
        t0 = time.time()
        out1  = net(images)
        out2 = out1[:, :, 10:12]
        priors = blaze_net.getPriors()

        # print ("output: ")
        # print (out1.shape)
        # print (out2.shape)
        # print (priors.shape)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_land = criterion((out1, out2, priors), targets)

        # print ("loss_l:", loss_l.shape)
        # print ("loss_c:", loss_c.shape)

        loss = loss_l + loss_c  # + loss_land
        loss_with_land = loss_land
        
        if (len(loss_with_land.data) > 0):
            with_landmark = "with_land_"
        
        # loss = loss.view(-1, 1)
        # loss_with_land = loss_with_land.view(-1, 1)
        # print ("loss:", loss, loss.shape)
        # print ("loss_with_land: ", loss_with_land.shape)

        loss = torch.cat([loss, loss_with_land], 1)
        # print ("loss:", loss, loss.shape)
        # loss = 0.1 * loss.mean()  + loss_with_land.mean()
        # loss = 0.8 * loss  + loss_with_land
        # loss.backward(torch.FloatTensor([loss.size(0), loss.size(1)]))
        loss.sum().backward()
        # loss_with_land.backward()

        optimizer.step()
        t1 = time.time()
        # print ("loss_l", loss_l.shape)
        loc_loss += loss_l.data[0]  #loss
        # conf_loss += loss_c.data[0] #loss


        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]))
            print('iter ' + repr(iteration) + ' || Loss: %.6f ||' % (loss.sum().data)) 

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration / 5000 > save_count:
            save_count += 1
            print('Saving state, iter:', iteration)
            torch.save(blaze_net.state_dict(), 'weights/blaze_face_' + with_landmark + 
                       repr(iteration) + "_loss_" + str(float(loss.sum().data)) + '.pth')

    torch.save(blaze_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')



def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)




def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
