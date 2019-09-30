import torch
import torch.nn as nn
from torch.autograd import Variable
from layers.functions.prior_box import *
from data.config import voc, coco, celeba
from layers.box_utils import decode, nms
from layers.functions.detection import Detect
import os.path
import numpy as np


def load_mnn_output():
    load_out = np.loadtxt("output.txt").astype(np.float32)
    out = torch.from_numpy(load_out)
    out = out.view(1, 896, 16) 
    return out


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)


class BlazeBlock(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride>1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)

class DoubleBlazeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.apply(weights_init)


    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)



# copy from dface,  2019-7-29   loss funtion
class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()


    def cls_loss(self,gt_label,pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label,0)
        valid_gt_label = torch.masked_select(gt_label,mask)
        valid_pred_label = torch.masked_select(pred_label,mask)
        return self.loss_cls(valid_pred_label,valid_gt_label)*self.cls_factor


    def box_loss(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        #get the mask element which != 0
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor


    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label,-2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)*self.land_factor





class BlazeFace(nn.Module):
    def __init__(self, phase, num_classes):
        super(BlazeFace, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.blazeBlock = nn.Sequential(
            BlazeBlock(in_channels=16, out_channels=16),
            # BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=16, out_channels=32, stride=2),
            # BlazeBlock(in_channels=48, out_channels=48),
            BlazeBlock(in_channels=32, out_channels=32),
        )

        self.doubleBlazeBlock = nn.Sequential(
            DoubleBlazeBlock(in_channels=32, out_channels=64, mid_channels=16, stride=2),
            # DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlock(in_channels=64, out_channels=64, mid_channels=16),
        )


        self.conv2d_8x8_classificators = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1),
            nn.BatchNorm2d(6),
        )

        self.conv2d_16x16_classificators = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
        )

        self.conv2d_8x8_regressors = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )

        self.conv2d_16x16_regressors = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
        )

        self.phase = phase
        self.cfg = celeba
        self.priorbox = PriorBox(self.cfg)
        self.num_classes = num_classes
        # self.priors = Variable(self.priorbox.forward(), volatile=True)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

        with torch.no_grad():
            self.priors = self.priorbox.forward()

        print(self.priors.shape)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print ("x1:", x.shape)

        x = self.firstconv(x)
        # print ("x :", x.shape)

        x = self.blazeBlock(x)
        # print ("x :", x.shape)

        y1 = self.doubleBlazeBlock(x)

        # print ("y1:", y1.shape)
        batch = y1.size(0)
        # print (self.conv2d_16x16_classificators(y1).view( 512, 1).shape)
        # classificators = torch.cat([self.conv2d_16x16_classificators(y1).view( 512, 1), self.conv2d_8x8_classificators(y2).view( 384, 1)])
        output = self.conv2d_16x16_regressors(y1).view(batch,  512, 16)
        # print ("output:", output.shape)
        # output = load_mnn_output()

        if self.phase == "test":
            output = self.detect(
                output,                # loc preds
                # self.softmax(classificators.view(batch, 896, 1)),                # conf preds
                # classificators.view(batch, 896, 1),
                self.softmax(output[:, :, 10:12]), self.softmax(output[:, :, 4:6]), 
                # self.priors.type(type(x.data))                  # default boxes
                self.priors
            )
        # else:
        #     output = (
        #         output,
        #         # classificators.view(batch, 896, 1),
        #         output[:, :, 10:12],
        #         self.priors
        #     )


        # if self.phase == "test":
        #     output = self.detect(
        #         regressors.view(batch, 896, 16),                # loc preds
        #         # self.softmax(classificators.view(batch, 896, 1)),                # conf preds
        #         # classificators.view(batch, 896, 1),
        #         self.softmax(regressors.view(batch, 896, 16)[:, :, 10:12]),
        #         self.priors.type(type(x.data))                  # default boxes
        #     )
        # else:
        #     output = (
        #         regressors.view(batch, 896, 16),
        #         # classificators.view(batch, 896, 1),
        #         regressors.view(batch, 896, 16)[:, :, 10:12],
        #         self.priors
        #     )
        return output

    def getPriors(self):
        return self.priors


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_blazeface(phase, size=128, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    return BlazeFace(phase, num_classes)


if __name__=='__main__':
    model = build_blazeface("train")
    print(model)

    input = torch.randn(1, 2, 128, 128)
    out1 = model(input)
    print(out1.shape)
    # print(out2.shape)