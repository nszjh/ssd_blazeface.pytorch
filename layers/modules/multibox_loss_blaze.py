# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.config import celeba as cfg
from ..box_utils import match, match1, log_sum_exp

UseLandmark = 2

class BlazeMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(BlazeMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data1, conf_data, priors = predictions
        loc_data = loc_data1[:, :, 12:]

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes


        # print ("loc_data", loc_data.shape)
        # print ("conf_data",  conf_data.shape)
        # print ("targets",  targets.shape)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        # print ("num_priors",  num_priors.shape)
        # print ("loc_t",  loc_t.shape)
        # print ("conf_t",  conf_t.shape)

        label_test = torch.ones(num, 1,  1)

        label_test[:, 0, :] = 0


        if UseLandmark == 1:
            landmark_data = loc_data1[:, :, :10]
            landmark_t = torch.Tensor(num, num_priors, 10)
        elif UseLandmark == 2:
            loc_2_data = loc_data1[:, :, 6:10]
            loc_2_t = torch.Tensor(num, num_priors, 4)
            conf_2_t = torch.LongTensor(num, num_priors)
            conf_2_data = loc_data1[:, :, 4:6]


        # loc_t = torch.Tensor(num_priors, 4)

        for idx in range(num):  #遍历各个batch，即各张图片
            truths = targets[idx][:, 10:].data
            labels = label_test[idx][:, -1].data

            defaults = priors.data

            # print (labels.shape)
            # print (defaults.shape)
            if UseLandmark == 2:
                truths_2 = targets[idx][:, 6:10].data
                match(self.threshold, truths_2, defaults, self.variance, labels,
                  loc_2_t, conf_2_t, idx)
                  
                match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
                
            elif UseLandmark == 1:
                landmarks = targets[idx][:, :10].data
                match1(self.threshold, truths, defaults, self.variance, labels, landmarks, 
                  loc_t, conf_t, landmark_t, idx)
            else:
                match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)


        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            if UseLandmark == 1:
                landmark_t = landmark_t.cuda()
            elif UseLandmark == 2:
                loc_2_t = loc_t.cuda()
                conf_2_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        num_pos = pos.sum(dim=1, keepdim=True)

        # print ("loc_t:", loc_t.shape)
        # print ("conf_t:", conf_t, conf_t.shape)
        # print ("pos:", pos, pos.shape)
        # print ("landmark_t", landmark_t.shape)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print ("pos_idx: ", pos_idx,  pos_idx.shape)
        # print ("pos: ", pos.shape)

        if UseLandmark == 2:
            loc_2_t = Variable(loc_2_t, requires_grad=False)
            conf_2_t = Variable(conf_2_t, requires_grad=False)
            pos_2 = conf_2_t > 0
            num_pos_2 = pos_2.sum(dim=1, keepdim=True)
            pos_idx_2 = pos_2.unsqueeze(pos_2.dim()).expand_as(loc_2_data)


        ############### add landmark loss ###############
        if UseLandmark == 1:
            pos_land_idx = pos.unsqueeze(pos.dim()).expand_as(landmark_data)
            landmark_p = landmark_data[pos_land_idx].view(-1, 10)
            landmark_t = landmark_t[pos_land_idx].view(-1, 10)
            loss_fn_land = torch.nn.MSELoss(reduction='none')
            loss_land = loss_fn_land(landmark_p, landmark_t)
        ############### add landmark loss ###############


        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # print ("loc_p: ", loc_p.shape)
        # print ("loc_t: ", loc_t)
        # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        loss_l = loss_fn(loc_p, loc_t)

        # print ("loss_l: ", loss_l)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # batch_conf = conf_data.view(self.num_classes, -1)
        # batch_conf = conf_data.view(1, -1)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(1, -1))



        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now

        _, loss_idx = loss_c.sort(1, descending=True)

        # print ("loss:", loss_c, loss_idx)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # print("pos_idx:", pos_idx)
        # print("neg_idx:", neg_idx)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # print (conf_p, conf_p.shape)
        # print ("targets_weighted:" , targets_weighted, targets_weighted.shape)

        loss_c_fn = torch.nn.CrossEntropyLoss()
        loss_c = F.cross_entropy(conf_p, targets_weighted)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        if UseLandmark == 1:
            loss_land /= N

        elif UseLandmark == 2:
            loc_2_p = loc_2_data[pos_idx_2].view(-1, 4)
            loc_2_t = loc_2_t[pos_idx_2].view(-1, 4)
            # print ("loc_p: ", loc_p.shape)
            # print ("loc_t: ", loc_t)
            # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
            loss_fn = torch.nn.SmoothL1Loss(reduction='none')
            loss_2_l = loss_fn(loc_2_p, loc_2_t)

            # print ("loss_l: ", loss_l)
            # Compute max conf across batch for hard negative mining
            batch_2_conf = conf_2_data.view(-1, self.num_classes)
            # batch_conf = conf_data.view(self.num_classes, -1)
            # batch_conf = conf_data.view(1, -1)
            loss_2_c = log_sum_exp(batch_2_conf) - batch_2_conf.gather(1, conf_2_t.view(-1, 1))
            # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(1, -1))



            # Hard Negative Mining
            loss_2_c = loss_2_c.view(num, -1)
            loss_2_c[pos_2] = 0  # filter out pos boxes for now

            _, loss_idx = loss_2_c.sort(1, descending=True)

            # print ("loss:", loss_c, loss_idx)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos_2.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos_2.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_2_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_2_data)

            # print("pos_idx:", pos_idx)
            # print("neg_idx:", neg_idx)
            conf_p = conf_2_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_2_t[(pos+neg).gt(0)]
            # print (conf_p, conf_p.shape)
            # print ("targets_weighted:" , targets_weighted, targets_weighted.shape)

            loss_c_fn = torch.nn.CrossEntropyLoss()
            loss_2_c = F.cross_entropy(conf_p, targets_weighted)

            # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
            N = num_pos.data.sum()
            loss_2_l /= N
            loss_2_c /= N
        else:
            loss_land = -loss_c
        
        # print ("loss_c: ", loss_c, loss_l)

        if UseLandmark == 2:
            return loss_l, loss_c, [loss_2_l, loss_2_c]
        else:    
            return loss_l, loss_c, loss_land

    