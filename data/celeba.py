"""CELEBA  Dataset 
"""
from   config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import dlib


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
CELEBA_ROOT = osp.join(HOME, "/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/celeba_align/")
USE_PRE_DETECT = 0

class CelebaAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        i = 0
        while i < target.size:
            if i >= 10:
                break

            if (i % 2) == 0:
                target[0, i] = int(target[0, i] * 128 / 178)
            else:
                target[0, i] = int(target[0, i] * 128 / 218)
            i = i + 1

        # print (target)
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CelebaDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                #  image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=CelebaAnnotationTransform(),
                 dataset_name='CELEBA'):
        self.root = root
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        landmark_file = osp.join(root, 'Anno', 'list_landmarks_align_celeba.txt')
        list_bbox_celeba_file = osp.join(root, 'Anno', 'list_bbox_celeba_align.txt')

        self.landmarks = np.loadtxt(landmark_file, skiprows=2, usecols=range(1, 11))
        self.bbox      = np.loadtxt(list_bbox_celeba_file, skiprows=2, usecols=range(1, 5))

        # self._imgpath = osp.join('%s', 'pre_img_align_celeba/celebA', '%s.jpg')
        self.ids = list()
        # for (year, name) in image_sets:
        for line in open(osp.join(root, 'Anno', 'identity_CelebA.txt')):
            self.ids.append(line.split(' ')[0])

        if USE_PRE_DETECT == 1:
            self.detector = dlib.get_frontal_face_detector()
            self.totalLine = np.zeros([len(self.ids), 5])

        else:
            self.detector = 0
        

    def __getitem__(self, index):

        if USE_PRE_DETECT == 0:
            im, gt, h, w = self.pull_item(index)
        else:
            im, gt, h, w = self.pull_item_for_pre_detect(index)

        return im, gt

    def __len__(self):
        return len(self.ids)


    def save_pre_detect_label_file(self, totalLine):
        np.savetxt("list_bbox_celeba_align.txt", totalLine, fmt="%d", delimiter=" ", newline="\n")



    def pull_item_for_pre_detect(self, index):
        img_id = self.ids[index]

        landmark = np.array(self.landmarks[index, :])

        img_path = osp.join(self.root, 'pre_img_align_celeba', 'celebA', img_id)
        # print (img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # print ("image id: " + str(index))
        # print (img_id)

        if self.detector:
            faces = self.detector(img, 1)
            line = np.zeros([1, 5])

            if (len(faces)):
                for i, d in enumerate(faces):
                    self.totalLine[index, 0] = index + 1
                    self.totalLine[index, 1] = d.left()
                    self.totalLine[index, 2] = d.top()
                    self.totalLine[index, 3] = d.right() - d.left()
                    self.totalLine[index, 4] = d.bottom() - d.top()
                    break
            else:
                self.totalLine[index, 0] = index + 1
                
                # self.totalLine = np.vstack((self.totalLine, line))
            # print (self.totalLine)

        return torch.from_numpy(img).permute(2, 0, 1), self.totalLine, height, width

        # return torch.from_numpy(img), target, height, width

    
    def pull_item(self, index):
        img_id = self.ids[index]
        landmark = np.array(self.landmarks[index, :])
        bbox = np.array(self.bbox[index, :])
        landmark = np.reshape(landmark,  (1, 10))
        bbox = np.reshape(bbox, (1, 4))

        # print (index)
        target = np.hstack((landmark, bbox))

        img_path = osp.join(self.root, 'pre_img_align_celeba', 'celebA', img_id)
        # print (img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # print (img.shape)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # img = img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()
            # img = img.transpose(2, 0, 1).long()
            img = img.astype(np.float32)
            target = target.astype(np.float32)

            # print ("img_id:", img_id, ", target:", target)

            target = target / 128
            target[-1, 12] = target[-1, 10] + target[-1, 12]
            target[-1, 13] = target[-1, 11] + target[-1, 13]


        if self.transform is not None:
            
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            print ("transform finished")

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self.root + img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__  == "__main__":

    dataset = CelebaDetection(CELEBA_ROOT)
    data_loader = data.DataLoader(dataset, 1,
                                num_workers = 1,
                                shuffle=True,
                                pin_memory=True)

    batch_iterator = iter(data_loader)

    print (len(dataset.ids))
    for i in range(len(dataset.ids)):
        # print (len(element))
        if (i % 100) == 0:
            print (i)        
        images, targets = next(batch_iterator)

    print (targets.shape)
    targets = np.squeeze(targets)
    print (targets.shape)

    # dataset.save_pre_detect_label_file(targets)

    