"""
NV private  Dataset 
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
# import dlib



# note: if you used our download scripts, this should be right
NVP_ROOT =  "X:/"

USE_PRE_DETECT = 0

class NvpAnnotationTransform(object):
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

    def __init__(self, keep_difficult=False):
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
                if (i % 2) == 0:
                    target[0, i] = int(target[0, i] * 128 / 320)
                else:
                    target[0, i] = int(target[0, i] * 128 / 240)
            i = i + 1

        # print (target)
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class NvpDetection(data.Dataset):
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
                 transform=None, target_transform=NvpAnnotationTransform(),
                 dataset_name='NVP3D'):
        self.root = root
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        list_bbox_nvp_file = osp.join(root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition.txt')

        self.bbox  = np.loadtxt(list_bbox_nvp_file, skiprows=2, usecols=range(1, 5))
        self.ids   = list()
        # for (year, name) in image_sets:
        i = 0
        for line in open(osp.join(root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition.txt')):
            i = i+1
            if i < 3:
                continue
            self.ids.append(line.split(' ')[0])

        print ("ids:" , len(self.ids))

        self.path = osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet', 'faceData')

        # self._imgpath = osp.join('%s', 'pre_img_align_celeba/celebA', '%s.jpg')
        

    def __getitem__(self, index):

        if USE_PRE_DETECT == 0:
            im, gt, h, w = self.pull_item(index)
        else:
            im, gt, h, w = self.pull_item_for_pre_detect(index)

        return im, gt

    def __len__(self):
        return len(self.ids)


   
    def pull_item(self, index):
        img_id = self.ids[index]
        bbox = np.array(self.bbox[index, :])
        bbox = np.reshape(bbox, (1, 4))
        landmark = np.ones([1, 10])

        # print (index)
        target = np.hstack((landmark, bbox))

        img_path = osp.join(self.path, img_id)        
        depth_path = osp.join(self.path, img_id.split('.')[0] + ".npy")
        # print ("path:", img_path, depth_path)
        depth_data  = np.load(depth_path)
        # depth_data  = np.ones([240, 320])
        # depth_data  = np.loadtxt(depth_path)
        # np.save(osp.join(path, img_id.split('.')[0]), depth_data) 
        # # print (depth_data.shape, depth_data)
        np_depth_data = np.array(depth_data[:])
        np_depth_data = np.reshape(np_depth_data, (240, 320))
        # print (np_depth_data.shape, np_depth_data)

        # print (img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # # print (img.shape)
        if self.target_transform is not None:
            ########### 2d image ###########
            # img = img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()
            # img = img.transpose(2, 0, 1).long()
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32)
            img = img / 255.0

            ########### depth data ###########
            np_depth_data = cv2.resize(np_depth_data, (128, 128))
            # print (np_depth_data.shape, np_depth_data)
            np_depth_data = np_depth_data * 7200
            # np_depth_data = np_depth_data.astype(np.float32)
            np_depth_data = np_depth_data / 65535
            # print (np_depth_data)

            ########### target ###########
            target = self.target_transform(target)
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

        # print (img.shape, np_depth_data)
        img[:, :, 0] = np_depth_data
        img[:, :, 1] = np_depth_data

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

    dataset = NvpDetection(NVP_ROOT)
    data_loader = data.DataLoader(dataset, 1,
                                num_workers = 1,
                                shuffle=True,
                                pin_memory=True)

    batch_iterator = iter(data_loader)

    # print (len(dataset.ids))
    for i in range(4890):
        # print (len(element))
        if (i % 100) == 0:
            print (i)        
        images, targets = next(batch_iterator)

    print (targets.shape)
    targets = np.squeeze(targets)
    print (targets.shape)

    # dataset.save_pre_detect_label_file(targets)

    