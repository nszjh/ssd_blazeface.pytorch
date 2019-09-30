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
NVP_ROOT =  "/media/nv"

USE_MODE = 3

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
            if i >= 6:
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
        list_gray_bbox_nvp_file = osp.join(root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition_gray.txt')
        list_depth_bbox_nvp_file = osp.join(root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition_depth.txt')

        self.bbox  = np.loadtxt(list_bbox_nvp_file, skiprows=2, usecols=range(1, 5))
        self.gray_bbox = np.loadtxt(list_gray_bbox_nvp_file, skiprows=2, usecols=range(1, 5))
        self.depth_bbox = np.loadtxt(list_depth_bbox_nvp_file, skiprows=2, usecols=range(1, 5))

        self.ids   = self.load_idx('facePosition.txt')
        self.gray_ids  = self.load_idx('facePosition_gray.txt')
        self.depth_ids  = self.load_idx('facePosition_depth.txt')

        print ("ids:" , len(self.ids))
        print ("gray_ids:" , len(self.gray_ids))
        print ("depth_ids:" , len(self.depth_ids))

        self.path = osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet', 'faceData')
        self.path_trans = osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet', 'faceData_trans')
        self.dataset_path = osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet')

        self.gray = []
        self.depth = []
        self.gray_file = 0
        self.depth_file = 0

        # self._imgpath = osp.join('%s', 'pre_img_align_celeba/celebA', '%s.jpg')
        
    def load_idx(self, filename):
        i = 0
        ids = list()
        for line in open(osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  filename)):
            i = i+1
            if i < 3:
                continue
            ids.append(line.split(' ')[0])
        return ids


    def __getitem__(self, index):

        if USE_MODE == 0:
            im, gt, h, w = self.pull_item(index)
        elif USE_MODE == 1:
            im, gt, h, w = self.pull_item_transform(index)
        elif USE_MODE == 2:
            im, gt, h, w = self.pull_item_augmentation(index)
        elif USE_MODE == 3:
            im, gt, h, w = self.pull_item_2(index)
        return im, gt

    def __len__(self):
        return len(self.gray_ids)


    def pull_item_transform(self, index):
        img_id = self.ids[index]
        bbox = np.array(self.bbox[index, :])
        bbox = np.reshape(bbox, (1, 4))
        landmark = np.ones([1, 10])

        # print (index)
        target = np.hstack((landmark, bbox))

        img_path = osp.join(self.path, img_id)        
        depth_path = osp.join(self.path, img_id.split('.')[0] + ".data")
        # print ("path:", img_path, depth_path)
        # depth_data  = np.load(depth_path)
        # depth_data  = np.ones([240, 320])
        depth_data  = np.loadtxt(depth_path)
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
            img = cv2.resize(img, (128, 128))
            
            ########### depth data ###########
            np_depth_data = cv2.resize(np_depth_data, (128, 128))
            # print (np_depth_data.shape, np_depth_data)
            np_depth_data = np_depth_data * 7200
            # np_depth_data = np_depth_data.astype(np.float32)
            mark = np_depth_data > 60000
            np_depth_data[mark] = 0

            np_depth_data = 255 * np_depth_data / 30000
            # np_depth_data = np_depth_data / 65535
            # print (np_depth_data)

            ########### target ###########
            target = self.target_transform(target)
            target = target.astype(np.float32)
            # print ("img_id:", img_id, ", target:", target)
            target = target / 128
            target[-1, 12] = target[-1, 10] + target[-1, 12]
            target[-1, 13] = target[-1, 11] + target[-1, 13]



        train_data  = np.ones([128, 128, 2])

        print (osp.join( self.path_trans, img_id.split('.')[0] + '_gray' + '.bmp'))
        cv2.imwrite(osp.join('faceData_trans', self.path_trans, img_id.split('.')[0] + '_gray' + '.bmp'), img[:, :, 0])
        cv2.imwrite(osp.join('faceData_trans', self.path_trans, img_id.split('.')[0] + '_depth' + '.bmp'), np_depth_data)
        # np.save(osp.join('faceData_trans', depth_path), np_depth_data)

        return torch.from_numpy(train_data).permute(2, 0, 1), target, height, width


    def pull_item_augmentation(self, index):
        img_id = self.ids[index]
        bbox = np.array(self.bbox[index, :])
        bbox = np.reshape(bbox, (1, 4))
        landmark = np.ones([1, 10])

        # print (index)
        target = np.hstack((landmark, bbox))

        img_path = osp.join( self.path_trans, img_id.split('.')[0] + '_gray' + '.bmp')       
        depth_path = osp.join( self.path_trans, img_id.split('.')[0] + '_depth' + '.bmp')
        
        # print (img_path)
        img = cv2.imread(img_path, 0)
        np_depth_data = cv2.imread(depth_path, 0)
        print (img.shape, np_depth_data.shape)
        height, width = img.shape


        img_flip = cv2.flip(img, 1)
        np_depth_data_flip = cv2.flip(np_depth_data, 1)

        cv2.imwrite(osp.join('faceData_trans', self.path_trans, img_id.split('.')[0] + '_gray_flip' + '.bmp'), img_flip)
        cv2.imwrite(osp.join('faceData_trans', self.path_trans, img_id.split('.')[0] + '_depth_flip' + '.bmp'), np_depth_data_flip)


        train_data  = np.ones([128, 128, 2])
        
        img = img.astype(np.float32)
        img = img / 255.0

        np_depth_data = np_depth_data.astype(np.float32)
        np_depth_data = np_depth_data / 255.0


        if self.target_transform is not None:
            ########### target ###########
            target = self.target_transform(target)

        # print ("img:", img)
        # print ("depth", np_depth_data)

        if USE_MODE == 2:
            gray = osp.join('faceData_trans', img_id.split('.')[0] + '_gray' + '.bmp')
            depth = osp.join( 'faceData_trans', img_id.split('.')[0] + '_depth' + '.bmp')
            gray_flip = osp.join( 'faceData_trans', img_id.split('.')[0] + '_gray_flip' + '.bmp')
            depth_flip = osp.join( 'faceData_trans', img_id.split('.')[0] + '_depth_flip' + '.bmp')
            return [gray, depth, gray_flip, depth_flip], target, height, width

        # np.save(osp.join('faceData_trans', depth_path), np_depth_data)

        return torch.from_numpy(train_data).permute(2, 0, 1), target, height, width


    def pull_item_2(self, index):
        gray_id = self.gray_ids[index]
        depth_id = self.depth_ids[index]

        gray_bbox = np.array(self.gray_bbox[index, :])
        gray_bbox = np.reshape(gray_bbox, (1, 4))

        depth_bbox = np.array(self.depth_bbox[index, :])
        depth_bbox = np.reshape(depth_bbox, (1, 4))

        landmark = np.ones([1, 6])

        # print (index)
        # target = np.hstack((landmark, depth_bbox, gray_bbox))
        target = np.hstack((landmark, gray_bbox, depth_bbox))


        img_path = osp.join( self.dataset_path, gray_id)       
        depth_path = osp.join( self.dataset_path, depth_id)
        # # print ("path:", img_path, depth_path)

        # print (img_path)
        # print (depth_path)
        img = cv2.imread(img_path, 0)
        # print (img.shape)
        height, width = img.shape

        np_depth_data = cv2.imread(depth_path, 0)

        # # print (img.shape)
        if self.target_transform is not None:
            ########### 2d image ###########
            # img = img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()
            # img = img.transpose(2, 0, 1).long()
            # img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32)
            img = img / 255.0

            ########### depth data ###########
            np_depth_data = np_depth_data.astype(np.float32)
            # print (np_depth_data, img)
            np_depth_data = np_depth_data / 255.0
            # np_depth_data = np_depth_data / 65535
            # print (np_depth_data)

            ########### target ###########
            # target = self.target_transform(target)
            target = target.astype(np.float32)
            # print ("img_id:", img_id, ", target:", target)
            target = target / 128
            target[-1, 12] = target[-1, 10] + target[-1, 12]
            target[-1, 13] = target[-1, 11] + target[-1, 13]

            target[-1, 8] = target[-1, 6] + target[-1, 8]
            target[-1, 9] = target[-1, 7] + target[-1, 9]

        # print (img.shape, np_depth_data)
        # print (target)

        train_data  = np.ones([128, 128, 2])
        train_data = train_data.astype(np.float32)

        # print (train_data.shape)
        train_data[:, :, 0] = np_depth_data
        train_data[:, :, 1] = img

        return torch.from_numpy(train_data).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width


    def pull_item(self, index):
        img_id = self.ids[index]
        bbox = np.array(self.bbox[index, :])
        bbox = np.reshape(bbox, (1, 4))
        landmark = np.ones([1, 10])

        # print (index)
        target = np.hstack((landmark, bbox))
        print (target)

        img_path = osp.join( self.path, img_id.split('.')[0]  + '.bmp')       
        depth_path = osp.join( self.path, img_id.split('.')[0]  + '.npy')
        # # print ("path:", img_path, depth_path)
        depth_data  = np.load(depth_path)
        # # depth_data  = np.ones([240, 320])
        # # depth_data  = np.loadtxt(depth_path)
        # # np.save(osp.join(path, img_id.split('.')[0]), depth_data) 
        # # # print (depth_data.shape, depth_data)
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
            np_depth_data = np_depth_data.astype(np.float32)
            # print (np_depth_data, img)
            np_depth_data = cv2.resize(np_depth_data, (128, 128))
            # # print (np_depth_data.shape, np_depth_data)
            np_depth_data = np_depth_data * 7200
            # # np_depth_data = np_depth_data.astype(np.float32)
            np_depth_data = np_depth_data / 15000
            # np_depth_data = np_depth_data / 65535
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

        train_data  = np.ones([128, 128, 2])
        train_data = train_data.astype(np.float32)

        # print (train_data.shape)
        train_data[:, :, 0] = img[:, :, 0]
        train_data[:, :, 1] = np_depth_data

        return torch.from_numpy(train_data).permute(2, 0, 1), target, height, width
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



    def start_save_file(self, gray, depth, gray_flip, depth_flip, target):
        if USE_MODE == 2:
            self.gray_file = open(osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition_gray.txt'), 'w')
            self.depth_file = open(osp.join(self.root, '7174c323-375e-4334-b15e-019bd2c8af08', 'faceDataSet',  'facePosition_depth.txt'), 'w')
            self.gray_file.write('total: ' + str(len(gray)))
            self.gray_file.write('\n')
            self.gray_file.write('image_id  x_1  y_1  width  height')
            self.gray_file.write('\n')
            self.depth_file.write('total: ' + str(len(depth)))
            self.depth_file.write('\n')
            self.depth_file.write('image_id  x_1  y_1  width  height')
            self.depth_file.write('\n')

            for line in range(len(gray)):
                self.gray_file.write(gray[line][0] + ' ' + str(int(target[line][10].data)) + ' ' + str(int(target[line][11].data))  + ' ' + str(int(target[line][12].data)) + ' ' + str(int(target[line][13].data)))
                self.gray_file.write('\n')
                self.gray_file.write(gray_flip[line][0] + ' ' + str(128 - int(target[line][10].data) - int(target[line][12].data)) + ' ' + str(int(target[line][11].data))  + ' ' + str(int(target[line][12].data)) + ' ' + str(int(target[line][13].data)))
                self.gray_file.write('\n')
            for line in range(len(depth)):
                self.depth_file.write(depth[line][0] + ' ' + str(int(target[line][10].data))  + ' ' + str(int(target[line][11].data))  + ' ' + str(int(target[line][12].data))  + ' ' + str(int(target[line][13].data)))
                self.depth_file.write('\n')
                self.depth_file.write(depth_flip[line][0] + ' ' + str(128 - int(target[line][10].data) - int(target[line][12].data))  + ' ' + str(int(target[line][11].data))  + ' ' + str(int(target[line][12].data))  + ' ' + str(int(target[line][13].data)))                
                self.depth_file.write('\n')

            self.gray_file.close()
            self.depth_file.close()


if __name__  == "__main__":

    dataset = NvpDetection(NVP_ROOT)
    data_loader = data.DataLoader(dataset, 1,
                                num_workers = 1,
                                shuffle=True,
                                pin_memory=True)


    batch_iterator = iter(data_loader)


    # print (len(dataset.ids))
    depth_data = list()
    gray_data = list()
    depth_data_flip = list()
    gray_data_flip = list()
    targets_data = list()
    for i in range(4890):
        # print (len(element))
        if (i % 100) == 0:
            print (i)        
        datas, targets = next(batch_iterator)

        if USE_MODE == 2:
            gray_data.append(datas[0])
            depth_data.append(datas[1])
            gray_data_flip.append(datas[2])
            depth_data_flip.append(datas[3])
            targets_data.append(targets[0][0])

    print (depth_data)
    print (gray_data)
    print (targets_data)

    if USE_MODE == 2:
        dataset.start_save_file( gray_data, depth_data, gray_data_flip, depth_data_flip,  targets_data )
    


    # dataset.save_pre_detect_label_file(targets)

    