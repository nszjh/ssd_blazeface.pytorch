import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = '/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/celeba_align/img_align_celeba/'
save_root = '/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/celeba_align/pre_img_align_celeba/'
resize_size = 128

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 10) == 0:
        print('%d images complete' % i)