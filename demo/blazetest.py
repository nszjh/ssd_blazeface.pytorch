from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from thop import profile

# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np
import os.path as osp
# from  ..layers.functions.detection import Detect


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/blaze_face_1_loss_1.511288046836853.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
parser.add_argument('--file', default='000026.jpg', type=str,
                    help='detect image filename')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_mnn_output():
    load_out = np.loadtxt("output.txt").astype(np.float32)
    out = torch.from_numpy(load_out)
    out = out.view(1, 896, 16) 
    return out


def cv2_demo(net, filename = "000026.jpg"):
    def predict(frame23D, frame):

        # tran_frame = frame23D.astype(np.float32)
        height, width = frame23D.shape[:2]
        x = torch.from_numpy(frame23D).permute(2, 0, 1)

        print (x.shape)
        print (width, height)
  
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data



        print ("detection:", detections.shape, detections)
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                print ("get a box")
                pt = (detections[0, i, j, 1:] * width).cpu().numpy()
                print (pt)
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                # for k in range(5):
                #     cv2.circle(frame, (pt[4 + k * 2], pt[5 + k * 2]),  2, (0, 0, 255), -1)
                j += 1
        cv2.imshow('frame', frame)
        cv2.waitKey (0)
        cv2.destroyAllWindows()

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")

    path = osp.join("x:/7174c323-375e-4334-b15e-019bd2c8af08", 'faceDataSet', 'faceData')
    img_path = osp.join(path, filename)
    # depth_path = osp.join(path, filename.split('.')[0] + ".data")
    depth_path = osp.join(path, "ccn-265755" + ".data")
    # print ("path:", img_path, depth_path)
    depth_data  = np.loadtxt(depth_path)
    # print (depth_data.shape, depth_data)
    np_depth_data = np.array(depth_data[:])
    np_depth_data = np.reshape(np_depth_data, (240, 320))
    # print (np_depth_data.shape, np_depth_data)

    # print (img_path)
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    # print (img.shape)
    img = cv2.resize(img, (128, 128))
    np_depth_data = cv2.resize(np_depth_data, (128, 128))
    # print (np_depth_data.shape, np_depth_data)

    temp = img.copy()
    temp = temp.astype(np.float32)
    temp = temp / 255.0

    np_depth_data = cv2.resize(np_depth_data, (128, 128))
    np_depth_data = np_depth_data * 7200
    np_depth_data = np_depth_data.astype(np.float32)
    np_depth_data = np_depth_data / 65535.0

    cv2.imshow("test", np_depth_data)

    # print (img.shape, np_depth_data)
    temp[:, :, 0] = np_depth_data
    temp[:, :, 1] = np_depth_data

    key = cv2.waitKey(1) & 0xFF

    
    # update FPS counter
    # fps.update()
    frame = predict(temp, img)



if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data.__init__ import BaseTransform as labelmap
    from blazeface import build_blazeface

    test = build_blazeface('test')    # initialize SSD

    input = torch.randn(1, 3, 128, 128)
    flops, params = profile(test, inputs=(input, ))
    print (flops, params)

    test.load_state_dict(torch.load(args.weights))
    # transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    cv2_demo(test.eval(), args.file)
    
    ### export onnx format #####
    net = build_blazeface('train')    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    net.train(False)
    model_onnx_path = "torch_onnx_blazeface_simple.onnx"
    input_shape = (3, 128, 128)

    import torch.onnx as torch_onnx
    dummy_input = Variable(torch.randn(1, *input_shape))
    print (dummy_input.shape)

    output = torch_onnx.export(net, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
