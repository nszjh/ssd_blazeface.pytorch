from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

# from  ..layers.functions.detection import Detect


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/blaze_face_with_land_20000_loss_0.000517728622071.pth',
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
    def predict(frame):

        tran_frame = frame.astype(np.float32)
        height, width = tran_frame.shape[:2]
        x = torch.from_numpy(tran_frame).permute(2, 0, 1)

        print (x.shape)
        print (width, height)
  
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data



        print ("detection:", detections.shape)
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * width).cpu().numpy()
                print (pt)
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                for k in range(5):
                    cv2.circle(frame, (pt[4 + k * 2], pt[5 + k * 2]),  2, (0, 0, 255), -1)
                j += 1
        cv2.imshow('frame', frame)
        cv2.waitKey (0)
        cv2.destroyAllWindows()

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera
    # time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    # while True:
        # grab next frame
        # frame = stream.read()
    frame = cv2.imread("/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/celeba_align/pre_img_align_celeba/celebA/" + filename)
    print (type(frame))

    key = cv2.waitKey(1) & 0xFF

    
    # update FPS counter
    # fps.update()
    frame = predict(frame)



if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data.__init__ import BaseTransform as labelmap
    from blazeface import build_blazeface

    test = build_blazeface('test')    # initialize SSD
    test.load_state_dict(torch.load(args.weights))
    # transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    cv2_demo(test.eval(), args.file)
    
    ### export onnx format #####
    net = build_blazeface('train')    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    net.train(False)
    model_onnx_path = "torch_onnx_blazeface.onnx"
    input_shape = (3, 128, 128)

    import torch.onnx as torch_onnx
    dummy_input = Variable(torch.randn(1, *input_shape))
    print (dummy_input.shape)

    output = torch_onnx.export(net, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
