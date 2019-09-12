import cv2
import os.path as osp
import dlib

if __name__  == "__main__":
    detector = dlib.get_frontal_face_detector()

    faces = detector(img, 1)
    print(type(faces[0]), '\n')