from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


def initialize_areas():
    global rects, frame_areas

    rects_np = np.load('setting_roi.bin.npy')
    rects = []

    for rect_x in rects_np:
        (x0, y0), (x1, y1) = (rect_x[0][0],
                              rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
        event_params = {"top_left_pt": (x0, y0), "bottom_right_pt": (x1, y1)}
        rect = (event_params["top_left_pt"], event_params["bottom_right_pt"])
        rects.append(rect)


def frames_from_areas(frame_v1, rects):
    frame_array = []

    for rect_x in rects:
        (x0, y0), (x1, y1) = (rect_x[0][0],
                              rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
        frame_array.append(frame_v1[y0:y1, x0:x1])

    return frame_array


def unpdate_areas(frame_v1, orig_im_array, rects):
    # global rects, frame_v1, orig_im

    for rect_x in rects:
        (x0, y0), (x1, y1) = (rect_x[0][0],
                              rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
        orig_im = orig_im_array[rects.index(rect_x)]
        frame_v1[y0:y1, x0:x1] = 255 - orig_im

    # frame_v1[y0:y1, x0:x1] = 255 - orig_im


if __name__ == '__main__':

    #############################
    ##
    initialize_areas()

    ##########################################################################
    ##

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "weights/yolov3.weights"

    # быстрый
    # cfgfile = "cfg/yolov3-tiny.cfg"
    # weightsfile = "weights/yolov3-tiny.weights"

    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'video.avi'

    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame_v1 = cap.read()
        if ret:

            # rect_x = rects[0]
            # (x0, y0), (x1, y1) = (
            #     rect_x[0][0], rect_x[0][1]), (rect_x[1][0], rect_x[1][1])

            # frame = frame_v1.copy()
            # frame = frame_v1[y0:y1, x0:x1]
            frames_array = frames_from_areas(frame_v1, rects)

            # for rect_x in rects:
            #     (x0, y0), (x1, y1) = (
            #         rect_x[0][0], rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
            #     frame_areas.append(frame[y0:y1, x0:x1])

            img_array = orig_im_array = dim_array = []
            for frame in frames_array:
                img, orig_im, dim = prep_image(frame, inp_dim)

                img_array.append(img)
                orig_im_array.append(orig_im)
                dim_array.append(dim)


#            im_dim = torch.FloatTensor(dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(
                output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(
                    frames / (time.time() - start)))
                print("111")

                # frame_v1[y0:y1, x0:x1] = 255 - orig_im[y0:y1, x0:x1]
                unpdate_areas(frame_v1, orig_im_array, rects)
                cv2.imshow("frame", frame_v1)
                # cv2.imshow("frame", orig_im)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(
                output[:, 1:5], 0.0, float(inp_dim))/inp_dim

#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            # frame_v1[y0:y1, x0:x1] = 255 - orig_im[y0:y1, x0:x1]
            unpdate_areas(frame_v1, orig_im_array, rects)
            cv2.imshow("frame", frame_v1)
            # cv2.imshow("frame", orig_im)

            # status = cv2.imwrite('temp1C.jpg', orig_im)
            # print("222")

            ###############################
            ##
            for x in output:
                # objs = [classes[int(x[-1])]
                print("{:25s} ".format(classes[int(x[-1])]))

            print("###############################")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(
                frames / (time.time() - start)))

        else:
            break
