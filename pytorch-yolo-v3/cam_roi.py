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

# from settings import *

from send_email import send_mail
# import asyncio

import configparser
import json


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
    img = cv2.resize(orig_im, (inp_dim, inp_dim), interpolation=cv2.INTER_AREA)
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

    parser.add_argument("--video", dest='video',
                        help="Video to run selection ROI", default=0)
    parser.add_argument("--width", dest="width",
                        help="Width  of video",             default=0, type=int)
    parser.add_argument("--outputToFile", dest="outputToFile",
                        help="Output video to file ...",    default=0)
    parser.add_argument("--skipFrames", dest="skipFrames",
                        help="Skip frames ",                default=1, type=int)

    return parser.parse_args()


def initialize_areas():
    global rects, frame_area

    rects_np = np.load('setting_roi.bin.npy')
    rects = []
    frame_areas = []

    for rect_x in rects_np:
        (x0, y0), (x1, y1) = (rect_x[0][0],
                              rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
        event_params = {"top_left_pt": (x0, y0), "bottom_right_pt": (x1, y1)}
        rect = (event_params["top_left_pt"], event_params["bottom_right_pt"])
        rects.append(rect)


if __name__ == '__main__':

    #############################
    ##
    initialize_areas()

    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))

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

    # videofile = 'video.avi'
    # args.video = "C:\\Develop\\Projects\\opencv\\Pytorch\\select_save_load_ROI\\video\\3296\\18_00_00-19_00_00.avi"
    # args.width = 800
    # args.outputToFile = "C:\\Develop\\Projects\\opencv\\Pytorch\\select_save_load_ROI\\video\\3296\\output_v2_18_00_00-19_00_00.avi"
    # args.skipFrames = 15

    if args.video == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()

    time_start = time.time()
    people_queue_is_array = []
    people_queue_is_array_v2 = []
    frame_array = []

    for rect_x in rects:
        people_queue_is_array.append(0)
        people_queue_is_array_v2.append(0)
        frame_array.append(0)

    config = configparser.ConfigParser()
    config.read('settings.ini')

    default_config = config['DEFAULT']
    time_that_queue_is = json.loads(default_config['TIME_THAT_QUEUE_IS'])
    number_of_people_that_queue_is = json.loads(
        default_config['NUMBER_OF_PEOPLE_THAT_QUEUE_IS'])
    # skip_frames = json.loads(default_config['SKIP_FRAMES'])

    ################################
    ##
    ret, frame_v2 = cap.read()
    dim_v2 = (frame_v2.shape[1], frame_v2.shape[0])
    if args.width != 0:
        scale = args.width/frame_v2.shape[1]
        dim_v2 = (args.width, int(scale*frame_v2.shape[0]))
        # inp_dim = int(scale*inp_dim)

    curr_frame = 0

    out = cv2.VideoWriter(args.outputToFile, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, dim_v2)

    while cap.isOpened():

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # ret, frame_v1 = cap.read()
        ret, frame_v2 = cap.read()

        if (not ret) and (frame_v2 == None):
            break

        if not ret:
            continue

        frame_v1 = cv2.resize(frame_v2, dim_v2, interpolation=cv2.INTER_AREA)

        curr_frame = curr_frame + 1
        if curr_frame % args.skipFrames != 0:
            continue

        for rect_x in rects:
            (x0, y0), (x1, y1) = (
                rect_x[0][0], rect_x[0][1]), (rect_x[1][0], rect_x[1][1])

            frame = frame_v1[y0:y1, x0:x1]
            img, orig_im, dim = prep_image(frame, inp_dim)

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

                # frame_v1[y0:y1, x0:x1] = 255 - orig_im
                frame_v1[y0:y1, x0:x1] = orig_im
                cv2.rectangle(frame_v1, (x0, y0), (x1, y1), (0, 255, 0), 2)

                continue

            output[:, 1:5] = torch.clamp(
                output[:, 1:5], 0.0, float(inp_dim))/inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            # classes = load_classes('data/coco.names')
            # colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            # frame_v1[y0:y1, x0:x1] = 255 - orig_im
            frame_v1[y0:y1, x0:x1] = orig_im
            cv2.rectangle(frame_v1, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # status = cv2.imwrite('temp1C.jpg', orig_im)
            # print("222")

            ###############################
            ##
            people_queue_is = 0
            for x in output:
                # objs = [classes[int(x[-1])]
                print("{:25s} ".format(classes[int(x[-1])]))

                if classes[int(x[-1])] == 'person':  # 'cup':  # 'person'
                    people_queue_is = people_queue_is + 1

            index = rects.index(rect_x)
            # people_queue_is_array[index] = max(people_queue_is, people_queue_is_array[index]
            people_queue_is_array[index] = people_queue_is
            frame_array[index] = frame_v1.copy()

            print("###############################")

            frames += 1
            print("FPS of the video is {:5.2f}".format(
                frames / (time.time() - start)))

        if (time.time() - time_start) >= time_that_queue_is:
            time_start = time.time()

            for rect_x in rects:
                index = rects.index(rect_x)

                # file_to_attach =
                # NUMBER_OF_PEOPLE_QUEUE_AT_CASH_REGISTER:
                if (people_queue_is_array[index] >= number_of_people_that_queue_is) and (people_queue_is_array_v2[index] >= number_of_people_that_queue_is):
                    # cv2.imshow("frame1", frame_array[index])
                    status = cv2.imwrite(
                        'frame_temp{:n}.png'.format(index), frame_array[index])
                    print("Image written to file-system : ", status)

                    config = configparser.ConfigParser()
                    config.read('settings.ini')

                    default_config = config['DEFAULT']

                    sender_email = json.loads(
                        default_config['SENDER_EMAIL'])
                    sender_name = json.loads(default_config['SENDER_NAME'])
                    password = json.loads(default_config['PASSWORD'])

                    receiver_emails = json.loads(
                        default_config['RECEIVER_EMAILS'])
                    receiver_names = json.loads(
                        default_config['RECEIVER_NAMES'])

                    time_that_queue_is = json.loads(
                        default_config['TIME_THAT_QUEUE_IS'])
                    number_of_people_that_queue_is = json.loads(
                        default_config['NUMBER_OF_PEOPLE_THAT_QUEUE_IS'])

                    # Email body
                    email_html = open('email.html')
                    # email_body = email_html.read()
                    email_body = 'hello'

                    filename = 'frame_temp{:n}.png'.format(index)

                    send_mail(sender_email, sender_name, password, receiver_emails,
                              receiver_names, email_body, filename)

                # запомним предыдущие значения
                people_queue_is_array_v2[index] = people_queue_is_array[index]

            frame_array = []
            # people_queue_is_array = []
            for rect_x in rects:
                # people_queue_is_array.append(0)
                frame_array.append(0)

        if args.outputToFile != 0:
            out.write(frame_v1)
        cv2.imshow("frame", frame_v1)

    cap.release()
    out.release()

    cv2.destroyAllWindows()
