import cv2
import numpy as np

import argparse


def add_pts_to_array(params, x, y):
    global rects
    rect = (params["top_left_pt"], params["bottom_right_pt"])
    rects.append(rect)


def update_pts(params, x, y):
    global x_init, y_init
    params["top_left_pt"] = (min(x_init, x), min(y_init, y))
    params["bottom_right_pt"] = (max(x_init, x), max(y_init, y))
    # img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]


def draw_rectangles(event, x, y, flags, params):
    global x_init, y_init, drawing
    # First click initialize the init rectangle point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    # Meanwhile mouse button is pressed, update diagonal rectangle point
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        update_pts(params, x, y)
    # Once mouse botton is release
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        update_pts(params, x, y)
        add_pts_to_array(params, x, y)


def arg_parse():

    parser = argparse.ArgumentParser(description='select save load ROI')
    parser.add_argument("--video", dest='video',
                        help="Video to run selection ROI", default=0)
    parser.add_argument("--width", dest="width",
                        help="Width  of video",            default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    waitTime = 50

    drawing = False
    event_params = {"top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}

    rects = []

    # args.video = "C:\\Develop\\Projects\\opencv\\Pytorch\\select_save_load_ROI\\video\\6919\\2020-02-12-14-34-24.mp4"
    # args.width = 800
    print(args.video)
    print(args.width)

    if args.video == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow('Webcam')
    # Bind draw_rectangles function to every mouse event
    cv2.setMouseCallback('Webcam', draw_rectangles, event_params)

    ################################
    ##
    ret, frame = cap.read()
    dim = (frame.shape[1], frame.shape[0])
    if args.width != 0:
        scale = args.width/frame.shape[1]
        dim = (args.width, int(scale*frame.shape[0]))

    while True:
        ret, frame = cap.read()
        # img = cv2.resize(frame, None, fx=0.5, fy=0.5,
        #                  interpolation=cv2.INTER_AREA)

        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # рисуем текущую область
        if drawing:
            (x0, y0), (x1,
                       y1) = event_params["top_left_pt"], event_params["bottom_right_pt"]
            # img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # отрисовка областей которые запомнили в массив
        for rect_x in rects:
            (x0, y0), (x1, y1) = (
                rect_x[0][0], rect_x[0][1]), (rect_x[1][0], rect_x[1][1])
            # img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow('Webcam', img)
        c = cv2.waitKey(waitTime)

        if c == 27:
            break

        elif c == ord("r"):  # обнуляем массив областей
            rects = []
            event_params = {
                "top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}
            drawing = False

        elif c == ord("s"):  # сохраняем массив областей в файле
            rects_np = np.array(rects, np.int32)
            np.save('setting_roi.bin', rects_np)

        elif c == ord("l"):  # загружаем из файла массив областей
            rects_np = np.load('setting_roi.bin.npy')
            rects = []

            for rect_x in rects_np:
                (x0, y0), (x1, y1) = (
                    rect_x[0][0], rect_x[0][1]), (rect_x[1][0], rect_x[1][1])

                event_params = {
                    "top_left_pt": (x0, y0), "bottom_right_pt": (x1, y1)}

                rect = (event_params["top_left_pt"],
                        event_params["bottom_right_pt"])
                rects.append(rect)

                drawing = False
                event_params = {
                    "top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}

                cv2.setMouseCallback('Webcam', draw_rectangles, event_params)

    cap.release()
    cv2.destroyAllWindows()
