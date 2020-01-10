import numpy as np
import cv2

rect = (0, 0, 0, 0)
rectangle = False
rect_over = False
refPt = []


def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(
        image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask


def on_mouse(event, x, y, flags, params):
    # обработка событий мышки, для выделения областей

    global rect, rectangle, rect_over, refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        rect = (x, y, 0, 0)
        rectangle = True

    elif event == cv2.EVENT_LBUTTONUP:
        rect = (rect[0], rect[1], x, y)
        rectangle = False
        rect_over = True
        refPt.append(rect)

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            rect = (rect[0], rect[1], x, y)


# cap = cv2.VideoCapture('video.avi')
cap = cv2.VideoCapture(0)

waitTime = 50

# читаем первый фрейм видеопотока
(grabbed, frame) = cap.read()

# проименуем окно id окна
cv2.namedWindow('frame')
# навесили обработчик на клики мышки на окошкос id='frame'
cv2.setMouseCallback('frame', on_mouse)

while(cap.isOpened()):

    key = cv2.waitKey(waitTime)

    if key == 27:  # ESC
        break

    # сохраняем настройки
    if key == ord("r"):  # обнуляем массив областей
        rectangle = False
        rect_over = False
        refPt = []
        rect = (0, 0, 0, 0)

    elif key == ord("s"):  # сохраняем массив областей в файле
        refPt_np = np.array(refPt, np.int32)
        np.savetxt('setting_roi.txt', refPt_np)

    elif key == ord("l"):  # загружаем из файла массив областей
        refPt_np = np.loadtxt('setting_roi.txt', np.int32)
        refPt = []
        if refPt_np.shape == (4,):
            rect = (refPt_np[0], refPt_np[1], refPt_np[2], refPt_np[3])
            refPt.append(rect)
        else:
            for rect_x in refPt_np:
                rect = (rect_x[0], rect_x[1], rect_x[2], rect_x[3])
                refPt.append(rect)
        rectangle = False
        rect_over = True

    ################################################################################
    ##

    (grabbed, frame) = cap.read()

    if rectangle:
        cv2.rectangle(frame, (rect[0], rect[1]),
                      (rect[2], rect[3]), (255, 0, 255), 2)

    if rect_over:
        for rect_x in refPt:
            cv2.rectangle(
                frame, (rect_x[0], rect_x[1]), (rect_x[2], rect_x[3]), (255, 0, 255), 2)

    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
