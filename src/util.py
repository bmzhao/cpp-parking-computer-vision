import numpy as np
import cv2

small_kernel = np.ones((5, 5), np.uint8)
kernel = np.ones((10, 10), np.uint8)
minimum_area = 1


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def shrink(frame):
    height, width, depth = frame.shape
    return cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)


def preprocess(frame):
    '''
    blurs and grayscales input frame into output frame
    :param frame: frame of video
    :return: output
    '''
    if frame.shape[0] >= 720:
        frame = shrink(frame)

    result = cv2.GaussianBlur(frame, (5, 5), 2)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result

def gray_to_color(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def remove_noise(frame):
    # erosion = cv2.erode(frame, kernel, iterations=1)
    # result = cv2.dilate(frame, kernel, iterations=3)
    # result = cv2.dilate(frame, small_kernel, iterations=1)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, small_kernel)


def get_contours(frame, min_area=minimum_area):
    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_cars = list(filter(lambda contour: cv2.contourArea(contour) > min_area, contours))
    # for car in potential_cars:
    #     print(cv2.contourArea(car))
    return potential_cars
