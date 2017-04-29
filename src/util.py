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


def shrink(frame, ratio=2.0):
    height, width, depth = frame.shape
    return cv2.resize(frame, (int(width / ratio), int(height / ratio)), interpolation=cv2.INTER_AREA)


def preprocess(frame):
    '''
    shrinks, blurs, and grayscales input frame into output frame
    :param frame: frame of video
    :return: output
    '''
    if frame.shape[0] > 720:
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
    result = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=3)
    return result


def get_contours(frame, min_area=minimum_area, debug=False):
    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        for contour in contours:
            print(cv2.contourArea(contour))

    potential_cars = list(filter(lambda contour: cv2.contourArea(contour) > min_area, contours))
    potential_cars = list(map(cv2.convexHull, potential_cars))

    return potential_cars


def combine_frames(*frames):
    '''
    stack frames vertically together 
    useful for debugging various filters' effects 
    :param frames: all the frames to combine, assuming all have same rows and heights dimension
    :return: single numpy array of all frames stacked
    '''
    height = sum(map(lambda x: x.shape[0],frames))
    ratio = height/900 #I chose 900 becuase my monitor has height of 900 pixels
    def toColor(frame):
        if len(frame.shape) != 3:
            return shrink(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), ratio=ratio)
        else:
            return shrink(frame, ratio=ratio)

    return np.vstack(map(toColor, frames))
