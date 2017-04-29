import cv2
import os
import numpy as np
import sys
import cv2
import util
from cv2.bgsegm import createBackgroundSubtractorGMG, createBackgroundSubtractorMOG
import os


ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
RES_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'res'))


if __name__ == '__main__':

    cap = cv2.VideoCapture(os.path.join(RES_DIR, 'output.MP4'))
    # cap = cv2.VideoCapture(os.path.join(RES_DIR, 'CarsDrivingUnderBridge.mp4'))

    bg_subtract = createBackgroundSubtractorMOG()
    # bg_subtract = createBackgroundSubtractorGMG()
    # bg_subtract = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        preprocessed = util.preprocess(frame)
        subtracted = bg_subtract.apply(preprocessed, learningRate=0.001)
        de_noised = util.remove_noise(subtracted)

        contours = util.get_contours(de_noised,min_area=2000)

        de_noised = util.gray_to_color(de_noised)

        cv2.drawContours(de_noised, contours, -1, (0, 255, 0), 3)

        # cv2.imshow('frame', de_noised)

        # cv2.imshow('subtracted', subtracted)

        cv2.imshow('frame', util.combine_frames(preprocessed, de_noised, subtracted))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User pressed quit')
            break

    cap.release()
    cv2.destroyAllWindows()
