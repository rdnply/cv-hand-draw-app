import cv2
import numpy as np


class Masking:
    def __init__(self, frame_width, frame_height):
        self.bg_substractor = None
        self.bg_subtractor_lr = 0
        self.bg_sub_threshold = 30

        self.hand_hist = None

        self.top = 0
        self.right = frame_width // 2
        self.bottom = frame_height
        self.left = frame_width

        self.is_hand_hist_created = False
        self.is_bg_captured = False

    def init_bg_substractor(self):
        self.bg_substractor = cv2.createBackgroundSubtractorMOG2(10, self.bg_sub_threshold)
        self.is_bg_captured = True

    def bg_sub_masking(self, frame):
        fgmask = self.bg_substractor.apply(frame, learningRate=self.bg_subtractor_lr)

        kernel = np.ones((4, 4), np.uint8)
        # MORPH_OPEN removes noise
        # MORPH_CLOSE closes the holes in the object
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.bitwise_and(frame, frame, mask=fgmask)

    def get_roi_coord(self):
        return self.top, self.right, self.bottom, self.left
