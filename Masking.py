import cv2
import numpy as np


class Masking:
    def __init__(self, frame_width, frame_height):
        self.bg_subtractor = None
        self.bg_subtractor_lr = 0
        self.bg_sub_threshold = 50

        self.hand_hist = None
        self.blur_value = 41

        self.top = 0
        self.right = frame_width // 2
        self.bottom = frame_height
        self.left = frame_width

        self.is_hand_hist_created = False
        self.is_bg_captured = False

        self.xs = [7.0 / 20.0, 9.0 / 20.0, 11.0 / 20.0]
        self.ys = [7.0 / 20.0, 9.0 / 20.0, 11.0 / 20.0]

    def init_bg_subtractor(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(0, self.bg_sub_threshold)
        self.is_bg_captured = True

    def bg_sub_masking(self, frame):
        fgmask = self.bg_subtractor.apply(frame, learningRate=self.bg_subtractor_lr)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)  # removes noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)  # closes the holes in the object

        return cv2.bitwise_and(frame, frame, mask=fgmask)

    def get_roi_coord(self):
        return self.top, self.right, self.bottom, self.left

    def draw_rect(self, frame):
        """
        Draw rectangles where
        to get info about color of observable object
        """
        rows, cols, _ = frame.shape

        frame_with_rect = frame.copy()

        for x in self.xs:
            for y in self.ys:
                x0, y0 = int(x * rows), int(y * cols)
                cv2.rectangle(frame_with_rect, (y0, x0), (y0 + 20, x0 + 20), (0, 255, 0), 1)

        return frame_with_rect

    def create_hand_hist(self, frame):
        rows, cols, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([180, 20, 3], dtype=hsv_frame.dtype)

        i = 0
        for x in self.xs:
            for y in self.ys:
                x0, y0 = int(x * rows), int(y * cols)
                roi[i * 20:i * 20 + 20, :, :] = hsv_frame[x0:x0 + 20, y0:y0 + 20, :]

                i += 1

        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.hand_hist = cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
        self.is_hand_hist_created = True

        return self.hand_hist

    def hist_masking(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 60, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=7)
        thresh = cv2.dilate(thresh, kernel, iterations=13)

        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)

    def get_overall_mask(self, frame):
        bg_sub_mask = self.bg_sub_masking(frame)
        hist_mask = self.hist_masking(frame)

        return cv2.bitwise_and(bg_sub_mask, hist_mask)
