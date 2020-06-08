import cv2
import numpy as np


class DetectFingers:
    def __init__(self):
        pass

    def threshold(self, mask):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mask, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def count_fingers(self, contour, contour_and_hull):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            cnt = 0
            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s, 0])
                    end = tuple(contour[e, 0])
                    far = tuple(contour[f, 0])
                    angle = self.calculate_angle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > 10000 and angle <= np.pi / 2:
                        cnt += 1
                        cv2.circle(contour_and_hull, far, 8, [255, 0, 0], -1)
            return True, cnt
        return False, 0

    def calculate_angle(self, far, start, end):
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        return angle

    def get_max_contour(self, contours):
        max_index = 0
        max_area = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if area > max_area:
                max_area = area
                max_index = i

        return contours[max_index]

    def detect_hand(self, frame, mask):
        thresh = self.threshold(mask)
        cv2.imshow("Overall thresh", thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = self.get_max_contour(contours)

            # Draw contour and hull
            contour_and_hull = np.zeros(frame.shape, np.uint8)
            hull = cv2.convexHull(max_contour)
            cv2.drawContours(contour_and_hull, [max_contour], 0, (0, 255, 0), 2)
            cv2.drawContours(contour_and_hull, [hull], 0, (0, 0, 255), 3)

            found, cnt = self.count_fingers(max_contour, contour_and_hull)
            cv2.imshow("Contour and Hull", contour_and_hull)

            if found:
                print(cnt)