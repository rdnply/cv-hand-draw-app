import cv2
import numpy as np
import math


class DetectFingers:
    def __init__(self):
        self.max_angle = 60

    def threshold(self, mask):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mask, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def reorder_defects(self, defects):
        s, e, f, _ = defects[0, 0]
        first_value = f
        fars_by_peak = {e: [f]}
        prev_key = e
        for i in range(len(defects)):

            if i == len(defects) - 1:
                fars_by_peak[prev_key] = [f, first_value]
                break

            s, e, f, d = defects[i, 0]
            if d < 1000:
                continue
                
            fars_by_peak[prev_key].append(f)
            fars_by_peak[e] = [f]
            prev_key = e

        return fars_by_peak


    def count_fingers(self, contour, contour_and_hull):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            cnt = 0
            if type(defects) != type(None):
                defects = self.reorder_defects(defects)

                for key in defects:
                    fars = defects[key]
                    first_far = contour[fars[0], 0]
                    second_far = contour[fars[1], 0]
                    peak = contour[key, 0]

                    print(first_far, second_far, peak)

                    angle = self.calculate_angle(first_far, second_far, peak)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if angle <= math.pi / 2:
                        cnt += 1
                        cv2.circle(contour_and_hull, tuple(peak), 8, [150, 150, 150], -1)
                        cv2.circle(contour_and_hull, tuple(first_far), 8, [255, 0, 0], -1)
                        cv2.circle(contour_and_hull, tuple(second_far), 8, [255, 0, 0], -1)
            return True, cnt
        return False, 0

    # def count_fingers(self, contour, contour_and_hull):
    #     hull = cv2.convexHull(contour, returnPoints=False)
    #     if len(hull) > 3:
    #         defects = cv2.convexityDefects(contour, hull)
    #         cnt = 0
    #
    #         # if type(defects) != type(None):
    #         #     ss, ee ,ff ,dd = defects[0, 0]
    #         #     st = tuple(contour[0, 0])
    #         #     en = tuple(contour[len(contour)//6, 0])
    #         #     fa = tuple(contour[len(contour)-1, 0])
    #         #     cv2.circle(contour_and_hull, fa, 8, [255, 0, 0], -1)
    #         #     cv2.circle(contour_and_hull, st, 8, [0, 255, 0], -1)
    #         #     cv2.circle(contour_and_hull, en, 8, [255, 0, 255], -1)
    #
    #         s, e, f, first_d = defects[0, 0]
    #         first_start = tuple(contour[s, 0])
    #         first_end = tuple(contour[e, 0])
    #         first_far = tuple(contour[f, 0])
    #
    #         if type(defects) != type(None):
    #             for i in range(defects.shape[0]):
    #                 s, e, f, d = defects[i, 0]
    #                 start = tuple(contour[s, 0])
    #                 end = tuple(contour[e, 0])
    #                 far = tuple(contour[f, 0])
    #                 angle = self.calculate_angle(start, end, far)
    #
    #                 # Ignore the defects which are small and wide
    #                 # Probably not fingers
    #                 if d > 10000 and angle <= np.pi / 2:
    #                     cnt += 1
    #                     cv2.circle(contour_and_hull, far, 8, [255, 0, 0], -1)
    #         return True, cnt
    #     return False, 0

    def calculate_angle(self, start, end, far):
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        # angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

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

    def clustering(self, hull):
        # max_dist = 50
        max_dist = 30

        clusters = []
        used = np.zeros(len(hull))
        for i in range(len(hull)):
            s = set()

            if used[i]:
                continue

            for j in range(len(hull)):
                if i != j and not used[j] and self.dist(hull[i][0], hull[j][0]) < max_dist:
                    t = (hull[j][0][0], hull[j][0][1])
                    s.add(t)
                    used[j] = 1

            if not used[i]:
                t = (hull[i][0][0], hull[i][0][1])
                s.add(t)
                used[i] = 1

            if len(s) > 0:
                clusters.append(s)

        return clusters

    def clear_hull(self, hull):
        clusters = self.clustering(hull)
        points = []
        for c in clusters:
            centroid = self.get_centroid(c)
            points.append(centroid)

        return points

    def get_centroid(self, cluster):
        cluster = list(cluster)

        centroid = None
        min_y = float('inf')

        for c in cluster:
            if min_y > c[1]:
                min_y = c[1]
                centroid = c

        return centroid

        # sum_by_x = 0
        # sum_by_y = 0
        # for cl in cluster:
        #     x, y = cl
        #     sum_by_x += x
        #     sum_by_y += y
        #
        # sum_by_x /= len(cluster)
        # sum_by_y /= len(cluster)
        # avrg = (sum_by_x, sum_by_y)
        #
        # min_diff = float('inf')
        # min_ind = 0
        #
        # for i in range(len(cluster)):
        #     if min_diff > self.dist(cluster[i], avrg):
        #         min_diff = self.dist(cluster[i], avrg)
        #         min_ind = i

        # for i in range(1, len(cluster)):
        #     if min_diff > self.dist(cluster[i], cluster[min_ind]):
        #         min_diff = self.dist(cluster[i], cluster[min_ind])
        #         min_ind = i

        # return cluster[min_ind]

    def dist(self, first, second):
        return np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)

    def detect_hand(self, frame, mask):
        thresh = self.threshold(mask)
        cv2.imshow("Overall thresh", thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = self.get_max_contour(contours)

            # Draw contour and hull
            contour_and_hull = np.zeros(frame.shape, np.uint8)
            hull = cv2.convexHull(max_contour)
            hull_points = self.clear_hull(hull)
            cv2.drawContours(contour_and_hull, [max_contour], 0, (0, 255, 0), 2)
            cv2.drawContours(contour_and_hull, [hull], 0, (0, 0, 255), 3)

            for p in hull_points:
                cv2.circle(contour_and_hull, p, 4, [255, 0, 0], 2)

            found, cnt = self.count_fingers(max_contour, contour_and_hull)
            cv2.imshow("Contour and Hull", contour_and_hull)

            if found:
                print(cnt)
