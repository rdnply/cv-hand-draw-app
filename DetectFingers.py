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

    # def reorder_defects(self, defects):
    #     defects = self.sort_defects(defects)
    #     defects = defects[defects[0, 0, 3] >= 1000]
    #     s, e, f, _ = defects[0, 0]
    #     first_value = f
    #     fars_by_peak = {e: [f]}
    #     first_key = e
    #     prev_key = e
    #     for i in range(len(defects)):
    #
    #         if i == len(defects) - 1:
    #             fars_by_peak[prev_key] = [f, first_value]
    #             break
    #
    #         s, e, f, d = defects[i, 0]
    #         if d < 1000:
    #             continue
    #
    #         fars_by_peak[prev_key].append(f)
    #         fars_by_peak[e] = [f]
    #         prev_key = e
    #
    #     fars_by_peak.pop(first_key)
    #
    #     return fars_by_peak

    def del_with_less_dist(self, dist, defects):
        res = []
        for i in range(len(defects)):
            _, _, _, d = defects[i, 0]
            if d >= dist:
                res.append(defects[i, 0])

        return np.array(res)

    def reorder_defects(self, defects):
        defects = self.sort_defects(defects)
        defects = self.del_with_less_dist(1000, defects)
        # defects = defects[defects[0, 0, 3] >= 1000]
        if len(defects) == 0:
            return []

        s, e, f, _ = defects[0]
        fars_by_peak = {s: [f], e: [f]}
        first_key = s
        prev_key = e
        for i in range(len(defects)):

            if i == len(defects) - 1:
                s, e, f, _ = defects[i]
                fars_by_peak[first_key].append(f)
                fars_by_peak[prev_key].append(f)
                break

            s, e, f, _ = defects[i]
            fars_by_peak[prev_key].append(f)
            fars_by_peak[e] = [f]
            prev_key = e

        # fars_by_peak.pop(first_key)

        return fars_by_peak

    def sort_defects(self, defects):
        min_key = defects.min()

        new = []
        tail = []
        find_min_key = False
        for i in range(len(defects)):
            s, e, f, _ = defects[i, 0]
            if s != min_key and not find_min_key:
                tail.append(defects[i])
            elif s == min_key:
                new.append(defects[i])
                find_min_key = True
            else:
                new.append(defects[i])

        for t in tail:
            new.append(t)

        return np.array(new)

    def print_all_defects(self, defects, contour):
        for i in range(len(defects)):
            s, e, f, d = defects[i, 0]
            print("Start index: ", s, ",end_index:", e, ",far_index: ", f, ",dist: ", d)
            ss = contour[s, 0]
            ee = contour[e, 0]
            ff = contour[f, 0]
            print("Start point: ", ss, ",end point: ", ee, ",far point: ", ff)

    def count_fingers(self, contour, contour_and_hull):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects1 = cv2.convexityDefects(contour, hull)
            # if type(defects1) != type(None):
            #     defects = self.reorder_defects(defects1)
            #     # self.print_all_defects(defects, contour)
            #     for i in range(len(defects)):
            #         s, e, f, d = defects[i]
            #         if d < 1000:
            #             continue
            #         far = contour[f, 0]
            #         start = contour[s, 0]
            #         end = contour[e, 0]
            #         # cv2.circle(contour_and_hull, tuple(start), 8, [150, 0, 150], -1)
            #         # cv2.circle(contour_and_hull, tuple(end), 8, [150, 0, 150], -1)
            #         cv2.circle(contour_and_hull, tuple(far), 8, [255, 0, 0], -1)
            #
            # return False, 0

            cnt = 0
            if type(defects1) != type(None):
                # self.print_all_defects(defects, contour)
                defects = self.reorder_defects(defects1)

                i = 1
                for key in defects:
                    fars = defects[key]
                    first_far = contour[fars[0], 0]
                    second_far = contour[fars[1], 0]
                    peak = contour[key, 0]

                    print(first_far, second_far, peak)

                    angle = self.calculate_angle(first_far, second_far, peak)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if angle <= math.pi / 3:
                        cnt += 1
                        # cv2.circle(contour_and_hull, tuple(peak), 8, [150, 150, 150], -1)
                        cv2.circle(contour_and_hull, tuple(peak), 8, [150, i * 25, 150], -1)
                        cv2.circle(contour_and_hull, tuple(first_far), 8, [255, 0, 0], -1)
                        cv2.circle(contour_and_hull, tuple(second_far), 8, [255, 0, 0], -1)
                    i += 1
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

            # i = 0
            # for p in hull_points:
            #     cv2.circle(contour_and_hull, p, 4, [255, i * 15, max(255 - i * 15, 0)], i)
            #     i += 1

            for p in hull_points:
                cv2.circle(contour_and_hull, p, 4, [255, 0, 0], 2)

            found, cnt = self.count_fingers(max_contour, contour_and_hull)
            cv2.imshow("Contour and Hull", contour_and_hull)

            if found:
                print(cnt)
