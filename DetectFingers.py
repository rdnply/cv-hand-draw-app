import cv2
import numpy as np
import math


class DetectFingers:
    def __init__(self):
        self.max_angle = 60
        self.default_color = [0, 0, 255]
        self.selected_color = self.default_color
        self.drawing_points = {}
        self.is_delete = False

    def threshold(self, mask):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mask, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

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

    def get_max_y(self, defects, contour):
        mx = 0
        for key in defects:
            peak = contour[key, 0]
            mx = max(mx, peak[1])

        return mx

    def count_fingers(self, contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            fingertips = []
            if defects is not None:
                defects = self.reorder_defects(defects)
                max_y = self.get_max_y(defects, contour)
                for key in defects:
                    fars = defects[key]
                    first_far = contour[fars[0], 0]
                    second_far = contour[fars[1], 0]
                    peak = contour[key, 0]

                    angle = self.calculate_angle(first_far, second_far, peak)
                    if angle <= math.pi / 3 and peak[1] != max_y:
                        fingertips.append(peak)

            return fingertips

        return []

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

    def draw_contour_and_hull(self, frame, contour):
        contour_and_hull = np.zeros(frame.shape, np.uint8)
        hull = cv2.convexHull(contour)
        cv2.drawContours(contour_and_hull, [contour], 0, (0, 255, 0), 2)
        cv2.drawContours(contour_and_hull, [hull], 0, (0, 0, 255), 3)

        hull_points = self.clear_hull(hull)
        for p in hull_points:
            cv2.circle(contour_and_hull, p, 4, [255, 0, 0], 2)

        return contour_and_hull

    def draw_fingertips(self, contour_and_hull, fingertips):
        for f in fingertips:
            cv2.circle(contour_and_hull, tuple(f), 8, [255, 0, 255], -1)

        return contour_and_hull

    def draw(self, contour_and_hull):
        for p in self.drawing_points:
            cv2.circle(contour_and_hull, p, 4, self.drawing_points[p], -1)

    def centroid_of_hand(self, max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    def farthest_point(self, max_contour, centroid):
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
        if defects is not None and centroid is not None:
            s = defects[:, 0][:, 0]
            cx, cy = centroid

            x = np.array(max_contour[s][:, 0][:, 0], dtype=np.float)
            y = np.array(max_contour[s][:, 0][:, 1], dtype=np.float)

            xp = cv2.pow(cv2.subtract(x, cx), 2)
            yp = cv2.pow(cv2.subtract(y, cy), 2)
            dist = cv2.sqrt(cv2.add(xp, yp))

            dist_max_i = np.argmax(dist)

            if dist_max_i < len(s):
                farthest_defect = s[dist_max_i]
                farthest_point = tuple(max_contour[farthest_defect, 0])
                return farthest_point
            else:
                return None

    def detect_hand(self, frame, mask):
        thresh = self.threshold(mask)
        # cv2.imshow("Overall thresh", thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = self.get_max_contour(contours)

            contour_and_hull = self.draw_contour_and_hull(frame, max_contour)
            fingertips = self.count_fingers(max_contour)

            num_fingers = len(fingertips)
            if num_fingers == 1:
                centroid = self.centroid_of_hand(max_contour)
                cv2.circle(contour_and_hull, tuple(centroid), 4, [50, 255, 0], -1)
                far_point = self.farthest_point(max_contour, centroid)
                if far_point is not None:
                    self.drawing_points[far_point] = self.default_color
                    cv2.circle(contour_and_hull, tuple(far_point), 8, [255, 255, 255], -1)

                    # if self.is_delete and far_point in self.drawing_points:
                    #     self.drawing_points.pop(far_point)
                    # else:
                    #     self.drawing_points[far_point] = self.default_color
                    #     cv2.circle(contour_and_hull, tuple(far_point), 8, [255, 255, 255], -1)
                    #     # contour_and_hull = self.draw_fingertips(contour_and_hull, fingertips)

            elif num_fingers == 2:
                self.selected_color = [255, 0, 0]
            elif num_fingers == 3:
                self.selected_color = [0, 255, 0]
            elif num_fingers == 4:
                self.selected_color = [0, 255, 255]
            elif num_fingers == 5:
                self.drawing_points.clear()
                # self.is_delete = not self.is_delete
            elif num_fingers == 0:
                self.default_color = self.selected_color

            self.draw(frame)
            cv2.imshow("Contour and Hull", contour_and_hull)
