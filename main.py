import cv2
import numpy as np

from Masking import Masking


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():
    capture = cv2.VideoCapture(0)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask = Masking(frame_width, frame_height)

    top, right, bottom, left = mask.get_roi_coord()

    while capture.isOpened():
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        pressed_key = cv2.waitKey(1) & 0xFF

        # get the ROI
        roi = frame[top:bottom, right:left]

        roi = cv2.bilateralFilter(roi, 5, 50, 100)

        if pressed_key == ord('h'):
            mask.create_hand_hist(roi)
        elif pressed_key == ord('b'):
            mask.init_bg_substractor()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if mask.is_hand_hist_created and mask.is_bg_captured:
            bg_sub_mask = mask.bg_sub_masking(roi)
            hist_mask = mask.hist_masking(roi)
            cv2.imshow("hist_mask", hist_mask)
            cv2.imshow("bg_mask", bg_sub_mask)

            cv2.imshow("thr_hist_mask", mask.threshold(hist_mask))
            cv2.imshow("thr_bg_mask", mask.threshold(bg_sub_mask))

            m = cv2.bitwise_and(bg_sub_mask, hist_mask)
            cv2.imshow("Mask", m)

            cv2.imshow("thresh_mask", mask.threshold(bg_sub_mask))
        elif not mask.is_hand_hist_created:
            roi = mask.draw_rect(roi)

        cv2.imshow("Camera", rescale_frame(roi))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()
