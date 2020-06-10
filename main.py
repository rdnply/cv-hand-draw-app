import cv2

from DetectFingers import DetectFingers
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
    detect = DetectFingers()

    top, right, bottom, left = mask.get_roi_coord()

    while capture.isOpened():
        _, frame = capture.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)

        pressed_key = cv2.waitKey(1) & 0xFF

        # get the ROI
        roi = frame[top:bottom, right:left]

        if pressed_key == ord('h'):
            mask.create_hand_hist(roi)
        elif pressed_key == ord('b'):
            mask.init_bg_subtractor()

        if mask.is_bg_captured:
            bg_sub_mask = mask.bg_sub_masking(roi)
            hist_mask = None
            if mask.is_hand_hist_created:
                hist_mask = mask.hist_masking(roi)
                cv2.imshow("hist_mask", hist_mask)
                cv2.imshow("thr_hist_mask", detect.threshold(hist_mask))

            cv2.imshow("bg_mask", bg_sub_mask)
            cv2.imshow("thr_bg_mask", detect.threshold(bg_sub_mask))

            overall_mask = bg_sub_mask
            if hist_mask is not None:
                overall_mask = cv2.bitwise_and(bg_sub_mask, hist_mask)

            cv2.imshow("Mask", overall_mask)
            cv2.imshow("thresh_mask", detect.threshold(overall_mask))
            detect.detect_hand(roi, overall_mask)

        elif not mask.is_hand_hist_created and not mask.is_bg_captured:
            roi = mask.draw_rect(roi)

        cv2.imshow("Camera", rescale_frame(roi))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()
