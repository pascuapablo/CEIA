import unittest
import numpy as np
import cv2 as cv


class MyTestCase(unittest.TestCase):
    def test_something(self):

        x, y, w, h = 300, 200, 100, 50
        track_window = (x, y, w, h)

        roi = cv.imread("./resources/estuche.jpg")

        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, 10)

        cap = cv.VideoCapture(0)
        while (1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                ret, track_window = cv.CamShift(dst, track_window, term_crit)

                x, y, w, h = track_window
                img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

                cv.imshow('Seguimiento', img2)

                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break
        cv.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    unittest.main()
