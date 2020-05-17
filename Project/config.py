import cv2
import numpy as np

# config for center detection with multi-scale pattern matching
# function: setCochleaCenterTemplateMatching
pattern_matching = {
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "canny_thr1": 50,
    "canny_thr2": 200,
    "pat_match_method": cv2.TM_CCOEFF,
    "save_file": False,
    "verbose": False,
    "show_plot": False,
    "image_scaling": np.linspace(0.2, 1.0, 20)[::-1]
}

hough_circles = {
    "method": cv2.HOUGH_GRADIENT,
    "accumulator_value": 0.1,
    "min_dist": 10,
    "canny_thr1": 50,
    "canny_thr2": 200,
    "verbose": False
}

preprocessing = {
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110
}
