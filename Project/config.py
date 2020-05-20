import cv2
import numpy as np

# config for center detection with multi-scale pattern matching
# function: setCochleaCenterTemplateMatching
general = {
    "data_directory": "./DATA",
    "save_imgs": "./GEN_IMG"
}

pattern_matching_cochlea_center = {
    "pattern_path": "./pattern/round2.png",
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "canny_thr1": 50,
    "canny_thr2": 200,
    "pat_match_method": cv2.TM_CCOEFF,
    "save_file": False,
    "verbose": False,
    "show_plot": False,
    "image_scaling": np.linspace(0.2, 1.0, 20)[::-1],
    "mask_radius": 250
}

# Parameters for the hough approach to the cochlea center detection
# not verry effective with those parameters.
hough_circles_cochlea_center = {
    "method": cv2.HOUGH_GRADIENT,
    "accumulator_value": 0.1,
    "min_dist": 10,
    "canny_thr1": 50,
    "canny_thr2": 200,
    "verbose": False
}

preprocessing_1 = {
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "canny_thr1": 50,
    "canny_thr2": 200
}

cochlea_area = {
    "save_file": False,
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "mask_radius": 250,
    "dilate_kernel": 25,
    "iterations": 5
}
