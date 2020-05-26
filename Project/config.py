import cv2
import numpy as np

# config for center detection with multi-scale pattern matching
# function: setCochleaCenterTemplateMatching
general = {
    "Running on MacOS?": False,
    "data_directory": "./DATA",
    "save_imgs": "./GEN_IMG"
}

# Parameters for the function detecting the cochlea center
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

# Used in various preprocessing functions to have a good contrast on the pre-surgery image.
preprocessing_1 = {
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "canny_thr1": 50,
    "canny_thr2": 200
}

cochlea_area = {
    "save_file": True,
    "blur": 65,
    "thr_low_gray": 40,
    "thr_up_gray": 110,
    "mask_radius": 250,
    "dilate_kernel": 25,
    "iterations": 5
}

set_electrode_coordinates = {
    "show_img": True,
    "verbose": True
}

calculate_angular_insertion_depth = {
    "verbose": False
}

electrodes_enumeration = {
    # Check if the next electrode blob is between those two angles
    "min_angle": 5,
    #
    "max_angle": 115,
    # IMPORTANT: SHOW IMAGES OF DETECTED ELECTRODE BEFORE AND AFTER SORTING
    # AND NUMBERING.
    "Show found electrodes on image?": False,
    #
    "Radius threshold": (0.59, 1.5),
    # save image files, Show found electrodes has to be set to true too
    "save_file": False
}
