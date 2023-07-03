
import os
import sys
import inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
from tqdm import tqdm


def generate_patterns(proj_resolution):
    # generate patterns
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_resolution[0], height=proj_resolution[1])
    _, patterns = graycode.generate()
    black, white = graycode.getImagesForShadowMasks(np.zeros_like(patterns[0]), np.zeros_like(patterns[0]))
    patterns = patterns + [white, black]  # horizontal, vertical, black-white
    return patterns




##-----------------------------------------------------------------------------------------------------



if __name__ == '__main__':


    ## STEP-2: capture sl images
    sl_patterns = generate_patterns(proj_resolution=(1920, 1280))

    for i, x in tqdm(enumerate(sl_patterns)):
        capname = '{}'.format(i)
        cv2.namedWindow(capname, cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow(capname, yml['major_screen_resolution'], 0)
        cv2.setWindowProperty(capname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(capname, x)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


