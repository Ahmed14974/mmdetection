from skimage.morphology import thin
import pandas as pd
import numpy as np 
from PIL import Image, ImageDraw
import cv2 as cv 

def dlo():
    # read data. Currently dataset masks but probably want 
    # model output masks eventually
    df = pd.read_csv('/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                    'combined_v3_typed_new_composite/COCO_format_cropped/train.csv')
    df = df[df['contains_shiptrack']]
    image = df.iloc[0]['mask']   # loop over rows
    image = np.load(image)      
    print(np.count_nonzero(image))
    image_proc = np.abs(image.astype(int)).astype(np.uint8)
    # Image.fromarray(image_proc).save("dlo_orig.png")

    # Skeletonize
    def skeleton(image):
        thinned = thin(image, max_iter = 5).astype(np.uint8) #ndarray -> ndarray(bool)
        print(np.count_nonzero(thinned))
        thinned_proc = 255 * thinned
        Image.fromarray(thinned_proc).save("dlo_thin_max5.png")
        print(thinned.shape)
        return thinned 
    thinned = skeleton(image)

    # Find contours
    # https://github.com/opencv/opencv/blob/4.x/samples/python/contours.py
    contours0, hierarchy = cv.findContours(thinned.copy(), mode = cv.RETR_TREE, 
                                            method = cv.CHAIN_APPROX_SIMPLE)
    contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]
    h, w = thinned.shape[:2]
    levels = 0
    vis = np.zeros((h, w, 3), np.uint8)
    cv.drawContours(vis, contours, -1, (128,255,255), 1, cv.LINE_AA, 
                    hierarchy, abs(levels) )
    print(np.count_nonzero(vis))
    Image.fromarray(vis).save("dlo_contour_max5.png")

dlo()
