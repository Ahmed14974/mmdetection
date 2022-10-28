from skimage.morphology import thin
import pandas as pd
import numpy as np 
from PIL import Image, ImageDraw
import cv2 as cv 

l_s = 10 # 10 pixels is their param. Can improve
max_angle = 0.25 # radians

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(s, new_s):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v_s = [s[1][0][0] - s[0][0][0], s[1][0][1] - s[0][0][1]]
    v_new_s = [new_s[1][0][0] - new_s[0][0][0], new_s[1][0][1] - new_s[0][0][1]]
    v_s_u = unit_vector(v_s)
    v_news_u = unit_vector(v_new_s)
    return np.arccos(np.clip(np.dot(v_s_u, v_news_u), -1.0, 1.0))

"""
Implemented DLO paper step D using pseudocode.
"""
def traverseContour(cnt):
    print(cnt.shape) #ndarray shape (num_pts, 1, 2)
    colln = []
    p_s = cnt[0]
    print(p_s.shape)
    chain = []
    p_t = p_s 
    update_tip = True
    s = None
    action_types = [0, 0, 0, 0]

    for p_c in cnt: #p_c [[x y]] format
        dist_cs = np.linalg.norm(p_c - p_s)
        if dist_cs >= l_s:
            p_e = (p_s + l_s * (p_c - p_s) / dist_cs).astype(int)
            new_s = (p_s, p_e) #create segment
            if (chain == []) or (angle_between(s, new_s) < max_angle):
                action_types[0] += 1
                chain.append(new_s)
                s = new_s
                p_s = p_e
                p_t = p_e
            else:
                action_types[1] += 1
                colln.append(chain)
                chain = []
                p_s = p_t 
            update_tip = True
        elif dist_cs >= np.linalg.norm(p_t - p_s):
            action_types[2] += 1
            if update_tip:
                p_t = p_c
        else:
            action_types[3] += 1
            update_tip = False
    colln.append(chain)
    print(action_types)
    return colln

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
        # Image.fromarray(thinned_proc).save("dlo_thin_max5.png")
        print(thinned.shape)
        return thinned 
    thinned = skeleton(image)

    # Find contours
    # https://github.com/opencv/opencv/blob/4.x/samples/python/contours.py
    contours0, hierarchy = cv.findContours(thinned.copy(), mode = cv.RETR_TREE, 
                                            method = cv.CHAIN_APPROX_SIMPLE)
    # contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0] DONT DO THIS
    h, w = thinned.shape[:2]
    levels = 0
    vis = np.zeros((h, w, 3), np.uint8)
    cv.drawContours(vis, contours0, -1, (128,255,255), 1, cv.LINE_AA, 
                    hierarchy, abs(levels) )
    print(np.count_nonzero(vis))
    Image.fromarray(vis).save("dlo_contour_max5.png")
    
    #fit doo segments 
    chain_collns = [traverseContour(contour) for contour in contours0]
    vis = np.zeros((h, w, 3), np.uint8)
    color = 0
    for colln in chain_collns:
        for chain in colln:
            for segment in chain:
                cv.line(vis, segment[0][0], segment[1][0], (color, 255 - color, 255), 1)
            print(np.count_nonzero(vis))
        color += 30
    Image.fromarray(vis).save(f"dlo_segments_all.png")
            

dlo()
