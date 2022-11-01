from skimage.morphology import thin
import pandas as pd
import numpy as np 
from PIL import Image, ImageDraw
import cv2 as cv 
import itertools
from dlo_merge import *

l_s = 10 # 10 pixels is their param. Can improve
max_angle = 0.25 # radians
rect_width = 3 #pixels

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
Given contour, map chains of fixed-len line segments.
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

"""
Get skeleton from white on black image.
"""
def skeleton(image):
    thinned = thin(image, max_iter = 5).astype(np.uint8) #ndarray -> ndarray(bool)
    print(np.count_nonzero(thinned))
    thinned_proc = 255 * thinned
    # Image.fromarray(thinned_proc).save("dlo_thin_max5.png")
    print(thinned.shape)
    return thinned 

def draw_chain_collns(chain_collns, h, w):
    vis = np.zeros((h, w, 3), np.uint8)
    color = 0
    for colln in chain_collns:
        for chain in colln:
            for segment in chain:
                cv.line(vis, segment[0][0], segment[1][0], (color, 255 - color, 255), 1)
            print(np.count_nonzero(vis))
            color += 30
    Image.fromarray(vis).save(f"dlo_segments_pruned.png")

"""
Given midpoints of opposite side of rectangle and fixed width, find corners.
Returned in counterclockwise order.
"""
# https://stackoverflow.com/questions/71582441/draw-a-rectangle-using-points-using-mid-points-of-opposite-sides
def get_corners(point1,point2,width):
    m1 = (point1[1]-point2[1])/(point1[0]-point2[0])
    m2 = -1/m1
    cor_x = np.sqrt((width/2)**2 / (m2**2 + 1))
    cor_y = np.sqrt((width/2)**2 / (m2**-2 + 1))
    if m2 >= 0:
        corner1 = (point1[0] + cor_x, point1[1] + cor_y)
        corner2 = (point1[0] - cor_x, point1[1] - cor_y)
        corner3 = (point2[0] - cor_x, point2[1] - cor_y)
        corner4 = (point2[0] + cor_x, point2[1] + cor_y)
    else:
        corner1 = (point1[0] - cor_x, point1[1] + cor_y)
        corner2 = (point1[0] + cor_x, point1[1] - cor_y)
        corner3 = (point2[0] + cor_x, point2[1] - cor_y)
        corner4 = (point2[0] - cor_x, point2[1] + cor_y)
    return corner1, corner2, corner3, corner4

"""
https://stackoverflow.com/questions/10962379/how-to-check-intersection-between-2-rotated-rectangles
Checks if the two polygons are intersecting.
"""
def do_polygons_intersect(a, b):
    """
 * Helper function to determine whether there is an intersection between the two polygons described
 * by the lists of vertices. Uses the Separating Axis Theorem
 *
 * @param a an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
 * @param b an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
 * @return true if there is any intersection between the 2 polygons, false otherwise
    """
    polygons = [a, b]
    minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None
    for i in range(len(polygons)):
        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        polygon = polygons[i]
        for i1 in range(len(polygon)):
            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]
            # find the line perpendicular to this edge
            normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] }
            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(len(a)):
                projected = normal['x'] * a[j][0] + normal['y'] * a[j][1]
                if (minA is None) or (projected < minA): 
                    minA = projected
                if (maxA is None) or (projected > maxA):
                    maxA = projected
            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(len(b)): 
                projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected
                if (maxB is None) or (projected > maxB):
                    maxB = projected
            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                return False
    return True


"""
Given 2 fixed-len segments, make their respective rectangles
and return whether they overlap.
"""
def overlap_segs(seg1, seg2):
    rect1 = get_corners(seg1[0][0], seg1[1][0], rect_width)
    rect2 = get_corners(seg2[0][0], seg2[1][0], rect_width)
    return do_polygons_intersect(rect1, rect2)

"""
Step E of DLO. Given collection of possibly overlapping DOO chains,
prune overlapping segments.
"""
def prune(chain_colln):
    print(chain_colln)
    num_intersect = 0
    for i in range(len(chain_colln)):
        for j in range(i + 1, len(chain_colln)):
            shorter = None
            longer = None
            if len(chain_colln[i]) < len(chain_colln[j]):
                shorter = i 
                longer = j
            else:
                shorter = j
                longer = i
            keep = [True for _ in range(len(chain_colln[shorter]))]
            for k in range(len(chain_colln[shorter])):
                for seg2 in chain_colln[longer]:
                    if overlap_segs(chain_colln[shorter][k], seg2):
                        num_intersect += 1
                        keep[k] = False
            chain_colln[shorter] = list(itertools.compress(chain_colln[shorter], keep))
    print(f"{num_intersect} intersections")
    print(chain_colln)
    return chain_colln
            


def dlo():
    # read data. Currently dataset masks but probably want 
    # model output masks eventually
    df = pd.read_csv('/deep/group/aicc-bootcamp/cloud-pollution/data/'\
                    'combined_v3_typed_new_composite/COCO_format_cropped/train.csv')
    df = df[df['contains_shiptrack']]
    image = df.iloc[0]['mask']   # loop over rows eventually
    image = np.load(image)      
    print(np.count_nonzero(image))
    image_proc = np.abs(image.astype(int)).astype(np.uint8)
    # Image.fromarray(image_proc).save("dlo_orig.png")

    # Skeletonize
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
    Image.fromarray(vis).save("dlo_contour_max6.png")
    
    #fit and prune doo segments 
    chain_collns = [traverseContour(contour) for contour in contours0]
    # draw_chain_collns(chain_collns, h, w)
    pruned = [prune(chain_colln) for chain_colln in chain_collns]
    draw_chain_collns(pruned, h, w)

    # merge and draw chains
    merged = merge_all_chains(pruned)
    draw_chain(merged)

dlo()
