"""
Helper functions for merge step (Step G) of DLO method.
"""
import numpy as np 
import math
import cv2 as cv 
from PIL import Image, ImageDraw
from itertools import combinations
import sys
sys.path.append('../..')
from eval.seg_metrics import calculate_iou 

l_s = 8
COST_THRESHOLD = 50 # TODO: TUNE THIS
ALL_IMGS = False 
DEBUG = False 
"""
Functions with *_cost calculate the cost of merging two segments. 

Args:
    s1, s2 (_type_): [[endpoint1, endpoint2], index_of_segment]
        where
            endpoints: [x, y]
            index_of_segment: 0 or -1, depending on whether the segment was the first or last segment of the chain

"""

def euclidean_cost(s1, s2):
    """Calculates the euclidean cost of merging two segments. 

    Returns:
        int: L2 norm
    """
    return np.linalg.norm(s1[0][s1[-1]] - s2[0][s2[-1]])

def dir_cost(s1, s2):
    """ Calculates the direction cost of merging two segments.

    Returns:
        int: direction cost
    """
    v_s1 = s1[0][1] - s1[0][0]
    v_s2 = s2[0][1] - s2[0][0]
    return float(np.abs(np.arccos(np.inner(-v_s1, v_s2)/(np.linalg.norm(s1[0]) * np.linalg.norm(s2[0])))))

def curv_cost(s1, s2):
    """ Calculates the curvature cost of merging two segments.

    Returns:
        int: curvature cost
    """
    v_s1 = s1[0][1] - s1[0][0]
    v_s2 = s2[0][1] - s2[0][0]
    v_s21 = s2[0][s2[-1]] - s1[0][s1[-1]]
    v_s12 = s1[0][s1[-1]] - s2[0][s2[-1]]

    c1 = float(np.abs(np.arccos(np.inner(v_s1, v_s21)/(np.linalg.norm(s1[0]) * np.linalg.norm(v_s21)))))
    c2 = float(np.abs(np.arccos(np.inner(v_s2, v_s12)/(np.linalg.norm(s2[0]) * np.linalg.norm(v_s12)))))

    return np.maximum(c1, c2)

def merge_cost(s1, s2, weights): 
    """Calculates the summed cost of merging two segments.

    Args:
        weights (list): Weights to be used for linear combination of cost types.

    Returns:
        int: Total cost of merging two segments. 
    """
    w_c, w_d, w_e = weights
    return np.sum([w_c * curv_cost(s1,s2), w_d * dir_cost(s1,s2), w_e * euclidean_cost(s1,s2)])

def merge_two_chains(c1, c2, h, w):
    """Merges two chains based on lowest cost. Implement part H.

    Args:
        c1, c2: Lists of form [s1, s2, ...]
            where s1, s2... are as defined above. 

    Returns:
        ret: if type b or c in paper, return list [c1, c2]
            else return List of form [[s1, s2, ...]] that contains the union of all 
            segments in c1, merging path, and c2 . 
    """
    combos = [[[c1[ai], ai], [c2[bi], bi]] for ai in [0,-1] for bi in [0,-1]]
    #combos = [ [[c1[0], 0], [c2[0], 0]], [[c1[-1], -1], [c2[0], 0]], [[c1[0], 0],
    # [c2[-1], -1]], [[c1[-1], -1], [c2[-1], -1]] ]
    costs = []
    for combo in combos:
        s1, s2 = combo
        weights = [1,1,1]
        costs.append(merge_cost(s1, s2, weights))
    if DEBUG: print("costs is", costs)
    if np.min(costs) > COST_THRESHOLD:
        # TODO: change this to returning two chains separately. Temp returning c1+c2 to keep working code
        return c1 + c2
        # return [c1, c2]

    to_connect = combos[np.argmin(costs)] #segments to connect
    index1 = to_connect[0][-1] # head or tail of chain 1
    index2 = to_connect[-1][-1]
    e1 = c1[index1][index1] #if head segment, grab 0th endpoint. else grab -1th
    e2 = c2[index2][index2] 
#TODO finetune gap size- if small gap, just straight line b/w them
    if np.linalg.norm(e1 - e2) < l_s * 2:
        return concatenate_with_new_chain(to_connect, c1, c2, [(e1, e2)], True)

    lines_intersection = get_intersection(c1[index1][0], c1[index1][-1], 
                                            c2[index2][0], c2[index2][-1])
    t1 = None
    t2 = None
    if np.isinf(lines_intersection[0][0]): #parallel lines
        t1 = project_point(e1, c2[index2])
        t2 = project_point(e2, c1[index1])
    else:
        dist_1 = np.linalg.norm(e1 - lines_intersection) 
        dist_2 = np.linalg.norm(e2 - lines_intersection)
        t1 = find_t(dist_1, dist_2, lines_intersection, e2[0])
        t2 = find_t(dist_2, dist_1, lines_intersection, e1[0])
    t1_ahead = is_ahead(t1, e2, c2[index2][-1 - index2])
    t2_ahead = is_ahead(t2, e1, c1[index1][-1 - index1])
    new_chain = []
    scenario = ""
    e1toe2 = True
    if t1_ahead and t2_ahead: #scenario 10b 
        new_chain = draw_line(e1, e2) #kinda hacky
        scenario = "b"
    elif t1_ahead or t2_ahead: #scenario 10a
        if t1_ahead: # use circ1. going e1 to t1 to e2.
            new_chain = draw_arc_then_line(t1, e1, e2, c1[index1], c2[index2], 
                                            h, w)        
        else: #going e2 to t2 to e1.
            e1toe2 = False
            new_chain = draw_arc_then_line(t2, e2, e1, c2[index2], c1[index1], 
                                            h, w)
        scenario = "a"   
    else:
        new_chain = draw_line(e1, e2) #again kinda hacky
        scenario = "c"
    if DEBUG:
        print(f"scenario {scenario}")
        print((f"chain c1 ends are {c1[:min(len(c1), 2)]} and {c1[max(-1 * len(c1), -2):]}", 
            f"chain c2 ends are {c2[:min(len(c2), 2)]} and {c2[max(-1 * len(c2), -2):]}"))
    return concatenate_with_new_chain(to_connect, c1, c2, new_chain, e1toe2)

def concatenate_with_new_chain(to_connect, c1, c2, new_chain, e1toe2):
    ret = []
    if (to_connect[0][-1] == -1) and (to_connect[-1][-1] == 0):
        if e1toe2:
            ret = c1 + new_chain + c2
        else:
            ret = c1 + new_chain[::-1] + c2
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == -1):
        if e1toe2:
            ret = c2 + new_chain[::-1] + c1
        else:
            ret = c2 + new_chain + c1
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == 0):
        if e1toe2:
            ret = c1[::-1] + new_chain + c2
        else:
            ret = c1[::-1] + new_chain[::-1] + c2
    elif (to_connect[0][-1] == -1) and (to_connect[-1][-1] == -1):
        if e1toe2:
            ret = c1 + new_chain + c2[::-1]
        else:
            ret = c1 + new_chain[::-1] + c2[::-1]
    # return [ret] 
    return ret

def draw_line(ei, ej):
    """
    Args: ei, ej
    Return: chain of segments connecting them
    """
    new_chain = []
    dist = np.linalg.norm(ej - ei)
    del_x = (ej[0][0] - ei[0][0]) * (l_s / dist) #TODO: zero division
    del_y = (ej[0][1] - ei[0][1]) * (l_s / dist)
    p_s = ei
    p_e = p_s
    counter = 0
    while np.linalg.norm(ej - p_e) >= l_s:
        if counter == 10:
            break
        p_s = p_e
        p_e = p_s + np.array([[del_x, del_y]])
        p_s = p_s.astype(int)
        p_e = p_e.astype(int)
        new_chain.append((p_s, p_e))
        counter += 1
    return new_chain

def draw_arc_then_line(ti, ei, ej, last_segi, last_segj, h, w):
    """
    Implement scenario a. Draw around circle ei to ti, then
    straight line ti to ej, with fixed len segments.
    Args:
        ei, ej, ti [[x y]]
        last_segi, last_segj (array([[x0 y0]]), array([[x1, y1]]))
    """
    new_chain = []
    circi_center = find_circ_center(ei, last_segi, ti, last_segj)
    if DEBUG: 
        print("ei, ej, last_segi, ti, last_segj", ei, ej, last_segi, ti, last_segj)
    circi_r = np.linalg.norm(ei - circi_center) 
    if DEBUG: 
        print(f"center and radius are {circi_center}, {circi_r}")
    p_s = ei
    p_e = ei
    if circi_r >= l_s / 2:
        candidates = get_circle_intersections(ei[0][0], ei[0][1], l_s, 
                                                circi_center[0][0], 
                                                circi_center[0][1], circi_r)
        if DEBUG: 
            print(f"candidates are {candidates}")
        if ang((candidates[0], ei), last_segi) > ang((candidates[1], ei), last_segi):
            p_e = candidates[0]
        else:
            p_e = candidates[1]
        p_s = p_s.astype(int)
        p_e = p_e.astype(int)
        if out_of_bounds(p_e, h, w):
            return []
        new_chain.append((p_s, p_e))
        if DEBUG:
            print("new chain append", p_s, p_e)
        ctr = 0
        # go around the circle arc ei to ti
        while np.linalg.norm(ti - p_e) >= l_s:
            if ctr == 10: #hopefully not that big of a gap. Else buggy
                break
            if DEBUG:
                print("drawing around circle")
            prev_s = p_s
            p_s = p_e 
            candidates = get_circle_intersections(p_s[0][0], p_s[0][1], l_s, 
                                                circi_center[0][0], 
                                                circi_center[0][1], circi_r)
            if np.linalg.norm(candidates[0] - prev_s) < 2: # can finetune threshold
                p_e = candidates[1]
            else:
                p_e = candidates[0]
            p_s = p_s.astype(int)
            p_e = p_e.astype(int)
            if out_of_bounds(p_e, h, w):
                return []
            new_chain.append((p_s, p_e))
            ctr += 1
    else:
        new_chain = draw_line(ei, ti)
        p_e = ti
    remaining_dist = np.linalg.norm(ej - p_e)
    del_x = (ej[0][0] - p_e[0][0]) * (l_s / remaining_dist)
    del_y = (ej[0][1] - p_e[0][1]) * (l_s / remaining_dist)
    ctr = 0
    while np.linalg.norm(ej - p_e) >= l_s: #TODO: refactor into draw_line fxn??
        if ctr == 10: #hopefully not connecting such a big gap. Else buggy
            break
        p_s = p_e
        p_e = p_s + np.array([[del_x, del_y]])
        p_s = p_s.astype(int)
        p_e = p_e.astype(int)
        if out_of_bounds(p_e, h, w):
            return []
        new_chain.append((p_s, p_e))
        ctr += 1
    new_chain.append((p_e, ej))
    return new_chain

def out_of_bounds(point, h, w):
    return (point[0][0] < 0 or point[0][0] > w or point[0][1] < 0 or point[0][1] > h)

def find_pair_cost(pair):
    """
    For a given pair of chains (c1, c2) find their min merge cost,
    trying all end pairs.
    """
    c1 = pair[0]
    c2 = pair[1]
    combos = [[[c1[ai], ai], [c2[bi], bi]] for ai in [0,-1] for bi in [0,-1]]
    #combos = [ [[c1[0], 0], [c2[0], 0]], [[c1[-1], -1], [c2[0], 0]], [[c1[0], 0],
    # [c2[-1], -1]], [[c1[-1], -1], [c2[-1], -1]] ]
    costs = []
    for combo in combos:
        s1, s2 = combo
        weights = [1,1,1]
        costs.append(merge_cost(s1, s2, weights))
    return min(costs)
    
#TODO: if Lyna has implemented this, switch code out
def merge_all_chains(pruned, h, w):
    """Merges list of collection of chains.
    Every step, find + try to merge pair of chains with lowest cost.
    If cannot merge (cost > threshold), stop merging and return.
    Remaining separate chains should be separate instances (?)

    Args:
        pruned: [[c1, c2, ...]]
            where c1, c2, ... are as  defined above. 

    Returns:
        to_merge: Merged chain of form [[s1, s2, ...]].
    """
    to_merge = []
    tried = set()
    chain_pair_to_cost = {}
    for chain_clcn in pruned:
        for chain in chain_clcn:
            if chain:
                to_merge.append(chain)

    #     while True:
    #         combs = combinations(to_merge, 2)
    #         for pair in list(combs):
    #             print(type(pair)) #tuple but contains lists which is a problem
    #             chain_pair_to_cost[pair] = find_pair_cost(pair) 
    #             #TODO: get hashable key instead. maybe ((c1_end_x, c1_end_y), (c2_end_x, c2_end_y))? 
    #         pair_to_merge = min(chain_pair_to_cost, key=chain_pair_to_cost.get)
    #         if pair_to_merge in tried:
    #             break
    #         merged = merge_two_chains(pair_to_merge[0], pair_to_merge[1])
    #         if len(merged) > 1:
    #             tried.add(pair_to_merge)
    #         to_append += merged
    # return to_merge 

        while len(to_merge) > 1:
            c1 = to_merge.pop()
            c2 = to_merge.pop()
            merged = merge_two_chains(c1, c2, h, w)
            to_merge.append(merged)

    return to_merge

def draw_chain(chain, h ,w, img_path='dlo_test_imgs/dlo_segments_merged.png', color=255, width=10):
    """Generates an image of a chain.

    Args:
        chain: List of form [s1, s2, ...] to be drawn. 
        h (int): Height of final img
        w (int): Width of final img
        img_path (str, optional): Save path of generated image. Defaults to 'dlo_test_imgs/dlo_segments_merged.png'.
        color (int, optional): Desired grayscale color of drawn line. Defaults to white=255
        width (optional): desired line width. default 10 pixels

    Return:
        np array.
    """

    img = np.zeros((h, w), np.uint8)
    for segment in chain:
        cv.line(img, segment[0][0], segment[1][0], color, thickness=10)
    if ALL_IMGS:
        Image.fromarray(img).save(img_path)
    return img

def get_intersection(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return np.array([[float('inf'), float('inf')]])
    return np.array([[x/z, y/z]])

""" 
e1 closer to intersection than not e1 iff pointing toward intersection
     t1 ahead of e2 if pointing toward & t1 closer to intersection than e2 or
     pointing away & t1 farther from intersection than e2
"""
def is_ahead(ti, ej, tail_j):
    """
    Args: ti, ej, tail_j [[x y]] 
    """
    if ((ej[0][0] < ti[0][0] and ti[0][0] <= tail_j[0][0]) or 
        (ej[0][0] > ti[0][0] and ti[0][0] >= tail_j[0][0])): # ti in between ej, tail_j
        return False
    dist_ti_ej = np.linalg.norm(ej - ti)
    dist_ti_tailj = np.linalg.norm(tail_j - ti)
    return dist_ti_ej < dist_ti_tailj

def find_t(target_dist, other_dist, lines_intersection, other_arrow_end):
    ratio = target_dist / other_dist #TODO zero division??
    if DEBUG:
        print("find_t", target_dist, other_dist, lines_intersection, other_arrow_end)
    del_x = (lines_intersection[0][0] - other_arrow_end[0]) * ratio
    del_y = (lines_intersection[0][1] - other_arrow_end[1]) * ratio
    return np.array([[(lines_intersection[0][0] - del_x), (lines_intersection[0][1] - del_y)]])

def project_point(point, line):
    p1 = line[0]
    p2 = line[1]
    l2 = np.sum((p1-p2)**2)
    if l2 == 0:
        print('p1 and p2 are the same points')
    #The line extending the segment is parameterized as p1 + t (p2 - p1).
    #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    #if you need the point to project on line extention connecting p1 and p2
    t = np.sum((point - p1) * (p2 - p1)) / l2
    projection = p1 + t * (p2 - p1)
    return projection

def get_circle_intersections(x0, y0, r0, x1, y1, r1):
    "Args: x0, y0, r0 int; x1, y1, r1 float"
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return np.array([[x3, y3]]), np.array([[x4, y4]])

"""
Using that radius to tangent point is perp to tangent line, we get
2 point-slope form equations. Intersection of lines is center of circle
"""
def find_circ_center(tangent_pt1, tangent_line1, tangent_pt2, tangent_line2):
    """
    Return:
        center of tangent circle: np.array(x,y)
    Args: 
        tangent_pt1: [[x, y]]
        tangent_line1: (array([[x1, y1]]), array([[x2, y2]]))
        tangent_pt2: [[x, y]]
        tangent_line2: (array([[x1, y1]]), array([[x2, y2]]))
    """
    if DEBUG:
        print("find_circ_center", tangent_pt1, tangent_line1, tangent_pt2, tangent_line2)
    perp_line1 = None
    perp_line2 = None
    if (tangent_line1[0][0][0] == tangent_line1[1][0][0]): # x's equal so vertical
        perp_line1 = (tangent_pt1, tangent_pt1 + [1, 0])
    elif (tangent_line1[0][0][1] == tangent_line1[1][0][1]): # y's equal so horiz
        perp_line1 = (tangent_pt1, tangent_pt1 + [0, 1])
    else: # diagonal tangent_line1
        perp1 = -1 *((tangent_line1[0][0][0] - tangent_line1[1][0][0]) / # -1 / ((y1 - y2) / (x1 - x2)) = -(x1-x2) / (y1-y2)
            (tangent_line1[0][0][1] - tangent_line1[1][0][1]))
        perp_line1 = (tangent_pt1, tangent_pt1 + [1, perp1])

    if (tangent_line2[0][0][0] == tangent_line2[1][0][0]): # x's equal so vertical
        perp_line2 = (tangent_pt2, tangent_pt2 + [1, 0])
    elif (tangent_line2[0][0][1] == tangent_line2[1][0][1]): # y's equal so horiz
        perp_line2 = (tangent_pt2, tangent_pt2 + [0, 1])
    else: #diagonal tangent-line2
        perp2 = -1 * ((tangent_line2[0][0][0] - tangent_line2[1][0][0]) / 
            (tangent_line2[0][0][1] - tangent_line2[1][0][1]))
        perp_line2 = (tangent_pt2, tangent_pt2 + [1, perp2])
    intersection = get_intersection(perp_line1[0], perp_line1[1], perp_line2[0], perp_line2[1])
    return intersection
    
def ang(lineA, lineB):
    """
    Args:
        lineA: (array([[x1, y1]]), array([[x2, y2]]))
        lineB:(array([[x1, y1]]), array([[x2, y2]]))
    Return:
        Angle between lineA and lineB, less than 180 degrees.
    """
    if DEBUG:
        print("ang", lineA, lineB)
    # Get nicer vector form
    vA = [(lineA[0][0][0]-lineA[1][0][0]), (lineA[0][0][1]-lineA[1][0][1])]
    vB = [(lineB[0][0][0]-lineB[1][0][0]), (lineB[0][0][1]-lineB[1][0][1])]
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    # Get magnitudes
    magA = np.dot(vA, vA)**0.5
    magB = np.dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg
