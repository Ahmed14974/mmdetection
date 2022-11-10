"""
Helper functions for merge step (Step G) of DLO method.
"""
import numpy as np 
import math

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


def merge_two_chains(c1, c2):
    """Merges any two chains based on lowest cost. Implement part H.

    Args:
        c1, c2: Lists of form [s1, s2, ...]
            where s1, s2... are as defined above. 

    Returns:
        ret: List of form [s1, s2, ...] that contains the union of all segments in c1, c2 merged. 
    """
    combos = [[[c1[ai], ai], [c2[bi], bi]] for ai in [0,-1] for bi in [0,-1]]
    #combos = [ [[c1[0], 0], [c2[0], 0]], [[c1[-1], -1], [c2[0], 0]], [[c1[0], 0],
    # [c2[-1], -1]], [[c1[-1], -1], [c2[-1], -1]] ]

    costs = []
    for combo in combos:
        s1, s2 = combo
        weights = [1,1,1]
        costs.append(merge_cost(s1, s2, weights))

    to_connect = combos[np.argmin(costs)] #segments to connect
    index1 = to_connect[0][-1] # head or tail of chain 1
    index2 = to_connect[-1][-1]
    e1 = c1[index1][index1] #if head segment, grab 0th endpoint. else grab -1th
    e2 = c2[index2][index2] 
    lines_intersection = get_intersection(c1[index1][0], c1[index1][-1], 
                                            c2[index2][0], c2[index2][-1])
    dist_1 = np.linalg.norm(e1 - lines_intersection) 
    dist_2 = np.linalg.norm(e2 - lines_intersection)
    t1_ahead = is_ahead(dist_1, dist_2, e2, c2[index2][-1 - index2], lines_intersection)
    t2_ahead = is_ahead(dist_2, dist_1, e1, c1[index1][-1 - index1], lines_intersection)
    new_chain = []
    if t1_ahead and t2_ahead: #scenario 10b 
        print("10b") #turn radius "as desired"
    elif t1_ahead or t2_ahead: #scenario 10a
        print(f"t1_ahead is {t1_ahead}, t2_ahead is {t2_ahead}")
        if t1_ahead: # use circ1. going e1 to e2.
            new_chain = draw_arc_then_line(dist_1, dist_2, lines_intersection, 
                                            e1, e2, c1[index1], c2[index2])        
        else: #going e2 to e1.
            new_chain = draw_arc_then_line(dist_2, dist_1, lines_intersection, 
                                            e2, e1, c2[index2], c1[index1])
    else: #scenario 10c
        print("10c")
    
    ret = []
    # TODO: figure out if need to flip new_chain to fit between them
    if (to_connect[0][-1] == -1) and (to_connect[-1][-1] == 0):
        ret = c1 + c2
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == -1):
        ret = c2 + c1
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == 0):
        ret = c1[::-1] + c2
    elif (to_connect[0][-1] == -1) and (to_connect[-1][-1] == -1):
        ret = c1 + c2[::-1]

    return ret 

def draw_arc_then_line(dist_i, dist_j, lines_intersection, 
                        ei, ej, last_segi, last_segj):
    new_chain = []
    ti = find_t(dist_i, dist_j, lines_intersection, ej)
    circi_center = find_circ_center(ei, last_segi, ti, last_segj)
    circi_r = np.linalg.norm(ei - circi_center) 
    p_s = ei
    candidates = get_circle_intersections(ei[0], ei[1], l_s, 
                                            circi_center[0], 
                                            circi_center[1], circi_r)
    p_e = ei
    if ang((candidates[0], ei), last_segi) > ang((candidates[1], ei), last_segi):
        p_e = candidates[0]
    else:
        p_e = candidates[1]
    new_chain.append((p_s, p_e))
    # go around the circle arc ei to ti
    while np.linalg.norm(ti - p_e) >= l_s:
        printf("drawing around circle")
        prev_s = p_s
        p_s = p_e 
        candidates = get_circle_intersections(p_s[0], p_s[1], l_s, 
                                            circi_center[0], 
                                            circi_center[1], circi_r)
        if np.linalg.norm(candidates[0] - prev_s) < 0.01:
            p_e = candidates[1]
        else:
            p_e = candidates[0]
        new_chain.append((p_s, p_e))
    remaining_dist = np.linalg.norm(ej - p_e)
    del_x = (ej[0] - p_e[0]) * (l_s / remaining_dist)
    del_y = (ej[1] - p_e[1]) * (l_s / remaining_dist)
    while np.linalg.norm(ej - p_e) >= l_s:
        printf("drawing line ti to ej")
        p_s = p_e
        p_e = (p_s[0] + del_x, p_s[1] + del_y)
        new_chain.append((p_s, p_e))
    return new_chain

def merge_all_chains(pruned):
    """Merges list of collection of chains into a single chain.

    Args:
        pruned: [[c1, c2, ...]]
            where c1, c2, ... are as  defined above. 

    Returns:
        to_merge: Merged chain of form [[s1, s2, ...]].
    """
    to_merge = []
    for chain_clcn in pruned:
        for chain in chain_clcn:
            if chain:
                to_merge.append(chain)

        while len(to_merge) > 1:
            c1 = to_merge.pop()
            c2 = to_merge.pop()
            merged = merge_two_chains(c1, c2)
            to_merge.append(merged)

    return to_merge

def draw_chain(chain, h ,w, img_path='dlo_test_imgs/dlo_segments_merged.png', color=0):
    """Generates an image of a chain.

    Args:
        chain: List of form [[s1, s2, ...]] to be drawn. 
        h (int): Height of final img
        w (int): Width of final img
        img_path (str, optional): Save path of generated image. Defaults to 'dlo_test_imgs/dlo_segments_merged.png'.
        color (int, optional): Desired color of drawn line. Defaults to 0.
    """
    vis = np.zeros((h, w, 3), np.uint8)
    for segment in chain:
        cv.line(vis, segment[0][0], segment[1][0], (color, 255 - color, 255), 1)
    Image.fromarray(vis).save(img_path)

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
        return (float('inf'), float('inf'))
    return (x/z, y/z)

""" 
e1 closer to intersection than not e1 iff pointing toward intersection
     t1 ahead of e2 if pointing toward & t1 closer to intersection than e2 or
     pointing away & t1 farther from intersection than e2
"""
def is_ahead(other_dist, seg_dist, seg_end_to_connect, other_seg_end, lines_intersection):
    pointing_to = (seg_dist < np.linalg.norm(other_seg_end - lines_intersection))
    return ((pointing_to and other_dist < seg_dist) or 
            ((not pointing_to) and other_dist > seg_dist))

def find_t(target_dist, other_dist, lines_intersection, other_arrow_end):
    ratio = target_dist / other_dist 
    del_x = (lines_intersection[0] - other_arrow_end[0]) * ratio
    del_y = (lines_intersection[1] - other_arrow_end[1]) * ratio
    return ((lines_intersection[0] - del_x), (lines_intersection[1] - del_y))

def get_circle_intersections(x0, y0, r0, x1, y1, r1):
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
        
        return (x3, y3), (x4, y4)

"""
Using that radius to tangent point is perp to tangent line, we get
2 point-slope form equations. Intersection of lines is center of circle

TODO: Should error check for horiz/vertical
"""
def find_circ_center(tangent_pt1, tangent_line1, tangent_pt2, tangent_line2):
    perp1 = (tangent_line1[0][1] - tangent_line1[1][1] / 
            tangent_line1[0][0] - tangent_line1[1][0])
    perp2 = (tangent_line2[0][1] - tangent_line2[1][1] / 
            tangent_line2[0][0] - tangent_line2[1][0])
    x = (((perp2 * tangent_pt2[0] - perp1 * tangent_pt1[0]) - 
            (tangent_pt2[1] - tangent_pt1[1])) / (perp2 - perp1))
    y = perp1(x - tangent_pt1[0]) + tangent_pt1[1] 
    return (x,y)

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
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
