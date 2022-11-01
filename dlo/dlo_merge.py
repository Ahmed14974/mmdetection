"""
Helper functions for merge step (Step G) of DLO method.
"""
from dlo import *


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
    """Merges any two chains based on lowest cost. 

    Args:
        c1, c2: Lists of form [s1, s2, ...]
            where s1, s2... are as defined above. 

    Returns:
        ret: List of form [s1, s2, ...] that contains the union of all segments in c1, c2 merged. 
    """
    combos = [[[c1[ai], ai], [c2[bi], bi]] for ai in [0,-1] for bi in [0,-1]]

    costs = []
    for combo in combos:
        s1, s2 = combo
        weights = [1,1,1]
        costs.append(merge_cost(s1, s2, weights))

    to_connect = combos[np.argmin(costs)]

    # 4 cases
    ret = []
    if (to_connect[0][-1] == -1) and (to_connect[-1][-1] == 0):
        ret = c1 + c2
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == -1):
        ret = c2 + c1
    elif (to_connect[0][-1] == 0) and (to_connect[-1][-1] == 0):
        ret = c1[::-1] + c2
    elif (to_connect[0][-1] == -1) and (to_connect[-1][-1] == -1):
        ret = c1 + c2[::-1]

    return ret

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

def draw_chain(chain, h ,w, img_path='dlo_segments_merged.png', color=0):
    """Generates an image of a chain.

    Args:
        chain: List of form [[s1, s2, ...]] to be drawn. 
        h (int): Height of final img
        w (int): Width of final img
        img_path (str, optional): Save path of generated image. Defaults to 'dlo_segments_merged.png'.
        color (int, optional): Desired color of drawn line. Defaults to 0.
    """
    vis = np.zeros((h, w, 3), np.uint8)
    for segment in chain:
        cv.line(vis, segment[0][0], segment[1][0], (color, 255 - color, 255), 1)
    Image.fromarray(vis).save(img_path)