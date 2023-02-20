import numpy as np
#import cv2
import json
from itertools import combinations

# This code generates the homography transformation matrix to map bird's eye view to a 2D plane
def get_homo_matrix(pts_src=np.empty((4,2)), pts_dst=np.empty((4,2))):
    matrix, status = cv2.findHomography(pts_src, pts_dst)
    return matrix

# Maps standing locations using holomography transformation matrix
def map_std_locs(mat, std_loc=np.empty((1,2))):
    return cv2.perspectiveTransform(std_loc, mat)

def is_bbox_overlap(bbox1=np.empty((1,4)), bbox2=np.empty((1,4))):
    #if  (bbox1[0,0] >= bbox2[0,2]) or (bbox1[0,2] <= bbox2[0,0]) or (bbox1[0,3] <= bbox2[0,1]) or (bbox1[0,1] >= bbox2[0,3]):
    if (bbox1[0] >= bbox2[2]) or (bbox1[2] <= bbox2[0]) or (bbox1[3] <= bbox2[1]) or (
            bbox1[1] >= bbox2[3]):
        return False
    else:
        return True

def read_json(j_file):
    with open(j_file) as f:
        return json.loads(f.read())

def get_person_pairs(arr, r):
    return list(combinations(arr, r))

if __name__ == "__main__":
    arr = list(range(1, 6))
    print(get_person_pairs(arr, 2))
    print(get_person_pairs(arr, 2)[0][1])







