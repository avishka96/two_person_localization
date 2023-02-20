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
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return False
    else:
        return True

def read_json(j_file):
    with open(j_file) as f:
        return json.loads(f.read())

def get_person_pairs(arr, r):
    return list(combinations(arr, r))

def get_start_frame(frame, band, fps, person1, person2, dict):
    is_person1, is_person2 = False, False
    if ((frame - band*fps) >= 1):
        start = frame - band*fps
    else:
        start = 1
    for f in range(start, frame):
        for i in range(len(dict[str(f)])):
            is_person1 = is_person1 or (int(dict[str(f)][i]["id"]) == person1)
            is_person2 = is_person2 or (int(dict[str(f)][i]["id"]) == person2)
            if is_person1 and is_person2:
                start = f
                break
        if is_person1 and is_person2:
            break
    return start

def get_end_frame(frame, tot_frames, band, fps, person1, person2, dict):
    is_person1, is_person2 = True, True
    if ((frame + band*fps) <= tot_frames):
        end = frame + band*fps
    else:
        end = tot_frames
    for f in range(frame, tot_frames):
        for i in range(len(dict[str(f)])):
            is_person1 = is_person1 and (int(dict[str(f)][i]["id"]) == person1)
            is_person2 = is_person2 and (int(dict[str(f)][i]["id"]) == person2)
            if not (is_person1 and is_person2):
                end = f
                break
        if not (is_person1 and is_person2):
            break
    return end

def is_not_misdetection(person, frame, tot_frames, dict, fps, min_time):
    if (frame + min_time*fps > tot_frames):
        end_limit = tot_frames + 1
    else:
        end_limit = frame + min_time*fps + 1
    for j in range(frame, end_limit):
        res = False
        for index in range(len(dict[str(j)])):
            if person == int(dict[str(j)][index]["id"]):
                res = True
                break
        if not res:
            break
            return res
    return res

if __name__ == "__main__":
    arr = list(range(1, 6))
    print(get_person_pairs(arr, 2))
    print(get_person_pairs(arr, 2)[0][1])







