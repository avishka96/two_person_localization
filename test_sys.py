# libraries for perspective transform
import os
import cv2
import numpy as np
from helpers import *
import sys
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
import torch

dict_to_json = read_json('output/jsons/inference/EOE_vid1_15fps_kpts.json')

frame_count = len(dict_to_json)
gband = 2
out_fps = 15
outvid_dir = 'output/videos/'
vidfile = 'inference/EOE_vid1_15fps_kpts.mp4'
frame_width = 1920
frame_height = 1080
min_inter_time = 1

inter_end_dict = {}
for frame_key in range(1, frame_count + 1):
    num_person = len(dict_to_json[str(frame_key)])
    # person_ids = list(range(1, num_person+1))
    person_ids = set()
    for person in range(num_person):
        person_ids.add(dict_to_json[str(frame_key)][person]["id"])
    all_two_person_combs = get_person_pairs(sorted(person_ids), 2)
    # print(all_two_person_combs)
    for pair in all_two_person_combs:
        #print(f'pair = {pair}')
        if (pair not in inter_end_dict.keys()) or (frame_key > int(inter_end_dict[pair])):
            p1 = pair[0]
            p2 = pair[1]
            if is_not_misdetection(p1, frame_key, frame_count, dict_to_json, out_fps, min_inter_time) and is_not_misdetection(p2, frame_key, frame_count, dict_to_json, out_fps, min_inter_time):
                for idx in range(num_person):
                    if p1 == int(dict_to_json[str(frame_key)][idx]["id"]):
                        person1_bbox = np.array(dict_to_json[str(frame_key)][idx]["bbox"])
                        p1_idx = idx
                        # print(person1_bbox)
                    elif p2 == int(dict_to_json[str(frame_key)][idx]["id"]):
                        person2_bbox = np.array(dict_to_json[str(frame_key)][idx]["bbox"])
                        p2_idx = idx
                if is_bbox_overlap(person1_bbox, person2_bbox):
                    print(f'{p1} and {p2} intersect at frame {frame_key}')
                    start_frame = get_start_frame(frame_key, gband, out_fps, p1, p2, dict_to_json)
                    print(f'start frame = {start_frame}')
                    end_frame = frame_count
                    overlapping = True
                    pair_video_name = outvid_dir + str(os.path.splitext(vidfile)[0]) + '_P{}P{}.avi'.format(p1,
                                                                                                            p2)
                    pair_outvid = cv2.VideoWriter(pair_video_name,
                                                  cv2.VideoWriter_fourcc(*'MJPG'), out_fps,
                                                  (frame_width, frame_height))
                    for temp_frame in range(start_frame, frame_count + 1):
                        num_person = len(dict_to_json[str(temp_frame)])
                        for idx in range(num_person):
                            if p1 == int(dict_to_json[str(temp_frame)][idx]["id"]):
                                person1_bbox = np.array(dict_to_json[str(temp_frame)][idx]["bbox"])
                                p1_idx = idx
                                # print(person1_bbox)
                            elif p2 == int(dict_to_json[str(temp_frame)][idx]["id"]):
                                person2_bbox = np.array(dict_to_json[str(temp_frame)][idx]["bbox"])
                                p2_idx = idx
                        #print(is_bbox_overlap(person1_bbox, person2_bbox))
                        if (temp_frame > frame_key) and (not is_bbox_overlap(person1_bbox, person2_bbox)) and overlapping:
                            print(f'Not overlapping for {p1},{p2}')
                            overlapping = False
                            end_frame = get_end_frame(temp_frame, frame_count, gband, out_fps, p1, p2, dict_to_json)
                            print(f'end frame = {end_frame}')
                        if temp_frame <= end_frame:
                            white_bg = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
                            plot_skeleton_kpts(white_bg,
                                               torch.tensor(dict_to_json[str(temp_frame)][p1_idx]["skeleton"]), 3,
                                               orig_shape=white_bg.shape[:2])
                            plot_skeleton_kpts(white_bg,
                                               torch.tensor(dict_to_json[str(temp_frame)][p2_idx]["skeleton"]), 3,
                                               orig_shape=white_bg.shape[:2])
                            pair_outvid.write(white_bg)
                            if temp_frame == frame_count:
                                pair_outvid.release()
                                inter_end_dict[pair] = end_frame
                        else:
                            pair_outvid.release()
                            inter_end_dict[pair] = end_frame
                            print(inter_end_dict)
                            break
                else:
                   continue
            else:
                continue
        else:
            continue