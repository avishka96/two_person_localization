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
source = 'inference/EOE_vid1_15fps_kpts.mp4'
frame_width = 1920
frame_height = 1080

for frame_key in range(1, frame_count + 1):
    num_person = len(dict_to_json[str(frame_key)])
    # person_ids = list(range(1, num_person+1))
    person_ids = []
    for person in range(num_person):
        person_ids.append(dict_to_json[str(frame_key)][person]["id"])
    all_two_person_combs = get_person_pairs(person_ids, 2)
    print(all_two_person_combs)
    for pair in all_two_person_combs:
        p1 = pair[0]
        p2 = pair[1]
        for idx in range(num_person):
            if p1 == dict_to_json[str(frame_key)][idx]["id"]:
                person1_bbox = np.array(dict_to_json[str(frame_key)][idx]["bbox"])
                # print(person1_bbox)
            elif p2 == dict_to_json[str(frame_key)][idx]["id"]:
                person2_bbox = np.array(dict_to_json[str(frame_key)][pair[1] - 1]["bbox"])
        if is_bbox_overlap(person1_bbox, person2_bbox):
            start_frame = frame_key - gband * out_fps
            if start_frame < 1:
                start_frame = 1
            end_frame = frame_count
            overlapping = True
            pair_video_name = outvid_dir + str(os.path.splitext(source)[0]) + '_P{}P{}.avi'.format(pair[0], pair[1])
            pair_outvid = cv2.VideoWriter(pair_video_name,
                                          cv2.VideoWriter_fourcc(*'MJPG'), out_fps,
                                          (frame_width, frame_height))
            white_bg = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
            # print(torch.tensor(dict_to_json[str(frame_key)][pair[0]-1]["skeleton"]))
            plot_skeleton_kpts(white_bg, torch.tensor(dict_to_json[str(frame_key)][pair[0] - 1]["skeleton"]), 3,
                               orig_shape=white_bg.shape[:2])
            plot_skeleton_kpts(white_bg, torch.tensor(dict_to_json[str(frame_key)][pair[1] - 1]["skeleton"]), 3,
                               orig_shape=white_bg.shape[:2])
            pair_outvid.write(white_bg)
            for temp_frame in range(start_frame, frame_count + 1):
                num_person = len(dict_to_json[str(temp_frame)])
                for idx in range(num_person):
                    if p1 == dict_to_json[str(temp_frame)][idx]["id"]:
                        person1_bbox = np.array(dict_to_json[str(temp_frame)][idx]["bbox"])
                        # print(person1_bbox)
                    elif p2 == dict_to_json[str(temp_frame)][idx]["id"]:
                        person2_bbox = np.array(dict_to_json[str(temp_frame)][pair[1] - 1]["bbox"])
                if (not is_bbox_overlap(person1_bbox, person2_bbox) and overlapping):
                    overlapping = False
                    end_frame = temp_frame + gband * out_fps
                    if end_frame > frame_count:
                        end_frame = frame_count
                if temp_frame < end_frame:
                    white_bg = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
                    plot_skeleton_kpts(white_bg,
                                       torch.tensor(dict_to_json[str(temp_frame)][pair[0] - 1]["skeleton"]), 3,
                                       orig_shape=white_bg.shape[:2])
                    plot_skeleton_kpts(white_bg,
                                       torch.tensor(dict_to_json[str(temp_frame)][pair[1] - 1]["skeleton"]), 3,
                                       orig_shape=white_bg.shape[:2])
                    pair_outvid.write(white_bg)
                else:
                    pair_outvid.release()
                    break