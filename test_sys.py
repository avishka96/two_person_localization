# libraries for perspective transform
import os
import cv2
import numpy as np
from helpers import *
import sys
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
import torch

# ----------------------------------------------------------------------------------------------------------------------

dict_to_json = read_json('output/jsons/inference/UT1/seq5_15fps_kpts.json')

frame_count = len(dict_to_json)
gband = 1
out_fps = 15
outvid_dir = 'output/videos/'
outjson_dir = 'output/jsons/'
jsonfile = 'output/jsons/inference/UT1/seq5_15fps/1_P1P3.json'
vidfile = 'inference/UT1/seq5_15fps.avi'
frame_width = 1920
frame_height = 1080
min_inter_time = 1
# Blue, Green, Red, Blue-violet, Orange
names = ["Kick", "Push", "Handshake", "Hi-five", "Hug"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (226, 43, 136), (3, 97, 255)]

# System code starts here
inter_end_dict = {}
temp_interbb_dict = {}
pair_count = 1

# ----------------------------------------------------------------------------------------------------------------------
def visualize(iden, categ, confid, video_file=vidfile, json_file=jsonfile):
    # read json to dict
    int_dict = read_json(json_file)
    frameNum = 0

    # reading the input
    cap = cv2.VideoCapture(str(video_file))
    vis_vidfile = outvid_dir + str(os.path.splitext(vidfile)[0]) + '_visualizer.avi'

    # dir to save frames
    frames_dir = os.path.splitext(vis_vidfile)[0]
    try:
        os.makedirs(frames_dir)
    except FileExistsError:
        pass

    output = cv2.VideoWriter(
        vis_vidfile, cv2.VideoWriter_fourcc(*'MJPG'), out_fps, (frame_width, frame_height))

    while (True):
        ret, frame = cap.read()
        if (ret):

            # adding rectangle on each frame
            # cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 3)
            if str(frameNum) in int_dict.keys():
                draw_bboxes(frame, int_dict[str(frameNum)], iden, categ, confid)

            # writing the new frame in output
            output.write(frame)
            cv2.imwrite(frames_dir + '/frame_' + f'{frameNum}.jpg', frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            frameNum += 1

    cv2.destroyAllWindows()
    output.release()
    cap.release()

# Draw bboxes for interactions
def draw_bboxes(img, bbox, identity=None, category=None, confidence=None, names=names, colors=colors, nobbox=False, hide_labels=False):
    x1, y1, x2, y2 = bbox
    print(x1, y1, x2, y2)
    # tl = opt.line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tl = 3

    cat = int(category) if category is not None else 0
    id = int(identity) if identity is not None else 0

    color = colors[cat]

    if not nobbox:
        #print(img)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, tl)

    if not hide_labels:
        label = str(id) + ":" + names[cat] if identity is not None else f'{names[cat]} {confidence:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_x, c2_y = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (int(x1), int(y1)), (int(c2_x), int(c2_y)), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (int(x1), int(y1) - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    # if kpt_label:
    #     plot_skeleton_kpts(img, kpts, steps, orig_shape=orig_shape)

    return img

# ----------------------------------------------------------------------------------------------------------------------

# Create vid directory
try:
    os.makedirs(outvid_dir + f'{str(os.path.splitext(vidfile)[0])}/')
except FileExistsError:
    pass

# Create json directory
try:
    os.makedirs(outjson_dir + f'{str(os.path.splitext(vidfile)[0])}/')
except FileExistsError:
    pass

for frame_key in range(1, frame_count + 1):
    num_person = len(dict_to_json[str(frame_key)])
    # person_ids = list(range(1, num_person+1))
    person_ids = set()
    for person in range(num_person):
        person_ids.add(dict_to_json[str(frame_key)][person]["id"])
        all_two_person_combs = get_person_pairs(sorted(person_ids), 2)
        # print(all_two_person_combs)
        for pair in all_two_person_combs:
            # print(f'pair = {pair}')
            if (pair not in inter_end_dict.keys()) or (frame_key > int(inter_end_dict[pair])):
                p1 = pair[0]
                p2 = pair[1]
                if is_not_misdetection(p1, frame_key, frame_count, dict_to_json, out_fps,
                                           min_inter_time) and is_not_misdetection(p2, frame_key, frame_count,
                                                                                   dict_to_json, out_fps,
                                                                                   min_inter_time):
                    p1_idx, p2_idx, person1_bbox, person2_bbox = num_person, num_person, np.zeros(4), np.zeros(4)
                    try:
                        for idx in range(num_person):
                            if p1 == int(dict_to_json[str(frame_key)][idx]["id"]):
                                person1_bbox = np.array(dict_to_json[str(frame_key)][idx]["bbox"])
                                p1_idx = idx
                                # print(person1_bbox)
                            elif p2 == int(dict_to_json[str(frame_key)][idx]["id"]):
                                person2_bbox = np.array(dict_to_json[str(frame_key)][idx]["bbox"])
                                p2_idx = idx
                    except IndexError:
                        continue
                    if is_bbox_overlap(person1_bbox, person2_bbox):
                        print(f'{p1} and {p2} intersect at frame {frame_key}')
                        start_frame = get_start_frame(frame_key, gband, out_fps, p1, p2, dict_to_json)
                        print(f'frame = {frame_key}, start frame = {start_frame}')
                        end_frame = frame_count
                        overlapping = True
                        pair_video_name = outvid_dir + f'{str(os.path.splitext(vidfile)[0])}/' + f'{pair_count}_P{p1}P{p2}.avi'
                        pair_json_name = outjson_dir + f'{str(os.path.splitext(vidfile)[0])}/' + f'{pair_count}_P{p1}P{p2}.json'
                        pair_count += 1
                        #interdims_dict[inter_num] = []
                        pair_outvid = cv2.VideoWriter(pair_video_name,
                                                          cv2.VideoWriter_fourcc(*'MJPG'), out_fps,
                                                          (frame_width, frame_height))
                        for temp_frame in range(start_frame, frame_count + 1):
                            num_person = len(dict_to_json[str(temp_frame)])
                            p1_idx, p2_idx, person1_bbox, person2_bbox = num_person, num_person, np.zeros(
                                    4), np.zeros(4)
                            for idx in range(num_person):
                                if p1 == int(dict_to_json[str(temp_frame)][idx]["id"]):
                                    person1_bbox = np.array(dict_to_json[str(temp_frame)][idx]["bbox"])
                                    p1_idx = idx
                                    # print(person1_bbox)
                                if p2 == int(dict_to_json[str(temp_frame)][idx]["id"]):
                                    person2_bbox = np.array(dict_to_json[str(temp_frame)][idx]["bbox"])
                                    p2_idx = idx

                            # print(is_bbox_overlap(person1_bbox, person2_bbox))
                            if (temp_frame > frame_key) and (
                                not is_bbox_overlap(person1_bbox, person2_bbox)) and overlapping:
                                print(f'Not overlapping for {p1},{p2}')
                                overlapping = False
                                end_frame = get_end_frame(temp_frame, frame_count, gband, out_fps, p1, p2,
                                                              dict_to_json)
                                print(f'temp frame = {temp_frame}, end frame = {end_frame}')
                            if temp_frame <= end_frame:
                                # print(f'processing frame = {temp_frame}')
                                white_bg = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
                                if p1_idx < num_person:
                                    plot_skeleton_kpts(white_bg,
                                                           torch.tensor(
                                                               dict_to_json[str(temp_frame)][p1_idx]["skeleton"]), 3,
                                                           orig_shape=white_bg.shape[:2])
                                    outbbx1, outbby1, outbbx2, outbby2 = person1_bbox
                                    
                                if p2_idx < num_person:
                                    plot_skeleton_kpts(white_bg,
                                                           torch.tensor(
                                                               dict_to_json[str(temp_frame)][p2_idx]["skeleton"]), 3,
                                                           orig_shape=white_bg.shape[:2])
                                    temp_outbbx1, temp_outbby1, temp_outbbx2, temp_outbby2 = person2_bbox
                                    outbbx1, outbby1, outbbx2, outbby2 = min(outbbx1, temp_outbbx1), min(outbby1, temp_outbby1), max(outbbx2, temp_outbbx2), max(outbby2, temp_outbby2)
                                    temp_interbb_dict[temp_frame] = [outbbx1, outbby1, outbbx2, outbby2]
                                    #print(temp_interbb_dict)
                                pair_outvid.write(white_bg)
                                if temp_frame == frame_count:
                                    pair_outvid.release()
                                    inter_end_dict[pair] = end_frame
                                    # interdims_dict[inter_num] = [temp_interbb_dict]
                                    # print(interdims_dict)
                                    temp_json_object = json.dumps(temp_interbb_dict, indent=4)
                                    with open(pair_json_name, "w") as jsonfile:
                                        jsonfile.write(temp_json_object)
                                    temp_interbb_dict.clear()
                                    print(inter_end_dict)
                            else:
                                pair_outvid.release()
                                inter_end_dict[pair] = end_frame
                                # interdims_dict[inter_num] = [temp_interbb_dict]
                                # print(interdims_dict)
                                temp_json_object = json.dumps(temp_interbb_dict, indent=4)
                                with open(pair_json_name, "w") as jsonfile:
                                    jsonfile.write(temp_json_object)
                                temp_interbb_dict.clear()
                                print(inter_end_dict)
                                break
                    else:
                        continue
                else:
                    continue
            else:
                continue

visualize(iden=1, categ=2, confid=0.9432)



