import os.path
import sys
import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from torchvision import transforms
import torchvision.ops.boxes as bops
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
# from sort.sort_adj import *
from itertools import combinations
from sort.sort import *
from helpers import *

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True, track=True, nobbox=False, keep_bg=False,
        out_fps=20, outvid_dir='output/videos/', outjson_dir='output/jsons/', gband=2, min_inter_time=1):

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    dict_to_json = {}
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else:
        vidfile = str(os.path.splitext(source)[0]) + f'_{str(out_fps)}fps{os.path.splitext(source)[1]}'
        #vidfile = str(os.path.splitext(source)[0]) + f'_{str(out_fps)}fps.mp4'
        #fps_cmd = "ffmpeg -i {} -c:v libx264 -crf 0 -filter:v fps={} {}".format(source, str(out_fps), vidfile)
        fps_cmd = "ffmpeg -i {} -c:v libx264 -s 1920x1080 -crf 0 -filter:v fps={} {}".format(source, str(out_fps), vidfile)
        os.system(fps_cmd)
        #vidfile = source
        cap = cv2.VideoCapture(vidfile)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height
        print('video size = ({}*{})'.format(frame_width,frame_height))

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = outvid_dir + str(os.path.splitext(vidfile)[0]) + '_kpts.avi'
        create_dir_struct(os.path.split(out_video_name)[0])   # Create vid dir if missings
        out_json_name = outjson_dir + str(os.path.splitext(vidfile)[0]) + '_kpts.json'
        create_dir_struct(os.path.split(out_json_name)[0])    # Create json dir if missings
        print('Output video : {}'.format(out_video_name))
        out = cv2.VideoWriter(out_video_name,
                            cv2.VideoWriter_fourcc(*'MJPG'), out_fps,
                            (frame_width, frame_height))

        while(cap.isOpened): #loop until cap opened or video not complete
        
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if not keep_bg:
                    wr_im = 255*np.ones((frame_height, frame_width, 3), dtype=np.uint8)

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))

                        ## Tracker
                        dets_to_sort = np.empty((0, 6))
                        for x1, y1, x2, y2, conf, detclass in pose[:, :6].cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort,
                                                      np.array([x1, y1, x2, y2, conf, detclass])))

                        if opt.track:
                            tracked_dets = sort_tracker.update(dets_to_sort)    # add option unique_track_color later

                        arr_json = sort_tracks(dets_to_sort, tracked_dets, pose[:, 6:])

                        update_dict(frame_count+1, dict_to_json, arr_json)
                        #print('dict to json\n', dict_to_json)

                        for det_index, (*xyxy, idx) in enumerate(tracked_dets):
                            c = 0
                            kpts = pose[det_index, 6:]
                            #print(kpts)
                            if not keep_bg:
                                draw_bbox_kpts(wr_im, xyxy, identity=idx, kpts=kpts, names=names, colors=colors, steps=3,
                                           orig_shape=im0.shape[:2])
                            else:
                                draw_bbox_kpts(im0, xyxy, identity=idx, kpts=kpts, names=names, colors=colors, steps=3,
                                            orig_shape=im0.shape[:2])
                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    # cv2.imshow("keypoints with tracking", im0)
                    if not keep_bg:
                        cv2.imshow("keypoints with tracking", wr_im)
                        cv2.waitKey(1)  # 1 millisecond
                        out.write(wr_im)
                    else:
                        cv2.imshow("keypoints with tracking", im0)
                        cv2.waitKey(1)  # 1 millisecond
                        out.write(im0)

                else:
                    if not keep_bg:
                        out.write(wr_im)
                    else:
                        out.write(im0)

            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

        # write to json file
        json_object = json.dumps(dict_to_json, indent=4)
        with open(out_json_name, "w") as jsonfile:
            jsonfile.write(json_object)

        #plot the comparision graph
        #plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)

        # System code starts here
        inter_end_dict = {}
        pair_count = 1
        try:
            os.makedirs(outvid_dir + f'{str(os.path.splitext(vidfile)[0])}/')
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
                            pair_count += 1
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
                                    if p2_idx < num_person:
                                        plot_skeleton_kpts(white_bg,
                                                           torch.tensor(
                                                               dict_to_json[str(temp_frame)][p2_idx]["skeleton"]), 3,
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='weights/yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='inference/UT1/seq5.avi', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', default=False, help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    parser.add_argument('--track', default=True, action='store_true', help='run tracking')
    parser.add_argument('--nobbox', default=True, action='store_true', help='hide bbox')
    parser.add_argument('--keep_bg', default=False, action='store_true', help='white background')
    parser.add_argument('--out_fps', default=15, type=int, help='fps value')
    parser.add_argument('--outvid_dir', type=str, default='output/videos/', help='kpts video dir')
    parser.add_argument('--outjson_dir', type=str, default='output/jsons/', help='kpts json dir')
    parser.add_argument('--gband', default=2, type=int, help='+ or - band in seconds')
    parser.add_argument('--min_inter_time', default=1, type=int, help='min time for an interaction')
    opt = parser.parse_args()
    return opt

# function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")


# function to Draw Bounding boxes
def draw_bbox_kpts(img, bbox, identity=None, category=None, confidence=None, kpt_label=True, kpts=None, steps=2,
                   names=None, colors=None, orig_shape=None):
    x1, y1, x2, y2 = [int(i) for i in bbox]
    tl = opt.line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    cat = int(category) if category is not None else 0
    id = int(identity) if identity is not None else 0

    color = colors[cat]

    if not opt.nobbox:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

    if not opt.hide_labels:
        label = str(id) + ":" + names[cat] if identity is not None else f'{names[cat]} {confidence:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    if kpt_label:
        plot_skeleton_kpts(img, kpts, steps, orig_shape=orig_shape)

    return img

def sort_tracks(detout=np.zeros((0,4)), trkout=np.zeros((0,5)), kpts=np.zeros((0,51))):
    matched_ind = []
    sorted_trks = np.empty((0,56))
    numdets = len(detout)
    numtrks = len(trkout)
    print(detout)
    if numdets == 0:
        print('No Detections')
        return detout
    elif numtrks == 0:
        print('No tracks')
        return trkout
    else:
        for i in range(numdets):
            detbox = torch.tensor(np.reshape(detout[i, :4], (1,4)), dtype=torch.float)
            iou = 0
            iou_matched = False
            for j in range(numtrks):
                if j in matched_ind:
                    continue
                else:
                    trkbox = torch.tensor(np.reshape(trkout[j, :4], (1,4)), dtype=torch.float)
                    if ( bops.box_iou(detbox, trkbox)[0][0] > iou ):
                        iou = bops.box_iou(detbox, trkbox)[0][0]
                        matched_trk_idx = j
                        iou_matched = True
            if iou_matched:
                matched_ind.append(matched_trk_idx)
                matched_row = np.hstack((trkout[matched_trk_idx,4], trkout[matched_trk_idx,:4], kpts[i]))
                sorted_trks = np.vstack((sorted_trks, matched_row))
        print(sorted_trks)
        return sorted_trks

def update_dict(frame, dict, arr=np.empty((0,56))):
    person_list = []
    for row in arr:
        temp_dict = {"id": int(row[0]), "standing_loc": [(row[50] + row[53])/2, (row[51] + row[54])/2], "bbox": row[1:5].tolist(), "skeleton": row[5::].tolist()}
        person_list.append(temp_dict)
    dict[str(frame)] = person_list

def create_dir_struct(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

#main function
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    strip_optimizer(opt.device, opt.poseweights)
    # init tracker
    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)
    main(opt)
