# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
import pdb

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime
import demo.get_video_metadata as metadata

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector, Openpose_Hand_Detector
from integration.copy_and_paste import integration_copy_paste
from integration.eft import integration_eft_optimization


def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]
        hand_bbox_list = [hand_bbox_list[0], ]

    return body_bbox_list, hand_bbox_list


def run_regress(
    args, img_original_bgr, 
    body_bbox_list, hand_bbox_list, openpose_kp_imgcoord, bbox_detector,
    body_mocap, hand_mocap
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if (not cond1) and cond2:
            # run detection only when bbox is not available
            body_pose_list, body_bbox_list, hand_bbox_list, _ = \
                bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            # use openpose bbox
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        if args.use_openpose_bbox:
            assert openpose_file_path != '' and osp.exists(openpose_file_path)
            assert cond2, "Do not use openpose prediction in fm fast mode."
            assert args.single_person
            op_hand_bbox_list = Openpose_Hand_Detector.detect_hand_bbox(openpose_file_path, img_original_bgr)
            res_hand_bbox_list = [dict(left_hand = None, right_hand=None),]
            for hand_type in ['left_hand', 'right_hand']:
                if op_hand_bbox_list[0][hand_type] is None and hand_bbox_list[0][hand_type] is not None:
                    res_hand_bbox_list[0][hand_type] = hand_bbox_list[0][hand_type]
                else:
                    res_hand_bbox_list[0][hand_type] = op_hand_bbox_list[0][hand_type]
            hand_bbox_list = res_hand_bbox_list

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [None, ] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # body regression first 
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(pred_hand_list) 

    # integration by copy-and-paste
    if args.integrate_type=='copy_paste' or openpose_kp_imgcoord is None:
        print("Run copy-paste integration")
        integral_output_list = integration_copy_paste(
            pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape)
    # Optimization
    else:   
        print("Run optimization-based integration")
        integral_output_list = integration_eft_optimization(
            body_mocap, pred_body_list, pred_hand_list, 
            body_bbox_list, openpose_kp_imgcoord, 
            img_original_bgr, is_debug_vis=args.is_opt_debug_vis)
    
    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    rotate = 0
    if (input_type == 'video'):
        rotate = metadata.getVideoRotate(args.input_path)

    f = open(args.hrnet_joints_loc, "rb")
    joints_2d_hrnet = pickle.load(f)
    f.close()
    
    img_frame = 0
    while True:
        # load data
        load_bbox = False
        openpose_file_path = ''
        openpose_imgcoord = None

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

            if args.use_openpose_bbox:
                # Note: current openpose name should be {raw_image_name}_keypoints.json
                f_name = os.path.basename(image_path)[:-13] + "_keypoints.json"
                openpose_file_path = os.path.join(args.openpose_dir, f_name)
                assert os.path.exists(openpose_file_path), openpose_file_path

            # Optimization based integration. This requires openpose prediction
            if args.integrate_type=='opt':
                print(f"Loading openpose data from: {openpose_file_path}")
                # TODO: this works for single person in the image
                openpose_imgcoord = demo_utils.read_hrnet_wHand(joints_2d_hrnet[img_frame])

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
          # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")
        
        # bbox detection
        if not load_bbox:
            body_bbox_list, hand_bbox_list = list(), list()
        
        # regression (includes integration)
        # openpose_imgcoord is required for optimization-based integration
        body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
            args, img_original_bgr, 
            body_bbox_list, hand_bbox_list, openpose_imgcoord, bbox_detector,
            body_mocap, hand_mocap)
        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualization
        res_img = visualizer.visualize(
            img_original_bgr,
            rotate = rotate,
            pred_mesh_list = pred_mesh_list,
            body_bbox_list = body_bbox_list,
            hand_bbox_list = hand_bbox_list)
        
       # show result in the screen
        #if not args.no_display:
        #    res_img = res_img.astype(np.uint8)
        #    ImShow(res_img)

        # save result image
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'frank'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ['image_dir','video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector =  HandBboxDetector('third_view', device)
    
    #Set Mocap regressor
    body_mocap = BodyMocap(args.checkpoint_body_smplx, args.smpl_dir, device = device, use_smplx= True)
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)


    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == '__main__':
    main()
