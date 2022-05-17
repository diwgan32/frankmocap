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
from mocap_utils.timer import Timer
from datetime import datetime
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap

from integration.eft import integration_eft_optimization
import inference.utils as inference_utils

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


def run_regress_helper(
    img_original_bgr, 
    body_bbox_list, hand_bbox_list, openpose_kp_imgcoord,
    body_mocap, hand_mocap, prev_integral_output_list
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    
    # use pre-computed bbox or use slow detection mode
    if not cond1:
        return list(), list(), list()

    if len(body_bbox_list) < 1: 
        return list(), list(), list()

    # sort the bbox using bbox size 
    # only keep on bbox if args.single_person is set
    body_bbox_list, hand_bbox_list = __filter_bbox_list(
        body_bbox_list, hand_bbox_list, True)

    # hand & body pose regression
    pred_hand_list = hand_mocap.regress(
        img_original_bgr, hand_bbox_list, add_margin=True)
    pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
    assert len(hand_bbox_list) == len(pred_hand_list)
    assert len(pred_hand_list) == len(pred_body_list)

    if not (prev_integral_output_list is None):
        prev_left_hands = prev_integral_output_list[0]["pred_lhand_joints_weak"]
        prev_right_hands = prev_integral_output_list[0]["pred_rhand_joints_weak"]
    else:
        prev_left_hands = None
        prev_right_hands = None
    integral_output_list = integration_eft_optimization(
        body_mocap, pred_body_list, pred_hand_list, 
        body_bbox_list, openpose_kp_imgcoord, 
        img_original_bgr, prev_left_hands, prev_right_hands, is_debug_vis=False
    )

    return body_bbox_list, hand_bbox_list, integral_output_list


def run_regress(
    img_original_bgr,
    hrnet_kp_imgcoord,
    body_mocap, hand_mocap,
    prev_integral_output_list
):
    openpose_imgcoord = inference_utils.read_hrnet_wHand(hrnet_kp_imgcoord)
    hand_bbox_list = inference_utils.get_hrnet_hand_bbox(openpose_imgcoord, img_original_bgr.shape)
    body_bbox_list = inference_utils.get_hrnet_person_bbox(openpose_imgcoord, img_original_bgr.shape)

    return run_regress_helper(img_original_bgr, body_bbox_list, hand_bbox_list, openpose_imgcoord,
        body_mocap, hand_mocap, prev_integral_output_list)

if __name__ == '__main__':
    VIDEO_FILENAME = '/home/fsuser/videos/hand_occlusion.mov'
    JOINT_DATA_FILENAME = '/home/fsuser/pickle_files/hand_occlusion_hands.p'
    cap = cv2.VideoCapture(VIDEO_FILENAME)
    _, frame = cap.read()
    cap.release()
    with open(JOINT_DATA_FILENAME, 'rb') as f:
        data = pickle.load(f)

    prev_integral_output_list = None

    default_checkpoint_body_smplx ='./extra_data/body_module/pretrained_weights/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
    default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    
    device = torch.device('cuda')
    body_mocap = BodyMocap(default_checkpoint_body_smplx, './extra_data/smpl/', device = device, use_smplx= True)
    hand_mocap = HandMocap(default_checkpoint_hand, './extra_data/smpl/', device = device)

    run_regress(frame, data[0], body_mocap, hand_mocap, None)
