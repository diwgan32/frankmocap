import numpy as np

def get_hrnet_bbox_helper(joints, frame_shape):
    x_min = int(max(np.min(joints[:, 0]) - 30, 0))
    x_max = int(min(np.max(joints[:, 0]) + 30, frame_shape[1]))
    y_min = int(max(np.min(joints[:, 1]) - 30, 0))
    y_max = int(min(np.max(joints[:, 1]) + 30, frame_shape[0]))
    return np.array([x_min, y_min, x_max - x_min, y_max - y_min])

def get_hrnet_hand_bbox(op_output, frame_shape):
    ret = []
    ret.append({})
    right_hand = op_output["hand_right_keypoints_2d"]
    ret[0]["right_hand"] = get_hrnet_bbox_helper(right_hand, frame_shape)
    left_hand = op_output["hand_left_keypoints_2d"]
    ret[0]["left_hand"] = get_hrnet_bbox_helper(left_hand, frame_shape)
    return ret

def get_hrnet_person_bbox(op_output, frame_shape):
    ret = []
    kpts = op_output["pose_keypoints_2d"]
    ret.append(get_hrnet_bbox_helper(kpts, frame_shape))
    return ret

def read_hrnet_wHand(joint_data, gt_part=None, dataset=None):

    if gt_part is None:
        gt_part = np.zeros([24,3])
    if dataset is None:
        dataset ='coco'
    op_output ={}

    # read the openpose detection
    if len(joint_data) == 0:
        # no openpose detection
        # keyp25 = np.zeros([25,3])
        return None, None
    else:
        # size of person in pixels
        
        op_output["pose_keypoints_2d"] = np.zeros((25, 3))
        l_shoulder = joint_data[0]["keypoints"][5]
        r_shoulder = joint_data[0]["keypoints"][6]
        l_hip = joint_data[0]["keypoints"][11]
        r_hip = joint_data[0]["keypoints"][12]
        op_output["pose_keypoints_2d"] = joint_data[0]["keypoints"][[0, -1, 6, 8, 10, 5, 7, 9,  -1, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 20, 21, 22]]
        op_output["pose_keypoints_2d"][1] = (l_shoulder + r_shoulder)/2.0
        op_output["pose_keypoints_2d"][8] = (l_hip + r_hip)/2.0
        op_output["pose_keypoints_2d"][:, -1] = np.where(op_output["pose_keypoints_2d"][:, -1] > .5,
                op_output["pose_keypoints_2d"][:, -1],
                0)
        op_output["hand_right_keypoints_2d"] = joint_data[0]["keypoints"][112:112+21]
        op_output["hand_left_keypoints_2d"] = joint_data[0]["keypoints"][91:91+21]
        op_output["hand_right_keypoints_2d"][:, -1] = np.where(op_output["hand_right_keypoints_2d"][:, -1] > .05,
                op_output["hand_right_keypoints_2d"][:, -1],
                0)
        op_output["hand_left_keypoints_2d"][:, -1] = np.where(op_output["hand_left_keypoints_2d"][:, -1] > .05,
                op_output["hand_left_keypoints_2d"][:, -1],
                0)
    return op_output 