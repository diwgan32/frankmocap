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
    if (op_output is None):
        return []
    right_hand = op_output["hand_right_keypoints_2d"]
    ret[0]["right_hand"] = get_hrnet_bbox_helper(right_hand, frame_shape)
    left_hand = op_output["hand_left_keypoints_2d"]
    ret[0]["left_hand"] = get_hrnet_bbox_helper(left_hand, frame_shape)
    if (np.any(ret[0]["right_hand"] < 0 )):
        ret[0]["right_hand"] = None
    if (np.any(ret[0]["left_hand"] < 0)):
        ret[0]["left_hand"] = None
    return ret

def get_hrnet_person_bbox(op_output, frame_shape):
    ret = []
    if (op_output is None):
        return ret
    kpts = op_output["pose_keypoints_2d"]
    ret.append(get_hrnet_bbox_helper(kpts, frame_shape))
    if (np.any(ret[0] < 0)):
        return [None]
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

def _wrnch_get_person(frame, person_id):
    for i in range(len(frame)):
        if (frame[i]["id"] == person_id):
            return frame[i]
    return []

def scale_up(joints, frame_size):
    scores = joints[:, -1]
    joints[:, 0] = np.where(scores > 0, joints[:, 0]*frame_size[1], -1.0)
    joints[:, 1] = np.where(scores > 0, joints[:, 1]*frame_size[0], -1.0)

    return joints

def read_wrnch_wHand(wrnch_data, frame_size, gt_part=None, dataset=None):
    if gt_part is None:
        gt_part = np.zeros([24,3])
    if dataset is None:
        dataset ='coco'
    op_output = {}
    joint_data = _wrnch_get_person(wrnch_data["persons"], 0)
    if (not "pose2d" in joint_data):
        return None

    arr_2d = np.array(joint_data["pose2d"]["joints"]).reshape((25, 2))
    scores = np.array(joint_data["pose2d"]["scores"])
    # read the openpose detection
    if len(arr_2d) == 0:
        # no openpose detection
        # keyp25 = np.zeros([25,3])
        return None
    # size of person in pixels
    wrnch_to_openpose = [16, 8, 12, 11, 10, 13, 14, 15, 6, 2, 1, 0, 3, 4, 5, 17, 19, 18, 20, 22, -1, 24, 21, -1, 23] 
    op_output["pose_keypoints_2d"] = np.zeros((25, 3))
    op_output["pose_keypoints_2d"][:, :2] = arr_2d[wrnch_to_openpose]
    op_output["pose_keypoints_2d"][20, -1] = 0
    op_output["pose_keypoints_2d"][23, -1] = 0
    op_output["pose_keypoints_2d"][:, -1] = scores[wrnch_to_openpose]
    op_output["pose_keypoints_2d"] = scale_up(op_output["pose_keypoints_2d"], frame_size)
    hand_array_right = np.zeros((21, 3))
    hand_array_left = np.zeros((21, 3))
    hand_array_right_scores = np.ones(21)
    hand_array_left_scores = np.ones(21)
    op_output["hand_right_keypoints_2d"] = np.zeros((21, 3))
    op_output["hand_left_keypoints_2d"] = np.zeros((21, 3))
    if ("hand_pose" in joint_data):
        if (("right" in joint_data["hand_pose"])):
            hand_array_right = np.array(joint_data["hand_pose"]["right"]["joints"]).reshape((21, 2))
#            hand_array_right_scores = np.array(joint_data["hand_pose"]["right"]["scores"])

        if (("left" in joint_data["hand_pose"])):
            hand_array_left = np.array(joint_data["hand_pose"]["left"]["joints"]).reshape((21, 2))
#            hand_array_left_scores = np.array(joint_data["hand_pose"]["left"]["scores"])
    wrnch_to_openpose_hands = [0, 1, 6, 7, 8, 2, 10, 11, 12, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20] 
    op_output["hand_right_keypoints_2d"][:, :2] = hand_array_right[wrnch_to_openpose_hands]
    op_output["hand_left_keypoints_2d"][:, :2] = hand_array_left[wrnch_to_openpose_hands]
    op_output["hand_right_keypoints_2d"][:, -1] = hand_array_right_scores
    op_output["hand_left_keypoints_2d"][:, -1] = hand_array_left_scores
    op_output["hand_right_keypoints_2d"][:, -1] = np.where(op_output["hand_right_keypoints_2d"][:, -1] > .05,
            op_output["hand_right_keypoints_2d"][:, -1],
            0)
    op_output["hand_left_keypoints_2d"][:, -1] = np.where(op_output["hand_left_keypoints_2d"][:, -1] > .05,
            op_output["hand_left_keypoints_2d"][:, -1],
            0)
    op_output["hand_right_keypoints_2d"] = scale_up(op_output["hand_right_keypoints_2d"], frame_size)
    op_output["hand_left_keypoints_2d"] = scale_up(op_output["hand_left_keypoints_2d"], frame_size)
    return op_output 
