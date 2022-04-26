import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import cv2

ROOT_FOLDER = "../handle_cans_torch"
VIDEO_LOC = "/Volumes/Samsung_T5/TestVideos/handle_cans.mp4"
JOINTS_2D = "../handle_cans_hands.pkl"
JOINTS_3D = "../handle_cans_3d.p"

def generate_arrows_hands(a, scores):
    # These are the joint connections pulled in by hand from joint definitions
    connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                   [0, 5], [5, 6], [6, 7], [7, 8],
                   [0, 9], [9, 10], [10, 11], [11, 12],
                   [0, 13], [13, 14], [14, 15], [15, 16],
                   [0, 17], [17, 18], [18, 19], [19, 20]]
    arrow_locs = []
    arrow_dirs = []
    for connect in connections:
        if (scores[connect[0]] < .25 or scores[connect[1]] < .25):
            continue
        arrow_locs.append(a[connect[0]])
        arrow_dirs.append(a[connect[1]] - a[connect[0]])
    if (arrow_dirs == []):
        arrow_dirs = [[0, 0, 0]]
        arrow_locs = [[0, 0, 0]]
    return np.stack(arrow_locs), np.stack(arrow_dirs)

def generate_arrows_body(a, scores):
    # These are the joint connections pulled in by hand from joint definitions
    connections = [[0, 1], [1, 2], [2, 3], [0, 4],
    [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [9, 11], 
    [11, 12], [12, 13], [9, 14], [14, 15], [15, 16]]
    arrow_locs = []
    arrow_dirs = []
    for connect in connections:
        if (scores[connect[0]] < .25 or scores[connect[1]] < .25):
            continue
        arrow_locs.append(a[connect[0]])
        arrow_dirs.append(a[connect[1]] - a[connect[0]])
    if (arrow_dirs == []):
        arrow_dirs = [[0, 0, 0]]
        arrow_locs = [[0, 0, 0]]
    return np.stack(arrow_locs), np.stack(arrow_dirs)

def display_3d_body(joints3DList, fig, scores, color):
    arrow_locs, arrow_dirs = generate_arrows_body(joints3DList, scores)
    if (fig is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.get_axes()[0]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.quiver(arrow_locs[:, 0],
              arrow_locs[:, 1],
              arrow_locs[:, 2],
              arrow_dirs[:, 0],
              arrow_dirs[:, 1],
              arrow_dirs[:, 2],
              arrow_length_ratio=.01,
              color=color)
    return ax

def display_3d_hands(joints3DList, fig, scores, color):
    arrow_locs, arrow_dirs = generate_arrows_hands(joints3DList, scores)
    if (fig is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.get_axes()[0]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.quiver(arrow_locs[:, 0],
              -arrow_locs[:, 1],
              arrow_locs[:, 2],
              arrow_dirs[:, 0],
              -arrow_dirs[:, 1],
              arrow_dirs[:, 2],
              arrow_length_ratio=.01,
              color=color)
    return ax

def display_2d_joints(joints_2d, frame, color):
    connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                   [0, 5], [5, 6], [6, 7], [7, 8],
                   [0, 9], [9, 10], [10, 11], [11, 12],
                   [0, 13], [13, 14], [14, 15], [15, 16],
                   [0, 17], [17, 18], [18, 19], [19, 20]]
    arrow_locs = []
    arrow_dirs = []
    for i in range(joints_2d.shape[0]):
        frame = cv2.circle(frame, (int(joints_2d[i][0]), int(joints_2d[i][1])), 4, color, -1)

    return frame


def display_hand_skeleton_3d(data, l_wrist, r_wrist, fig=None):
    if (fig is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if not ("pred_output_list" in data):
        return None
    if (len(data["pred_output_list"]) == 0):
        return None
    #right_hand = data["pred_output_list"][0]["right_hand"]["pred_joints_img"]
    right_hand = data["pred_output_list"][0]["pred_rhand_joints_img"]
    right_hand -= right_hand[0]
    r_wrist[1] *= -1
    l_wrist[1] *= -1
    right_hand += l_wrist*800
    display_3d_hands(right_hand/800, fig, np.ones(right_hand.shape[0]), "red")

    #left_hand = data["pred_output_list"][0]["left_hand"]["pred_joints_img"]
    left_hand = data["pred_output_list"][0]["pred_lhand_joints_img"]
    left_hand -= left_hand[0]
    
    left_hand += r_wrist*800
    display_3d_hands(left_hand/800, fig, np.ones(left_hand.shape[0]), "blue")
    return fig

def display_body_skeleton_3d(data, fig=None):
    if (fig is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if (len(data) == 0):
        return None
    #right_hand = data["pred_output_list"][0]["right_hand"]["pred_joints_img"]
    body = data[0]["keypoints_3d"]
    print(body)
    display_3d_body(body, fig, np.ones(body.shape[0]), "black")

    return fig

def display_skeleton_2d(data, joints_2d_hrnet, frame):
    if not ("pred_output_list" in data):
        return
    if (len(data["pred_output_list"]) == 0):
        return

    right_hand = data["pred_output_list"][0]["pred_rhand_joints_img"]
    left_hand = data["pred_output_list"][0]["pred_lhand_joints_img"]

    left_hand_coco = joints_2d_hrnet[0]["keypoints"][21:, :]
    print(left_hand_coco)
    frame = display_2d_joints(right_hand, frame, (255, 0, 0))
    frame = display_2d_joints(left_hand, frame, (0, 0, 255))
    frame = display_2d_joints(left_hand_coco, frame, (0, 255, 255))
    cv2.imshow("asdf", frame)
    cv2.waitKey(0)

def display_full_body_3d(hand_data, body_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    body = body_data[0]["keypoints_3d"]

    display_body_skeleton_3d(body_data, fig=fig)
    left_wrist = body[13]
    right_wrist = body[16]
    display_hand_skeleton_3d(hand_data, left_wrist, right_wrist, fig=fig)
    
    plt.show()

if __name__ == "__main__":
    pickle_files = glob.glob(f"{ROOT_FOLDER}/mocap/*.pkl")
    pickle_files.sort()

    f = open(JOINTS_2D, "rb")
    joints_2d_hrnet = pickle.load(f)
    f.close()

    f = open(JOINTS_3D, "rb")
    joints_3d = pickle.load(f)
    f.close()
    cap = cv2.VideoCapture(VIDEO_LOC)
    i = 0
    for fname in pickle_files:
        ret, frame = cap.read()
        f = open(fname, "rb")
        data = pickle.load(f)
        f.close()
        display_full_body_3d(data, joints_3d[i])
        #display_skeleton_2d(data, joints_2d_hrnet[i], frame)
        i += 1
    cap.release()