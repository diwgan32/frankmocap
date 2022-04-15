import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import cv2

ROOT_FOLDER = "../handle_cans"
VIDEO_LOC = "/Volumes/Samsung_T5/TestVideos/handle_cans.mp4"

def generate_arrows(a, scores):
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


def display_3d_joints(joints3DList, fig, scores, color):
    arrow_locs, arrow_dirs = generate_arrows(joints3DList, scores)
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


def display_skeleton_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not ("pred_output_list" in data):
        return
    if (len(data["pred_output_list"]) == 0):
        return
    #right_hand = data["pred_output_list"][0]["right_hand"]["pred_joints_img"]
    right_hand = data["pred_output_list"][0]["pred_rhand_joints_img"]
    
    display_3d_joints(right_hand, fig, np.ones(right_hand.shape[0]), "red")

    #left_hand = data["pred_output_list"][0]["left_hand"]["pred_joints_img"]
    left_hand = data["pred_output_list"][0]["pred_lhand_joints_img"]
    display_3d_joints(left_hand, fig, np.ones(left_hand.shape[0]), "blue")
    plt.show()

def display_skeleton_2d(data, frame):
    if not ("pred_output_list" in data):
        return
    if (len(data["pred_output_list"]) == 0):
        return

    right_hand = data["pred_output_list"][0]["pred_rhand_joints_img"]
    left_hand = data["pred_output_list"][0]["pred_lhand_joints_img"]
    
    frame = display_2d_joints(right_hand, frame, (255, 0, 0))
    frame = display_2d_joints(left_hand, frame, (0, 0, 255))
    cv2.imshow("asdf", frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    pickle_files = glob.glob(f"{ROOT_FOLDER}/mocap/*.pkl")
    pickle_files.sort()
    cap = cv2.VideoCapture(VIDEO_LOC)
        
    for fname in pickle_files:
        ret, frame = cap.read()
        f = open(fname, "rb")
        data = pickle.load(f)
        f.close()
        #display_skeleton_3d(data)
        display_skeleton_2d(data, frame)
    cap.release()