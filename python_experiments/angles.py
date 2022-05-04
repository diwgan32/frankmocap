# Use FlaskServerTestBench to run
import numpy as np
import csv
import time

def get_joint(data, joint_name, frame_no):
    point = np.array([
        data[frame_no][f"{joint_name}.x"],
        data[frame_no][f"{joint_name}.y"],
        data[frame_no][f"{joint_name}.z"],
    ])
    return point

# N is number of fingers
def grad_F(x, S, fingers, wrist, N=4):
    sums = np.sum(fingers, axis=0)
    mat = np.array([
        [S[0][0], S[0][1], S[0][2], sums[0], wrist[0], x[0], 0],
        [S[1][0], S[1][1], S[1][2], sums[1], wrist[1], x[1], 0],
        [S[2][0], S[2][1], S[2][2], sums[2], wrist[2], x[2], 0],
        [sums[0], sums[1], sums[2], 4, 1, 0, 0],
        [wrist[0], wrist[1], wrist[2], 1, 0, 0, 0],
        [x[0], x[1], x[2], 0, 0, 0, -1] 
    ])
    
    x0 = np.array([x[0], x[1], x[2], x[3], x[4], x[5], 1])

    return np.dot(mat, x0)

def jac_F(x, S, fingers, wrist, N=4):
    sums = np.sum(fingers, axis=0)
    return np.array([
        [S[0][0] + x[5], S[0][1], S[0][2], sums[0], wrist[0], x[0]],
        [S[1][0], S[1][1]+x[5], S[1][2], sums[1], wrist[1], x[1]],
        [S[2][0], S[2][1], S[2][2] + x[5], sums[2], wrist[2], x[2]],
        [sums[0], sums[1], sums[2], 4, 1, 0],
        [wrist[0], wrist[1], wrist[2], 1, 0, 0],
        [2*x[0], 2*x[1], 2*x[2], 0, 0, 0]
    ])

def newton_solver(x, S, fingers, wrist, max_iter=1000, errtol=np.power(10., -5)):
    err = 2*errtol
    cc = 1
    x_now = x

    steps = np.concatenate((np.arange(.1, .001, -.005), np.arange(.1, .0005, -.005)))
    while (cc < max_iter and err > errtol):
        d = np.dot(-np.linalg.inv(jac_F(x_now, S, fingers, wrist)), grad_F(x_now, S, fingers, wrist))
        for s in steps:
            x_new = x_now + s*d

            if (np.linalg.norm(grad_F(x_new, S, fingers, wrist)) < 
                np.linalg.norm(grad_F(x_now, S, fingers, wrist))):
                break
            if (s == steps[-1]):
                raise ValueError("No proper step size could be found")

        err = np.linalg.norm(x_new - x_now)
        x_now = x_new
        cc = cc + 1

    if (cc > max_iter or err > errtol):
        raise ValueError("No convergence")
    return(x_now)

# fingers is 4 rows, one for pinky, ring, pointer, middle
# 3 columns for x, y, z
def get_flexion_angle_helper(wrist, elbow, fingers):
    S = np.dot(fingers.T, fingers)
    sol = newton_solver(np.array([1, 0, 0, 1, 1, 1]), S, fingers, wrist)
    normal = sol[0:3]
    
    testvector = (elbow - wrist)/np.linalg.norm(elbow - wrist) #from elbow to wrist
  
    testangle = (np.pi/2 - np.arccos(np.dot(testvector,normal)))*180.0/np.pi
  
    return(testangle)

def get_flexion_angle(wrist, elbow, pinky, ring, middle, pointer):
    return get_flexion_angle_helper(wrist, elbow, np.vstack((pinky, ring, middle, pointer)))

def cast_to_float_helper(row):
    for key in row.keys():
        row[key] = float(row[key])
    return row

if __name__ == "__main__":
    data = []
    line_count = 0
    with open('mocap_output_wrist_rom.csv', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')

        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                data.append(cast_to_float_helper(row))
            line_count += 1

        print(f'Processed {line_count} lines.')

    # Exlude CSV header
    num_frames = line_count - 1
    for i in range(num_frames):
        t = time.time()
        l_angle = get_flexion_angle(data, i, "L")
        elapsed = time.time() - t
        print(f"Took {elapsed} seconds")
        r_angle = get_flexion_angle(data, i, "R")

    
