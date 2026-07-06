import numpy as np


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
    [1421.35831, 0.0,        968.022155],
    [0.0,        1407.63866, 548.181164],
    [0.0,        0.0,        1.0]
])

dist = np.array([
    [0.0403473, -0.12000733, -0.00167514, 0.00079997, 0.09664386]
])


# ============================================================
# 2. solvePnP result: checkerboard frame -> camera frame
# p_cam = R_cam_checker @ p_checker + t_cam_checker
# ============================================================

R_cam_checker = np.array([
    [ 0.8572316,  0.51492901, -0.0205644 ],
    [-0.39851308,   0.66163901,  0.63515442],
    [  0.32610016, -0.54505223,   0.77238381]
])

t_cam_checker = np.array([
    [-0.24727995],
    [-0.01455042],
    [ 0.83672234]
])


T_cam_checker = np.eye(4)
T_cam_checker[:3, :3] = R_cam_checker
T_cam_checker[:3, 3] = t_cam_checker.flatten()


# ============================================================
# 3. Checkerboard origin in robot base frame
# Measured value: [29.7, 12.35, -3.81] cm
# Convert cm -> meters
# ============================================================

t_base_checker = np.array([
    29.7,
    12.35,
    -3.81
]) / 100.0


# ============================================================
# 4. Checkerboard orientation in robot base frame
#
# Assumption:
# checkerboard +x = base +x
# checkerboard +y = base -y
# checkerboard +z = base -z
# ============================================================

R_base_checker = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])


T_base_checker = np.eye(4)
T_base_checker[:3, :3] = R_base_checker
T_base_checker[:3, 3] = t_base_checker


# ============================================================
# 5. Compute camera frame -> robot base frame
#
# T_base_cam = T_base_checker @ inverse(T_cam_checker)
# ============================================================

T_checker_cam = np.linalg.inv(T_cam_checker)
T_base_cam = T_base_checker @ T_checker_cam


print("T_cam_checker:")
print(T_cam_checker)

print("\nT_base_checker:")
print(T_base_checker)

print("\nT_base_cam:")
print(T_base_cam)

print("\nCamera origin in robot base frame:")
print(T_base_cam[:3, 3])