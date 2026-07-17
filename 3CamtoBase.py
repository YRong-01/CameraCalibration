import numpy as np


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
   [1.51339478e+03, 0.00000000e+00, 9.92839853e+02],
    [0.00000000e+00, 1.50238718e+03, 4.88597228e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist = np.array([
   [-5.51964568e-03, 6.93801723e-01, 3.26991913e-04, 4.77776670e-03, -6.15503232e+00]
])


# ============================================================
# 2. solvePnP result: checkerboard frame -> camera frame
# p_cam = R_cam_checker @ p_checker + t_cam_checker
# ============================================================

R_cam_checker = np.array([
    [-0.74779584, -0.65192637, -0.12567177],
    [0.48554782, -0.66609337, 0.56618277],
    [-0.45281861, 0.36236947, 0.81464328]
])

t_cam_checker = np.array([
    [0.03043978],
    [0.0388617 ],
    [0.79775328]
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