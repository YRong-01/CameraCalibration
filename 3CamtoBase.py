import numpy as np


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
    [1.43553840e+03, 0, 9.52110915e+02],
    [0, 1.44008973e+03, 5.19050486e+02],
    [0, 0, 1]
])

dist = np.array([
    [0.09119437, -0.50490922, -0.00144585, 0.00378291, 1.05208589]
])


# ============================================================
# 2. solvePnP result: checkerboard frame -> camera frame
# p_cam = R_cam_checker @ p_checker + t_cam_checker
# ============================================================

R_cam_checker = np.array([
[ 0.99755869, -0.06862797, -0.01291729],
 [ 0.06142269,  0.77426959,  0.62986812],
 [-0.03322511, -0.62912384,  0.77659468]
])

t_cam_checker = np.array([
   [-0.03293103],
 [-0.1244683 ],
 [ 0.94162713]
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
    41.6,
    22.9,
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