import cv2
import numpy as np

CHECKERBOARD = (7, 8)
SQUARE_SIZE = 0.025

# Your intrinsic result
camera_matrix = np.array([
    [1421.35831, 0, 968.022155],
    [0, 1407.63866, 548.181164],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    [0.0403473, -0.12000733, -0.00167514, 0.00079997, 0.09664386]
], dtype=np.float32)

# Create 3D checkerboard points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp = objp * SQUARE_SIZE

# Load image
img = cv2.imread("./images/test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find checkerboard corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    corners2 = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Estimate extrinsic
    success, rvec, tvec = cv2.solvePnP(
        objp,
        corners2,
        camera_matrix,
        dist_coeffs
    )

    R, _ = cv2.Rodrigues(rvec)

    print("Rotation vector:")
    print(rvec)

    print("Rotation matrix:")
    print(R)

    print("Translation vector:")
    print(tvec)

else:
    print("Checkerboard not found.")