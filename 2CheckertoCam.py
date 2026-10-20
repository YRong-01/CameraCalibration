import cv2
import numpy as np

CHECKERBOARD = (7, 8)
SQUARE_SIZE = 0.025

# Your intrinsic result
camera_matrix = np.array([
    [1.43553840e+03, 0, 9.52110915e+02],
    [0, 1.44008973e+03, 5.19050486e+02],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    [0.09119437, -0.50490922, -0.00144585, 0.00378291, 1.05208589]
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