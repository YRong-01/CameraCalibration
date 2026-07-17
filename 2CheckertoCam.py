import os

import cv2
import numpy as np

CHECKERBOARD = (7, 8)
SQUARE_SIZE = 0.025

# Your intrinsic result
camera_matrix = np.array([
    [1.51339478e+03, 0.00000000e+00, 9.92839853e+02],
    [0.00000000e+00, 1.50238718e+03, 4.88597228e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([
    -5.51964568e-03, 6.93801723e-01, 3.26991913e-04, 4.77776670e-03, -6.15503232e+00
], dtype=np.float32)

# Create 3D checkerboard points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp = objp * SQUARE_SIZE

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")
# pick a checkerboard image; prefer test.jpg if present
preferred = os.path.join(images_dir, "test.jpg")
if os.path.exists(preferred):
    image_path = preferred
else:
    jpgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    if not jpgs:
        raise FileNotFoundError(f"Could not find any .jpg images in: {images_dir}")
    image_path = jpgs[0]

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read checkerboard image: {image_path}")

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