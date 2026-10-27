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

    # The first detected inner corner corresponds to objp[0] = (0, 0, 0).
    origin = tuple(np.round(corners2[0, 0]).astype(int))
    cv2.circle(img, origin, 10, (0, 0, 255), -1)
    cv2.putText(img, "Origin (0,0,0)", (origin[0] + 15, origin[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

    cv2.namedWindow("Checkerboard origin", cv2.WINDOW_NORMAL)
    cv2.imshow("Checkerboard origin", img)
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Checkerboard not found.")
