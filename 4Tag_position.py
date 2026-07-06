import cv2
import numpy as np


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
    [1421.35831, 0.0,        968.022155],
    [0.0,        1407.63866, 548.181164],
    [0.0,        0.0,        1.0]
], dtype=np.float32)

dist = np.array([
    [0.0403473, -0.12000733, -0.00167514, 0.00079997, 0.09664386]
], dtype=np.float32)


# ============================================================
# 2. Camera frame -> robot base frame
# Your latest T_base_cam
# ============================================================

T_base_cam = np.array([
    [ 0.85725896, -0.3865265,   0.34067652,  0.21830717],
    [-0.51494545, -0.66883921,  0.53629638, -0.46229873],
    [-0.0014499,  -0.6351747,  -0.77240847,  0.59859083],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float32)

T_cam_base = np.linalg.inv(T_base_cam)
print("T_cam_base = \n", T_cam_base)

# ============================================================
# 3. ArUco marker settings
# ============================================================

MARKER_SIZE = 0.0368  # 5 cm = 0.05 m

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)


# ============================================================
# 4. 3D marker corner coordinates in marker frame
#
# Marker center is the origin: (0, 0, 0)
# z = 0 because the marker is flat on the table
# ============================================================

half_size = MARKER_SIZE / 2.0

marker_3d_points = np.array([
    [-half_size,  half_size, 0.0],   # top-left corner
    [ half_size,  half_size, 0.0],   # top-right corner
    [ half_size, -half_size, 0.0],   # bottom-right corner
    [-half_size, -half_size, 0.0]    # bottom-left corner
], dtype=np.float32)


# ============================================================
# 5. Open camera
#
# Change the camera index if needed:
# 0 = default laptop camera
# 1 or 2 = possible external/robot camera
# ============================================================

camera_index = 1
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {camera_index}.")


# ============================================================
# 6. Main loop
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            tag_id = int(ids[i][0])

            # 2D pixel coordinates of this marker's corners
            marker_2d_points = corners[i][0].astype(np.float32)

            # Estimate marker pose relative to camera
            success, rvec, tvec = cv2.solvePnP(
                marker_3d_points,
                marker_2d_points,
                K,
                dist
            )

            if not success:
                continue

            # Draw coordinate axes on marker
            cv2.drawFrameAxes(
                frame,
                K,
                dist,
                rvec,
                tvec,
                MARKER_SIZE * 0.5
            )

            # tvec is marker center position in camera frame
            tvec = tvec.flatten()

            p_cam = np.array([
                tvec[0],
                tvec[1],
                tvec[2],
                1.0
            ], dtype=np.float32)

            # Convert marker center from camera frame to robot base frame
            p_base = T_base_cam @ p_cam

            print(f"Tag ID: {tag_id}")
            print(f"Position in camera frame: {p_cam[:3]}")
            print(f"Position in robot base frame: {p_base[:3]}")
            print("--------------------------------")

            # Display result on image
            center_pixel = marker_2d_points.mean(axis=0).astype(int)
            u, v = center_pixel

            cv2.putText(
                frame,
                f"ID {tag_id}",
                (u - 40, v - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Base: ({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})",
                (u - 120, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("ArUco Tag Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# ============================================================
# 7. Clean up
# ============================================================

cap.release()
cv2.destroyAllWindows()