import cv2
import numpy as np
import time


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
    [1.43553840e+03, 0, 9.52110915e+02],
        [0, 1.44008973e+03, 5.19050486e+02],
        [0, 0, 1]
], dtype=np.float32)

CALIBRATED_FRAME_WIDTH = 1920
CALIBRATED_FRAME_HEIGHT = 1080

dist = np.array([
    [0.09119437, -0.50490922, -0.00144585, 0.00378291, 1.05208589]
], dtype=np.float32)


# ============================================================
# 2. Camera frame -> robot base frame
# Your latest T_base_cam
# ============================================================

T_base_cam = np.array([
    [ 0.9975587,   0.06142269, -0.03322511,  0.48778147],
    [ 0.06862797, -0.77426959,  0.62912383, -0.4575121 ],
    [ 0.01291729, -0.62986813, -0.77659467,  0.61518938],
    [ 0.         ,  0.         ,  0.         ,  1.        ]
], dtype=np.float32)

T_cam_base = np.linalg.inv(T_base_cam)
print("T_cam_base = \n", T_cam_base)


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to roll-pitch-yaw angles in radians."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], dtype=np.float32)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.
    q0, q1 are [w, x, y, z]
    t in [0,1]
    """
    # Normalize inputs
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Very close - use linear interpolation
        q = q0 + t * (q1 - q0)
        q /= np.linalg.norm(q)
        return q

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = (s0 * q0) + (s1 * q1)
    q /= np.linalg.norm(q)
    return q


# ============================================================
# 3. AprilTag marker settings
# ============================================================

MARKER_SIZE = 0.0457 # measure the sidelength in meters

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

detector = None
if hasattr(cv2.aruco, "ArucoDetector"):
    # OpenCV 4.7+ API
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        aruco_params
    )
else:
    # OpenCV 4.6 and earlier API
    aruco_params = cv2.aruco.DetectorParameters_create()


def detect_apriltag_markers(gray: np.ndarray):
    """Detect AprilTags with both old and new OpenCV ArUco APIs."""
    gray = np.ascontiguousarray(gray)

    if detector is not None:
        return detector.detectMarkers(gray)

    return cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=aruco_params
    )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open the camera and request the resolution used for calibration."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that no other program is currently using the camera."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATED_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATED_FRAME_HEIGHT)
    return cap


def camera_matrix_for_frame(frame: np.ndarray) -> np.ndarray:
    """Scale the calibrated intrinsics to the actual captured frame size."""
    frame_height, frame_width = frame.shape[:2]

    if (
        frame_width == CALIBRATED_FRAME_WIDTH
        and frame_height == CALIBRATED_FRAME_HEIGHT
    ):
        return K

    scale_x = frame_width / CALIBRATED_FRAME_WIDTH
    scale_y = frame_height / CALIBRATED_FRAME_HEIGHT

    K_frame = K.copy()
    K_frame[0, 0] *= scale_x
    K_frame[0, 2] *= scale_x
    K_frame[1, 1] *= scale_y
    K_frame[1, 2] *= scale_y
    return K_frame


# ============================================================
# 4. 3D marker corner coordinates in marker frame
#
# The marker center is the origin (0, 0, 0), and the marker's local
# x-y plane lies on its printed surface. Therefore every corner has local
# z = 0 because an AprilTag is planar. The tag may have any position and
# orientation in the camera/robot frame; solvePnP estimates that 3D pose.
# ============================================================

half_size = MARKER_SIZE / 2.0

marker_3d_points = np.array([
    [-half_size,  half_size, 0.0],   # top-left corner
    [ half_size,  half_size, 0.0],   # top-right corner
    [ half_size, -half_size, 0.0],   # bottom-right corner
    [-half_size, -half_size, 0.0]    # bottom-left corner
], dtype=np.float32)


def estimate_marker_pose(marker_2d_points: np.ndarray, K_frame: np.ndarray):
    """Estimate an AprilTag pose in the robot base frame."""
    success, rvec, tvec = cv2.solvePnP(
        marker_3d_points,
        marker_2d_points,
        K_frame,
        dist,
        flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        return None

    p_cam = np.append(tvec.reshape(3), 1.0).astype(np.float32)
    p_base = T_base_cam @ p_cam
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    R_base_marker = T_base_cam[:3, :3] @ R_cam_marker
    q_base_marker = rotation_matrix_to_quaternion(R_base_marker).astype(np.float32)

    return {
        "pos": p_base[:3].astype(np.float32),
        "euler_deg": np.degrees(
            rotation_matrix_to_euler_xyz(R_base_marker)
        ).astype(np.float32),
        "q": q_base_marker,
        "rvec": rvec,
        "tvec": tvec,
    }


def find_tag_poses(
    tag_ids,
    camera_index: int = 4,
    num_samples: int = 10,
    timeout_seconds: float = 60.0,
):
    """Capture the first available base-frame pose for each AprilTag ID.

    ``num_samples`` is retained for compatibility with existing callers but is
    intentionally ignored; poses are no longer averaged across frames.
    """
    requested_ids = {int(tag_id) for tag_id in tag_ids}
    results = {}
    cap = open_camera(camera_index)
    start_time = time.monotonic()

    try:
        while time.monotonic() - start_time < timeout_seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            K_frame = camera_matrix_for_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_apriltag_markers(gray)
            if ids is None:
                continue

            flat_ids = np.asarray(ids).reshape(-1)
            for index, detected_id in enumerate(flat_ids):
                tag_id = int(detected_id)
                if tag_id not in requested_ids or tag_id in results:
                    continue

                marker_2d_points = corners[index][0].astype(np.float32)
                pose = estimate_marker_pose(marker_2d_points, K_frame)
                if pose is None:
                    continue

                results[tag_id] = {
                    "pos": pose["pos"],
                    "euler_deg": pose["euler_deg"],
                }
                print(f"Captured AprilTag {tag_id} pose")

            if requested_ids.issubset(results):
                break
    finally:
        cap.release()

    return results


def main():
    # ============================================================
    # 5. Open camera
    #
    # Change the camera index if needed:
    # Use camera_device_test.py to find the index of the camera connected
    # ============================================================

    camera_index = 4
    cap = open_camera(camera_index)

    # ============================================================
    # 6. Main loop
    # ============================================================

    # Per-tag orientation state: stores smoothed quaternion and last displayed euler angles
    orientation_state = {}
    slerp_alpha = 0.3  # how much of the new orientation to blend in (0..1)
    angle_alpha = 0.3  # smoothing for displayed Euler angles
    printed_frame_info = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        K_frame = camera_matrix_for_frame(frame)

        if not printed_frame_info:
            frame_height, frame_width = frame.shape[:2]
            print(f"Actual camera frame size: {frame_width}x{frame_height}")
            if (
                frame_width != CALIBRATED_FRAME_WIDTH
                or frame_height != CALIBRATED_FRAME_HEIGHT
            ):
                print(
                    "Scaling camera intrinsics from "
                    f"{CALIBRATED_FRAME_WIDTH}x{CALIBRATED_FRAME_HEIGHT} "
                    f"to {frame_width}x{frame_height}."
                )
            printed_frame_info = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTag markers
        corners, ids, rejected = detect_apriltag_markers(gray)

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
                    K_frame,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE
                )

                if not success:
                    continue

                # Draw coordinate axes on marker
                cv2.drawFrameAxes(
                    frame,
                    K_frame,
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

                # Convert marker orientation from camera frame to robot base frame
                R_cam_marker, _ = cv2.Rodrigues(rvec)
                R_base_marker = T_base_cam[:3, :3] @ R_cam_marker

                q_current = rotation_matrix_to_quaternion(R_base_marker).astype(np.float32)

                state = orientation_state.setdefault(
                    tag_id,
                    {"q_smoothed": None, "euler_deg": None}
                )

                # Maintain SLERP-based smoothed quaternion for live display
                if state["q_smoothed"] is None:
                    q_smoothed = q_current
                else:
                    if np.dot(state["q_smoothed"], q_current) < 0.0:
                        q_current = -q_current
                    q_smoothed = quaternion_slerp(state["q_smoothed"], q_current, slerp_alpha)

                state["q_smoothed"] = q_smoothed

                R_smoothed = quaternion_to_rotation_matrix(q_smoothed)
                euler_rad = rotation_matrix_to_euler_xyz(R_smoothed)
                euler_deg = np.degrees(euler_rad)

                # unwrap angles relative to previous display to avoid +/-180 discontinuities
                prev_angles = state["euler_deg"]
                if prev_angles is not None:
                    for k in range(3):
                        delta = euler_deg[k] - prev_angles[k]
                        if delta > 180.0:
                            euler_deg[k] -= 360.0
                        elif delta < -180.0:
                            euler_deg[k] += 360.0
                    euler_deg = prev_angles + angle_alpha * (euler_deg - prev_angles)

                state["euler_deg"] = euler_deg

                disp_pos = p_base[:3]
                disp_euler = euler_deg

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
                    f"Base: ({disp_pos[0]:.3f}, {disp_pos[1]:.3f}, {disp_pos[2]:.3f})",
                    (u - 120, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"RPY: ({disp_euler[0]:.1f}, {disp_euler[1]:.1f}, {disp_euler[2]:.1f})",
                    (u - 120, v + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("AprilTag Detection", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    # ============================================================
    # 7. Clean up
    # ============================================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
