import cv2
import numpy as np


# ============================================================
# 1. Camera intrinsic parameters
# ============================================================

K = np.array([
    [1.51339478e+03, 0.00000000e+00, 9.92839853e+02],
    [0.00000000e+00, 1.50238718e+03, 4.88597228e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist = np.array([
    [-5.51964568e-03, 6.93801723e-01, 3.26991913e-04, 4.77776670e-03, -6.15503232e+00]
], dtype=np.float32)


# ============================================================
# 2. Camera frame -> robot base frame
# Your latest T_base_cam
# ============================================================

T_base_cam = np.array([
    [-0.74779584, 0.48554782, -0.45281861, 0.66213106],
    [0.65192636, 0.66609337, -0.36236947, 0.36685141],
    [0.12567177, -0.56618278, -0.81464328, 0.62996175],
    [0.0, 0.0, 0.0, 1.0]
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


def average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    """Average a list of quaternions with sign consistency."""
    if not quaternions:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    q_avg = quaternions[0].astype(np.float32).copy()

    for q in quaternions[1:]:
        if np.dot(q_avg, q) < 0.0:
            q = -q
        q_avg += q.astype(np.float32)

    q_avg /= len(quaternions)
    q_avg /= np.linalg.norm(q_avg)
    return q_avg


# ============================================================
# 3. AprilTag marker settings
# ============================================================

MARKER_SIZE = 0.0457 # meters

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

# Create detector with compatibility for different OpenCV versions
try:
    aruco_params = cv2.aruco.DetectorParameters()
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()

if hasattr(cv2.aruco, 'ArucoDetector'):
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
else:
    detector = None


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


def main():
    # ============================================================
    # 5. Open camera
    #
    # Change the camera index if needed:
    # 0 = default laptop camera
    # 1 or 2 = possible external/robot camera
    # ============================================================

    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    use_image_fallback = False

    if not cap.isOpened():
        # try default camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Could not open camera index {camera_index} or 0. Falling back to static image if available.")
        use_image_fallback = True

    # Match the camera frame size to the calibrated image resolution implied by the camera matrix.
    target_width = int(round(K[0, 2] * 2.0))
    target_height = int(round(K[1, 2] * 2.0))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    # ============================================================
    # 6. Main loop
    # ============================================================

    # Per-tag orientation state: stores smoothed quaternion and last displayed euler angles
    orientation_state = {}
    slerp_alpha = 0.3  # how much of the new orientation to blend in (0..1)
    angle_alpha = 0.3  # smoothing for displayed Euler angles

    while True:
        if use_image_fallback:
            # look for a sample image in images/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            images_dir = os.path.join(script_dir, "images")
            jpgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
            if not jpgs:
                print("No images found for fallback processing.")
                break
            frame = cv2.imread(jpgs[0])
            ret = frame is not None
            # process single image then exit loop
            single_image_mode = True
        else:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTag markers (support both detector APIs)
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                tag_id = int(np.atleast_1d(ids[i]).flat[0])

                # 2D pixel coordinates of this marker's corners
                marker_2d_points = corners[i][0].astype(np.float32)

                # Estimate marker pose relative to camera
                success, rvec, tvec = cv2.solvePnP(
                    marker_3d_points,
                    marker_2d_points,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE
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

                # Convert marker orientation from camera frame to robot base frame
                R_cam_marker, _ = cv2.Rodrigues(rvec)
                R_base_marker = T_base_cam[:3, :3] @ R_cam_marker

                q_current = rotation_matrix_to_quaternion(R_base_marker).astype(np.float32)

                pos_current = p_base[:3].astype(np.float32)

                state = orientation_state.setdefault(tag_id, {"q_smoothed": None, "euler_deg": None, "pos_buffer": [], "q_buffer": [], "averaged_printed": False})

                # Update rolling buffers for averaging (10 samples)
                pos_buf = state["pos_buffer"]
                q_buf = state["q_buffer"]

                # keep quaternion sign consistent with last stored to avoid flips
                if len(q_buf) > 0 and np.dot(q_buf[-1], q_current) < 0.0:
                    q_to_store = -q_current
                else:
                    q_to_store = q_current

                pos_buf.append(pos_current)
                q_buf.append(q_to_store)
                if len(pos_buf) > 10:
                    pos_buf.pop(0)
                    q_buf.pop(0)

                averaged_ready = len(pos_buf) == 10

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

                # When buffer is full, compute averaged result. Print it only once per tag.
                if averaged_ready:
                    pos_mean = np.mean(pos_buf, axis=0)
                    q_avg = average_quaternions(q_buf)
                    R_avg = quaternion_to_rotation_matrix(q_avg)
                    euler_avg = np.degrees(rotation_matrix_to_euler_xyz(R_avg))

                    if not state.get("averaged_printed", False):
                        print(f"Tag ID: {tag_id} (averaged over 10 frames)")
                        print(f"Averaged position in robot base frame: {pos_mean}")
                        print(f"Averaged orientation (roll, pitch, yaw deg): {euler_avg}")
                        print("--------------------------------")
                        state["averaged_printed"] = True

                    disp_pos = pos_mean
                    disp_euler = euler_avg
                else:
                    # Not enough samples yet; do not spam the console. Show live values on overlay only.
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

        # show result (will open window in interactive use)
        try:
            cv2.imshow("AprilTag Detection", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        except Exception:
            # In headless environments, save an output image and exit
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'apriltag_output.jpg')
            cv2.imwrite(out_path, frame)
            print(f"Saved output to {out_path}")
            if use_image_fallback:
                break

    # ============================================================
    # 7. Clean up
    # ============================================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
