#!/usr/bin/env python3
"""Calibrate the camera, register it to the robot base, then position AprilTags.

Run with --help for options. Distances are meters; displayed RPY is degrees.
The camera must stay fixed between the reference image and positioning.
The reference checkerboard origin/axes must match --base-origin-cm and
BASE_CHECKER_ROTATION below (origin is the first detected inner corner).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKERBOARD = (7, 8)  # Number of inner corners, not squares
SQUARE_SIZE = 0.025
BASE_CHECKER_ROTATION = np.diag([1.0, -1.0, -1.0])
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def checker_points(square_size):
    points = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    points[:, :2] = np.mgrid[:CHECKERBOARD[0], :CHECKERBOARD[1]].T.reshape(-1, 2)
    return points * square_size


def read_checker(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return gray.shape[::-1], corners if found else None


def scaled_matrix(matrix, original_size, frame_size):
    """Scale intrinsics for resized images (requires the same camera field of view)."""
    scale = np.diag([frame_size[0] / original_size[0],
                     frame_size[1] / original_size[1], 1.0])
    return scale @ matrix


def calibrate(args):
    points = checker_points(args.square_size)
    objects, pixels = [], []
    image_size = None
    paths = sorted(args.images.glob('*.jpg'))
    # The fixed-board reference is separate from the intrinsic calibration views.
    paths = [p for p in paths if p.resolve() != args.reference.resolve()]
    if not paths:
        raise ValueError(f"No calibration JPG images found in {args.images}")
    print('Step 1: calibrating camera intrinsics')
    for path in paths:
        size, corners = read_checker(path)
        if corners is None:
            print(f'  Skipping {path.name}: checkerboard not found')
            continue
        if image_size is not None and size != image_size:
            raise ValueError(f'{path}: image size {size} differs from {image_size}. '
                             'Use calibration images from the same camera/resolution.')
        image_size = size
        objects.append(points.copy())
        pixels.append(corners)
        print(f'  Detected checkerboard: {path.name}')
    if len(objects) < 3:
        raise ValueError('At least 3 usable checkerboard views are required; '
                         'use more views with varied angles for a reliable calibration.')
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        objects, pixels, image_size, None, None,
    )
    print(f'Intrinsic RMS reprojection error: {rms:.4f} pixels')
    print(f'Camera intrinsic matrix (pixels):\n{matrix}')
    print(f'Distortion coefficients [k1, k2, p1, p2, k3]:\n{distortion}')

    print('Step 2: estimating checkerboard pose in the camera frame')
    reference_size, corners = read_checker(args.reference)
    if corners is None:
        raise ValueError(f'Checkerboard not found in {args.reference}')
    reference_matrix = scaled_matrix(matrix, image_size, reference_size)
    success, rvec, tvec = cv2.solvePnP(points, corners, reference_matrix, distortion)
    if not success:
        raise RuntimeError('Could not estimate reference checkerboard pose')
    camera_checker = np.eye(4)
    camera_checker[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_checker[:3, 3] = tvec.ravel()

    print('Step 3: computing camera-to-robot-base transform')
    base_checker = np.eye(4)
    base_checker[:3, :3] = BASE_CHECKER_ROTATION
    base_checker[:3, 3] = np.asarray(args.base_origin_cm) / 100.0
    base_camera = base_checker @ np.linalg.inv(camera_checker)
    result = dict(camera_matrix=matrix, dist_coeffs=distortion,
                  image_size=np.array(image_size), T_base_cam=base_camera,
                  T_cam_checker=camera_checker, T_base_checker=base_checker,
                  rms=np.array(rms))
    np.savez(args.output, **result)
    print(f'T_base_cam:\n{base_camera}')
    print(f'Saved calibration to {args.output}')
    return result


def euler_degrees(rotation):
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, np.arctan2(-rotation[2, 0], sy), yaw])


def position_tags(data, camera_index, marker_size):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('AprilTag detection requires OpenCV with the aruco module')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        detect = detector.detectMarkers
    else:
        params = cv2.aruco.DetectorParameters_create()
        def detect(gray):
            return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    half = marker_size / 2
    points = np.array([[-half, half, 0], [half, half, 0],
                       [half, -half, 0], [-half, -half, 0]], np.float32)
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {camera_index}')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(data['image_size'][0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(data['image_size'][1]))
        print('Step 4: AprilTag positioning. Press q to quit. Base: meters; RPY: degrees.')
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError('Failed to read camera frame')
            matrix = scaled_matrix(data['camera_matrix'], data['image_size'],
                                   frame.shape[1::-1])
            distortion = data['dist_coeffs']
            corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for tag_corners, tag_id in zip(corners, ids.ravel()):
                    pixels = tag_corners.reshape(4, 2).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        points, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    if not ok:
                        continue
                    base_position = data['T_base_cam'] @ np.r_[tvec.ravel(), 1.0]
                    rotation = data['T_base_cam'][:3, :3] @ cv2.Rodrigues(rvec)[0]
                    angles = euler_degrees(rotation)
                    u, v = pixels.mean(axis=0).astype(int)
                    lines = [f'ID {tag_id}',
                             'Base m: ({:.3f}, {:.3f}, {:.3f})'.format(*base_position[:3]),
                             'RPY deg: ({:.1f}, {:.1f}, {:.1f})'.format(*angles)]
                    for line_number, line in enumerate(lines):
                        cv2.putText(frame, line, (max(0, u - 120), max(20, v - 35) + 25 * line_number),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Calibration Positioning', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=SCRIPT_DIR / 'images')
    parser.add_argument('--reference', type=Path, default=SCRIPT_DIR / 'images/test.jpg',
                        help='Image of checkerboard at its measured robot-base placement')
    parser.add_argument('--base-origin-cm', type=float, nargs=3, default=[41.6, 22.9, -3.81],
                        metavar=('X', 'Y', 'Z'), help='Reference checkerboard origin in robot base, cm')
    parser.add_argument('--square-size', type=float, default=SQUARE_SIZE, help='Checker square side, meters')
    parser.add_argument('--marker-size', type=float, default=0.0457, help='AprilTag side, meters')
    parser.add_argument('--camera', type=int, default=4)
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'calibration_positioning.npz')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate-only', action='store_true', help='Save calibration without opening camera')
    mode.add_argument('--position-only', action='store_true', help='Use saved calibration without recalibrating')
    args = parser.parse_args()
    if args.square_size <= 0 or args.marker_size <= 0:
        parser.error('Square and marker sizes must be positive')
    try:
        if args.position_only:
            with np.load(args.output, allow_pickle=False) as saved:
                data = {key: saved[key] for key in saved.files}
            print(f"Camera intrinsic matrix (pixels):\n{data['camera_matrix']}")
            print('Distortion coefficients [k1, k2, p1, p2, k3]:\n'
                  f"{data['dist_coeffs']}")
            print(f"T_base_cam:\n{data['T_base_cam']}")
        else:
            data = calibrate(args)
        if not args.calibrate_only:
            position_tags(data, args.camera, args.marker_size)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
