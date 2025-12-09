import os
import copy
import time
from collections import deque
from enum import Enum, IntEnum

import cv2
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO

# Local imports
from handtracker.module_SARTE import HandTracker
from handtracker_wilor.module_WILOR import HandTracker_wilor
from gestureclassifier.model_update import create_model

# ==============================================================================
# 1. Constants
# ==============================================================================

FINGER_JOINTS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

TIP_JOINTS = [4, 8, 12, 16, 20]
BASELINE_VARIANCE = None


# ==============================================================================
# 2. Object Tracker Class
# ==============================================================================

class ObjTracker:
    """
    Handles object detection using YOLO and filters results based on depth
    proximity to the hand's wrist depth.
    """

    def __init__(self, det_cooltime=10):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_obj_path = os.path.join(curr_dir, 'pretrained_models', 'yolo11m.pt')
        self.detector_obj = YOLO(yolo_obj_path)

        self.detector_obj.to(self.device)

        # Warm-up the model with a test image
        test_img_path = os.path.join(curr_dir, './handtracker_wilor/demo_img/test1.jpg')
        if os.path.exists(test_img_path):
            test_img = cv2.imread(test_img_path)
            test_img = cv2.resize(test_img, (640, 360))
            _ = self.detector_obj(test_img, verbose=False)

        self.det_cooltime = det_cooltime
        self.obj_cnt = 0
        self.flag_detected = False

    def _run_detection(self, img, depth_image_float, d_wrist):
        """Internal method to run detection and filter by depth."""
        # Create a mask for depth values within 0.1m of the wrist
        mask = (depth_image_float > 0) & (depth_image_float - d_wrist <= 0.1)
        mask = mask.astype(np.uint8) * 255

        # Run YOLO detection
        results = self.detector_obj(img, verbose=False)

        obj_bb_nearby = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Exclude 'person' class
                if label == 'person':
                    continue

                # Check if the bounding box center is within the depth mask
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Check boundaries and mask value
                if cy < mask.shape[0] and cx < mask.shape[1]:
                    if mask[int(cy), int(cx)] == False:
                        continue
                    obj_bb_nearby.append([x1, y1, x2, y2, label])

        return obj_bb_nearby

    def detect_objs(self, img, depth_image_float, d_wrist):
        """Runs detection only when the cooldown counter is exceeded."""
        self.obj_cnt += 1

        if self.obj_cnt > self.det_cooltime:
            self.flag_detected = True
            self.obj_cnt = 0
            return self._run_detection(img, depth_image_float, d_wrist)
        else:
            self.flag_detected = False
            return []

    def detect_objs_no_cnt(self, img, depth_image_float, d_wrist):
        """Runs detection immediately without cooldown check."""
        self.flag_detected = True
        return self._run_detection(img, depth_image_float, d_wrist)


# ==============================================================================
# 3. Gesture Classifier Class
# ==============================================================================

class GestureClassfier:
    def __init__(self, ckpt="./gestureclassifier/checkpoints/checkpoint.tar", seq_len=16, model_opt=1):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set number of features based on model option
        if model_opt == 0 or (2 <= model_opt < 6):
            num_feature = 78
            self.flag_partial = False
        else:
            num_feature = 60
            self.flag_partial = True

        # Initialize model
        self.model_gesture = create_model(num_features=num_feature, num_classes=15, model_opt=model_opt)

        # Load checkpoint
        checkpoint = torch.load(ckpt, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        # Handle 'module.' prefix from DataParallel
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict

        self.model_gesture.load_state_dict(new_state_dict)
        self.model_gesture = torch.nn.DataParallel(self.model_gesture).cuda()
        self.model_gesture.eval()

        # Default arguments
        self.seq_len = seq_len
        self.idx_to_class = {
            0: 'CClock_index', 1: 'CClock_thumb',
            2: 'Clock_index', 3: 'Clock_thumb',
            4: 'Down_index', 5: 'Down_thumb',
            6: 'Left_index', 7: 'Left_thumb',
            8: 'Natural',
            9: 'Right_index', 10: 'Right_thumb',
            11: 'Tap_index', 12: 'Tap_thumb',
            13: 'Up_index', 14: 'Up_thumb'
        }
        self.partial_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20]

    def run(self, input_data):
        # input_data: queue (self.seq_len, 63+15) -> ndarray (self.seq_len, 78)
        input_data = np.array(input_data).reshape(self.seq_len, -1)

        # Normalize and optionally extract partial hand features
        input_data = self._normalize(input_data)
        if self.flag_partial:
            input_data = self._extract_partialhand(input_data)

        # Convert to tensor
        input_tensor = torch.from_numpy(input_data).to(self.device).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model_gesture(input_tensor)

        pred = output.argmax(1).cpu().numpy()
        gesture = self.idx_to_class[pred[0]]

        return pred[0], gesture

    def _normalize(self, pts, norm_ratio_x=180.0, norm_ratio_y=180.0, norm_ratio_z=100.0):
        """Normalize a single sample based on the root pose."""
        pts = np.asarray(pts)
        pts_norm = np.zeros((pts.shape[0], pts.shape[1]))

        for frame_idx in range(pts.shape[0]):
            target_pose = pts[frame_idx, :63].reshape(21, 3)
            target_angle = pts[frame_idx, 63:]

            # Use the first frame's root pose for normalization
            if frame_idx == 0:
                root_pose = target_pose[0, :]

            norm_pose = target_pose - root_pose
            norm_pose[:, 0] = norm_pose[:, 0] / norm_ratio_x
            norm_pose[:, 1] = norm_pose[:, 1] / norm_ratio_y
            norm_pose[:, 2] = norm_pose[:, 2] / norm_ratio_z

            pts_norm[frame_idx, :63] = norm_pose.flatten()
            pts_norm[frame_idx, 63:] = target_angle / 180.0

        return pts_norm

    def _extract_partialhand(self, pts_norm):
        """Extract features for partial hand joints."""
        pts_norm = np.asarray(pts_norm)
        pts_norm_part = []
        for frame_idx in range(pts_norm.shape[0]):
            target_pose = pts_norm[frame_idx, :63].reshape(21, 3)
            target_angle = pts_norm[frame_idx, 63:]

            target_pose = target_pose[self.partial_idx, :]
            target_pose = target_pose.flatten()

            pts_ = np.concatenate((target_pose, target_angle), axis=0)
            pts_norm_part.append(pts_)

        return np.array(pts_norm_part)

    def _compute_ang_from_joint(self, joint):
        """Compute angles between joints (21, 3)."""
        # Define parent and child joint indices
        v1_indices = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        v2_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        v1 = joint[v1_indices, :]
        v2 = joint[v2_indices, :]
        v = v2 - v1

        # Normalize vectors
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Calculate angles using arccos of dot product
        # Indices for angle calculation pairs
        angle_pairs_v1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        angle_pairs_v2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

        dot_product = np.einsum('nt,nt->n', v[angle_pairs_v1, :], v[angle_pairs_v2, :])
        angle = np.arccos(dot_product)
        angle = np.degrees(angle)

        return angle


# ==============================================================================
# 4. Hand Tracker Classes
# ==============================================================================

class HandTracker_v2:
    def __init__(self):
        self.model_hand = HandTracker_wilor()

    def run(self, input_img):
        return self.model_hand.run(input_img)


class HandTracker:
    def __init__(self):
        self.track_hand = HandTracker()

    def run(self, input_img):
        result_hand = self.track_hand.Process_single_newroi(input_img)
        return result_hand