import cv2
import json
from matplotlib.pylab import sign
import mediapipe as mp
from pathlib import Path
import os
from video_file_manager import VideoManager
from PIL import Image
import numpy as np
import subprocess
from Validators.orientation_validator import OrientationChecker
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from copy import deepcopy


class KeypointExtractorV1:
    '''Extracts pose and hand keypoints from videos using MediaPipe

    atributes:
        mp_pose: MediaPipe pose solution
        mp_hands: MediaPipe hands solution
        pose: Initialized MediaPipe pose model
        hands: Initialized MediaPipe hands model
    methods:
        extract_pose: Extracts pose keypoints from a single frame
        extract_hand: Extracts hand keypoints from a single frame
        extract_metadata: Extracts video metadata
        extract_to_json: Extracts keypoints and metadata from a video and saves to json
        extract_all: Processes all videos in a directory and saves keypoints to json files
        rotate_frame: Rotates a video frame by a specified angle
    '''

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_pose(self, image_rgb):
        '''
        takes:
            image_rgb: A single video frame in RGB format
        returns:
            extracted pose keypoints as a list of dictionaries with keys:
                cluster_id: 2 for pose keypoints
                landmark_id: MediaPipe landmark index
                x, y, z: Normalized coordinates of the keypoint
                visibility: Confidence score of the keypoint detection
        '''
        pose_landmarks = []

        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return pose_landmarks

        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i in range(15):
                pose_landmarks.append({
                    "cluster_id": 2,
                    "landmark_id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        return pose_landmarks

    def extract_hand(self, image_rgb, frame_index):
        '''
        takes:
            image_rgb: A single video frame in RGB format
            frame_index: current
        returns:
            extracted hand keypoints as a dictionary with keys "left" and "right", each containing a list of dictionaries with keys:
                cluster_id: 0 for left hand, 1 for right hand
                landmark_id: MediaPipe landmark index
                x, y, z: Normalized coordinates of the keypoint
            anomalous_hands: any times where ands had more then 21 points
        '''
        hands_data = {"left": [], "right": []}
        anomalous_hands = []

        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return hands_data, anomalous_hands

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            label = handedness.classification[0].label.lower()
            cluster_id = 0 if label == "left" else 1

            if len(hand_landmarks.landmark) > 21:
                anomalous_hands.append((frame_index, label))

            for idx, lm in enumerate(hand_landmarks.landmark):
                hands_data[label].append({
                    "cluster_id": cluster_id,
                    "landmark_id": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

        return hands_data, anomalous_hands

    def extract_metadata(self, cap, video_id=None, sign=None):
        '''
        takes:
            cap: OpenCV video capture object
            video_id: Optional video identifier extracted from filename
            sign: Optional sign label extracted from filename
        returns:
            metadata dictionary
        '''
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            "video_id": video_id,
            "sign": sign,
            "resolution": [frame_width, frame_height],
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "mediapipe_version": mp.__version__
        }
        return metadata

    def extract_to_json(self, filename, filepath):
        '''
        takes:
            filename: Path to the input video file
            filepath: Path to save the output JSON file
        returns:
            json dump of extracted keypoints and metadata'''
        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {filename}")

        # Detect rotation
        rotation_checker = OrientationChecker()
        rotation_angle = rotation_checker.get_rotation_metadata(filename)

        # assums sign corisponding to folder name, and video id corresponding
        # to file name
        sign = Path(filename).parent.name
        video_id = Path(filename).stem

        metadata = self.extract_metadata(cap, video_id=video_id, sign=sign)
        metadata["rotation_applied"] = rotation_angle
        frames = []

        frame_index = 0
        fps = metadata["fps"]

        anomalous_hands = []
        ret, frame = cap.read()
        while ret:
            # Apply rotation BEFORE converting to RGB
            if rotation_angle != 0:
                frame = self.rotate_frame(frame, rotation_angle)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_keypoints = self.extract_pose(image_rgb)
            hand_keypoints, maltiple_hands = self.extract_hand(
                image_rgb, frame_index)
            if len(maltiple_hands) > 0:
                # concatenate anomalous hands to main list with frame index for
                # later handling
                anomalous_hands.extend(maltiple_hands)

            frames.append({
                "frame_index": frame_index,
                "timestamp": frame_index / fps if fps else None,
                "pose": pose_keypoints,
                "hands": hand_keypoints
            })

            frame_index += 1
            ret, frame = cap.read()

        cap.release()

        output = {
            "metadata": metadata,
            "frames": frames
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        if len(anomalous_hands) > 0:
            self.handle_simultaneous_hands(anomalous_hands, filepath)

        print(f"  Keypoints saved to {filepath}")

        return output

    def handle_simultaneous_hands(self, anomalous_hands, filepath):
        '''takes cases where maltiples sets of keypoints in a single instance of a hand
        removes them from the main dataset and stores them within metadata for later use in the insertion validator.
        takes:
            anomalous_hands: all points where this happens, held as
            filepath: path to the json file where the current version is
                to be updated with corrected keypoints

        '''
        # open json file and load data
        with open(filepath, 'r') as f:
            data = json.load(f)

        for indx, side in anomalous_hands:
            # copys full frame of data into array of anomalous frames in
            # metadata
            data['metadata'].setdefault(
                'simultaneous_hands_frames', []).append(
                deepcopy(
                    data['frames'][indx]))
            # removes hand keypoints from main dataset
            data['frames'][indx]['hands'][side] = []
        # saves updated json file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def extract_all(self, directory, output_directory):
        '''
        takes:
            directory: Path to the input directory containing videos
            output_directory: Path to the output directory for JSON files
        '''
        video_manager = VideoManager()

        for root, dirs, files in os.walk(directory):

            if 'valid' in dirs:
                dirs[:] = ['valid']
            for file in files:
                if file.endswith('.mp4.encrypted'):
                    parts = file.split('_')

                    if len(parts) > 1:
                        p = parts[-1].split('.')
                        video_id = parts[1]
                        sign = p[0]
                    # for manually recorded videos without standard naming
                    else:
                        p = file.split('.')
                        # cuts last character of first part
                        video_id = 'manual-' + p[0][-1]
                        sign = p[0][:-1]

                    # decripts video
                    video_path = os.path.join(root, file)
                    decrypted_path = video_manager.decrypt_file(
                        video_path, video_manager.get_encryption_key())

                    sign_directory = Path(output_directory) / sign
                    sign_directory.mkdir(parents=True, exist_ok=True)
                    output_path = sign_directory / f"{video_id}.json"
                    self.extract_to_json(decrypted_path, output_path)

                    # reincrypts video
                    video_manager.encrypt_file(
                        decrypted_path, video_manager.get_encryption_key())

    def rotate_frame(self, frame, angle):
        """Rotate frame by given angle.
        takes:
            frame: Video frame to rotate
            angle: Rotation angle in degrees (90, 180, 270)
        returns:
            Rotated video frame
        """
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

# keypoint_extractor.py  –  fixed original (MediaPipe legacy)


class KeyPointExtractorV2:
    """Extracts hand keypoints from videos using MediaPipe (legacy API).

    Attributes:
        mp_hands: MediaPipe hands solution namespace
        hands:    Initialized MediaPipe Hands model
    Methods:
        extract_hand:              Extracts hand keypoints from a single RGB frame
        extract_metadata:          Extracts video metadata from an OpenCV capture
        extract_to_json:           Processes one video and writes keypoints to JSON
        handle_simultaneous_hands: Quarantines frames with duplicate hand labels
        extract_all:               Walks a directory and processes every video
        rotate_frame:              Rotates a frame by 90 / 180 / 270 degrees
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_hand(self, image_rgb, frame_index):
        """Extract hand keypoints from a single RGB frame.

        Args:
            image_rgb:   Frame in RGB format (numpy array).
            frame_index: Index of the current frame (used for anomaly logging).

        Returns:
            hands_data (dict):      {"left": [...], "right": [...]} each entry is a
                                    list of landmark dicts with keys cluster_id,
                                    landmark_id, x, y, z.
            anomalous_hands (list): (frame_index, label) tuples where the same hand
                                    label appeared more than once in a single frame.
        """
        hands_data = {"left": [], "right": []}
        anomalous_hands = []

        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return hands_data, anomalous_hands

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = handedness.classification[0].label.lower()
            cluster_id = 0 if label == "left" else 1

            # FIX: MediaPipe always returns exactly 21 landmarks, so
            # `len(...) > 21` never fires.  The real anomaly is the same
            # hand label appearing twice in one frame.
            if hands_data[label]:
                anomalous_hands.append((frame_index, label))
                continue  # discard the duplicate detection

            for idx, lm in enumerate(hand_landmarks.landmark):
                hands_data[label].append({
                    "cluster_id": cluster_id,
                    "landmark_id": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                })

        return hands_data, anomalous_hands

    def extract_metadata(self, cap, video_id=None, sign=None):
        """Build a metadata dict from an OpenCV VideoCapture object.

        Args:
            cap:      OpenCV VideoCapture (already opened).
            video_id: Optional identifier string.
            sign:     Optional sign label string.

        Returns:
            dict of video metadata.
        """
        return {
            "video_id": video_id,
            "sign": sign,
            "resolution": [
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ],
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "mediapipe_version": mp.__version__,
        }

    def extract_to_json(self, filename, filepath):
        """Process one video file and write keypoints to a JSON file.

        Args:
            filename: Path to the input video file.
            filepath: Destination path for the output JSON file.

        Returns:
            The full output dict that was written to JSON.
        """
        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {filename}")

        rotation_angle = OrientationChecker().getRotationMetadata(filename)

        sign = Path(filename).parent.name
        video_id = Path(filename).stem

        metadata = self.extract_metadata(cap, video_id=video_id, sign=sign)
        metadata["rotation_applied"] = rotation_angle

        frames = []
        anomalous_hands = []
        fps = metadata["fps"]
        frame_index = 0

        ret, frame = cap.read()
        while ret:
            if rotation_angle != 0:
                frame = self.rotate_frame(frame, rotation_angle)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_keypoints, multiple_hands = self.extract_hand(
                image_rgb, frame_index)

            if multiple_hands:
                anomalous_hands.extend(multiple_hands)

            frames.append({
                "frame_index": frame_index,
                "timestamp": frame_index / fps if fps else None,
                "hands": hand_keypoints,
            })

            frame_index += 1
            ret, frame = cap.read()

        cap.release()

        output = {"metadata": metadata, "frames": frames}

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        if anomalous_hands:
            self.handle_simultaneous_hands(anomalous_hands, filepath)

        print(f"  Keypoints saved to {filepath}")
        return output

    def handle_simultaneous_hands(self, anomalous_hands, filepath):
        """Quarantine frames where the same hand label was detected twice.

        Copies the full frame into metadata['simultaneous_hands_frames'] and
        clears the offending hand's keypoints from the main frame list.

        Args:
            anomalous_hands: List of (frame_index, label) tuples.
            filepath:        Path to the JSON file to update in-place.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        for idx, side in anomalous_hands:
            data["metadata"].setdefault(
                "simultaneous_hands_frames", []).append(
                deepcopy(
                    data["frames"][idx]))
            data["frames"][idx]["hands"][side] = []

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def extract_all(self, directory, output_directory):
        """Walk *directory* and extract keypoints from every encrypted video.

        Args:
            directory:        Root directory containing videos.
            output_directory: Root directory for output JSON files.
        """
        video_manager = VideoManager()

        for root, dirs, files in os.walk(directory):
            if "valid" in dirs:
                dirs[:] = ["valid"]

            for file in files:
                if not file.endswith(".mp4.encrypted"):
                    continue

                parts = file.split("_")
                if len(parts) > 1:
                    video_id = parts[1]
                    sign = parts[-1].split(".")[0]
                else:
                    p = file.split(".")
                    video_id = "manual-" + p[0][-1]
                    sign = p[0][:-1]

                video_path = os.path.join(root, file)
                key = video_manager.get_encryption_key()
                decrypted_path = video_manager.decrypt_file(video_path, key)

                sign_dir = Path(output_directory) / sign
                sign_dir.mkdir(parents=True, exist_ok=True)
                output_path = sign_dir / f"{video_id}.json"

                self.extract_to_json(decrypted_path, output_path)
                video_manager.encrypt_file(decrypted_path, key)

    def rotate_frame(self, frame, angle):
        """Rotate *frame* by *angle* degrees (must be 90, 180, or 270).

        Args:
            frame: OpenCV BGR frame.
            angle: Rotation angle in degrees.

        Returns:
            Rotated frame, or the original frame if angle is unrecognised.
        """
        rotations = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        code = rotations.get(angle)
        return cv2.rotate(frame, code) if code is not None else frame
