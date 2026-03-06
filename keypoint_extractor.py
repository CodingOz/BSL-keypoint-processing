import cv2
import json
from matplotlib.pylab import sign
import mediapipe as mp
from pathlib import Path
import os
from video_file_manager import VideoManager
import subprocess
from PIL import Image
import numpy as np
from Validators.orientation_validator import orientation_checker
from keypoint_validator import CubicSplineKeyPointInterpolator

class KeyPointExtractor:
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
        rotation_checker = orientation_checker()
        
        rotation_angle = rotation_checker.get_rotation_metadata(filename)
        print(f"  Detected rotation: {rotation_angle}°")
        
        metadata = self.extract_metadata(cap)
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
            hand_keypoints, maltiple_hands = self.extract_hand(image_rgb, frame_index)
            anomalous_hands.append(maltiple_hands)
            frames.append({
                "frame_index": frame_index,
                "timestamp": frame_index / fps if fps else None,
                "pose": pose_keypoints,
                "hands": hand_keypoints
            })

            frame_index += 1
            ret, frame = cap.read()

        cap.release()
        
        if len(anomalous_hands) > 0:
            self.handle_anomalous_hands(frames, anomalous_hands, filepath)

        output = {
            "metadata": metadata,
            "frames": frames
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Keypoints saved to {filepath}")
        
        return output
    
    def handle_simultaneous_hands(self, frames, anomalous_hands, filepath):
        '''Handles cases where maltiples sets of keypoints in a single instance of a hand
        takes: 
            frames: all extracted frames
            anomalous_hands: all points where this happens
        '''
        validator = CubicSplineKeyPointInterpolator()
        missing_hands = CubicSplineKeyPointInterpolator.checkForMissingHands()
        for indx, side in anomalous_hands:
            # split sets of keypoints into seperate sets
            swapped = (indx, 'right' if side == 'left' else 'left')
            if swapped in missing_hands:
                pass
            # splits  
            
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
                    decrypted_path = video_manager.decrypt_file(video_path, video_manager.get_encryption_key())
                    
                    sign_directory = Path(output_directory) / sign
                    sign_directory.mkdir(parents=True, exist_ok=True)
                    output_path = sign_directory / f"{video_id}.json"
                    self.extract_to_json(decrypted_path, output_path)
                    
                    # reincrypts video
                    video_manager.encrypt_file(decrypted_path, video_manager.get_encryption_key())
       
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
      
Extractor = KeyPointExtractor()


Extractor.extract_all(
    directory=r"C:/Users/Oscar Strong/Desktop/finalProgect/videoCorpus_sorted",
    output_directory=r"C:/Users/Oscar Strong/Desktop/finalProgect/KeypointCorpus_unprocessed"
)




