import json
import math
import traceback
import os
import numpy as np
from scipy.signal import medfilt


import sys
from pathlib import Path
# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.validation_helpers import *
try:
    from Kalman_filter import HandPositionKalmanFilter
except ImportError:
    # Handle import when run as a module from parent directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Kalman_filter import HandPositionKalmanFilter 
from scipy.interpolate import CubicSpline, interp1d
from dataclasses import dataclass

@ dataclass
class Sigh_lenths:
    filepath: str
    
    # total frames 
    total_frames: int
    
    # first frame with one hands 
    first_hand: int
    
    # first frame where both hands have been seen 
    first_with_both: int
    
    # frame where both hands have been seen 
    # aka the last frame of the hand that first disapears 
    last_with_both: int
    
    # last frame of the last hand visible
    last_hand: int
    
    # number of frames in there area where both hands are seen
    length_with_both: int
    
    # nunber of frames between the first and last point a band was seen
    lenth_with_hands: int
    
    
class KeyPointValidator:
    def __init__(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{filepath}': {e}")
            return 1
        
        # private attributes to store computed values for reuse
        self.__palms = None
        self.__momentums = None
        self.__accelerations = None
        
        self.filepath = filepath
        
        self.frames = self.data.get('frames', [])

    def checkForSimultaneousHands(self, show_logs = False):
        found_error = False
        
        errors = []

        for frame in self.frames:
            frame_index = frame.get('frame_index')
            timestamp = frame.get('timestamp')
            hands = frame.get('hands', {})

            for side in ('left', 'right'):
                landmarks = hands.get(side, []) if isinstance(hands, dict) else []
                
                # Count occurrences of each (cluster_id, landmark_id) pair
                pair_counts = {}
                for lm in landmarks:
                    if not isinstance(lm, dict):
                        continue
                    cluster_id = lm.get('cluster_id')
                    landmark_id = lm.get('landmark_id')
                    if cluster_id is not None and landmark_id is not None:
                        pair = (cluster_id, landmark_id)
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1

                # Check if any pair appears more than once
                duplicate_pairs = {pair: count for pair, count in pair_counts.items() if count > 1}
                if duplicate_pairs:
                    found_error = True
                    if show_logs:
                        print(
                            f"Simultaneous {side} hands detected in frame {frame_index} "
                            f"(timestamp={timestamp}): duplicate pairs={duplicate_pairs}"
                        )

                    errors.append((frame_index, side))
        return errors
    
    def checkForMissingHands(self, show_logs=False):
        found_error = False
        
        errors = []

        for frame in self.frames:
            frame_index = frame.get('frame_index')
            timestamp = frame.get('timestamp')
            hands = frame.get('hands', {})

            for side in ('left', 'right'):
                landmarks = hands.get(side, []) if isinstance(hands, dict) else []
                if not landmarks:
                    found_error = True
                    if show_logs:
                        print(f"missing {side} hand detected in frame {frame_index} (timestamp={timestamp})")
                    errors.append((frame_index, side))
        return errors
    
    def viewFrame(self, filepath, frame_index):

        
        if frame_index < 0 or frame_index >= len(self.frames):
            print(f"Frame index {frame_index} out of range. Total frames: {len(self.frames)}")
            return
        
        frame = self.frames[frame_index]
        print(json.dumps(frame, indent=2))
        
   # ------ setter functions for palm centers, momentum and acceleration --------
    
    def findPalmCenters(self, frameIdx, show_logs=False):
        # computes the average position of the palm points 
        # (landmark_id in [0, 1, 2, 5, 9, 13, 17]) for each hand
        # in the specified frame
        try:
            frame = self.frames[frameIdx]
            hands = frame.get('hands', {})
            palmCenters = {}
            
            for side in ('left', 'right'):
                landmarks = hands.get(side, []) if isinstance(hands, dict) else []
                palm_points = [lm for lm in landmarks if isinstance(lm, dict) and 
                            lm.get('landmark_id') in [0, 1, 2, 5, 9, 13, 17]]
                
                if palm_points:
                    avg_x = sum(lm['x'] for lm in palm_points) / len(palm_points)
                    avg_y = sum(lm['y'] for lm in palm_points) / len(palm_points)
                    palmCenters[side] = [avg_x, avg_y]
                    if show_logs:
                        print(f"Frame {frameIdx}, {side}: Found {len(palm_points)} palm points, center: [{avg_x:.3f}, {avg_y:.3f}]")
                else:
                    palmCenters[side] = [None, None]
                    if show_logs:
                        print(f"Frame {frameIdx}, {side}: No palm points found (landmarks: {len(landmarks)})")
            
            return palmCenters
            
        except Exception as e:
            print(f"ERROR in findPalmCenters({frameIdx}): {e}")
            traceback.print_exc()
            return {}

    def findAllPalmCenters(self, show_logs=False):
        self.__palms = []
        
        for i in range(len(self.frames)):
            self.palmCenters = self.findPalmCenters(i, show_logs=show_logs)
            self.__palms.append(self.palmCenters)
        
        return self.__palms
    
    def findMomentum(self, palms, palms2, show_logs=False):
        
        if show_logs:
            print(f"Frame palms 1: {palms}")
            print(f"Frame palms 2: {palms2}")
        
        # Left hand
        if palms2['left'] != [None, None] and palms['left'] != [None, None]:
            leftDx = palms2['left'][0] - palms['left'][0]  # X is index 0
            leftDy = palms2['left'][1] - palms['left'][1]  # Y is index 1
            leftDirX, leftDirY, leftMag = normalize_vector(leftDx, leftDy)
            estL = False 
        else:
            leftDirX, leftDirY, leftMag = 0, 0, 0
            estL = True
        
        # Right hand
        if palms2['right'] != [None, None] and palms['right'] != [None, None]:
            rightDx = palms2['right'][0] - palms['right'][0]
            rightDy = palms2['right'][1] - palms['right'][1]
            rightDirX, rightDirY, rightMag = normalize_vector(rightDx, rightDy)
            estR = False 
        else:
            estR = True 
            rightDirX, rightDirY, rightMag = 0, 0, 0
            
        return {
            'left': {
                'direction': (leftDirX, leftDirY),
                'magnitude': leftMag,
                'est': estL
            },
            'right': {
                'direction': (rightDirX, rightDirY),
                'magnitude': rightMag,
                'est': estR
            }
        }

    def storeMomentum(self, show_logs=False):
        # stores the momentum values in an array as a class atribute
        self.__momentums = []
        for i in range(len(self.frames)-1):
            if self.__palms is None:
                self.findAllPalmCenters(show_logs=show_logs)
            
            palms = self.__palms[i] if i < len(self.__palms) \
                else self.findPalmCenters(i)
            palms2 = self.__palms[i+1] if i+1 < len(self.__palms) \
                else self.findPalmCenters(i+1, show_logs=show_logs)
        
            momentum = self.findMomentum(palms, palms2, show_logs=show_logs)
            self.__momentums.append(momentum)
            
    def computeAcceleration(self, palms, palms2, palms3, show_logs=False): 
        """
        Compute acceleration with robustness against missing palm centers.
        Uses constant acceleration assumption to estimate missing momentum values.
        
        Args:
            frameIdx1, frameIdx2, frameIdx3: three frame indices (not necessarily consecutive)
            
        Returns:
            dict with keys:
                - 'acceleration': dict with acceleration metrics per hand
                - 'estimated': dict marking which values are estimated (True/False per hand per frame)
        """
                
        if not self.__momentums:
            self.storeMomentum()
        
        # Get momentums with estimation tracking
        
        momentum_1 = self.findMomentum(palms, palms2, show_logs=show_logs)
        momentum_2 = self.findMomentum(palms3, palms2, show_logs=show_logs)  # Use palms2 as second palm for momentum_2
        
        result = {}
        
        for side in ['left', 'right']:
            if momentum_1[side]['est']:
                result[side] = {
                    'acceleration': 0,
                    'acceleration_vector': (0, 0),
                    'angular_deviation': 0,
                    'angular_deviation_degrees': 0,
                    'magnitude_change': 0,
                    'est': True
                }
            else:
                v1_dir = momentum_1[side]['direction']
                v1Mag = momentum_1[side]['magnitude']
                
                v2_dir = momentum_2[side]['direction']
                v2Mag = momentum_2[side]['magnitude']
                
                # Acceleration: change in velocity
                accel_x = v2_dir[0] * v2Mag - v1_dir[0] * v1Mag
                accel_y = v2_dir[1] * v2Mag - v1_dir[1] * v1Mag
                acceleration = vector_magnitude((accel_x, accel_y))
                
                # Angular deviation: pure directional change
                angular_deviation = angle_between_vectors(v1_dir, v2_dir)
                
                # Magnitude change
                magnitude_change = v2Mag - v1Mag
                
                result[side] = {
                    'acceleration': acceleration,
                    'acceleration_vector': (accel_x, accel_y),
                    'angular_deviation': angular_deviation,
                    'angular_deviation_degrees': math.degrees(angular_deviation),
                    'magnitude_change': magnitude_change
                }
                
        return {
            'acceleration': result            
        }
        
    def storeAccelerations(self):
        self.__accelerations = []
        for i in range(len(self.frames)-2):
            if self.__palms is None:
                self.findAllPalmCenters()
            palms = self.__palms[i] if i < len(self.__palms) \
                else self.findPalmCenters(i)
            palms2 = self.__palms[i+1] if i+1 < len(self.__palms) \
                else self.findPalmCenters(i+1)
            palms3 = self.__palms[i+2] if i+2 < len(self.__palms) \
                else self.findPalmCenters(i+2)
            accel = self.computeAcceleration(palms, palms2, palms3)
            self.__accelerations.append(accel)
    
    # -------------------- getters --------------------
    
    def getAccelerations(self):
        if self.__accelerations is None:
            self.storeAccelerations()
        return self.__accelerations
    
    def getMomentums(self):
        if self.__momentums is None:
            self.storeMomentum()
        return self.__momentums
    
    def getPalmCenters(self):
        if self.__palms is None:
            self.findAllPalmCenters()
        return self.__palms

    def getDistance(self, palm1, palm2):
        lx1, ly1 = palm1['left'][0], palm1['left'][1]
        lx2, ly2 = palm2['left'][0], palm2['left'][1]
        
        rx1, ry1 = palm1['right'][0], palm1['right'][1]
        rx2, ry2 = palm2['right'][0], palm2['right'][1]
    
        left = math.hypot(lx2 - lx1, ly2 - ly1)
        right = math.hypot(rx2 - rx1, ry2 - ry1) 
        return [left, right]

    def getSignLenths(self):
        ''' return a Sigh_lenths object with filled values
        
        '''
        
        if self.__palms is None:
            self.findAllPalmCenters()
        
        total_frames = len(self.frames)
        
        # Initialize tracking variables
        first_hand = None
        first_with_both = None
        last_with_both = None
        last_hand = None
        
        # Track which hands are visible in each frame
        left_visible = []
        right_visible = []
        both_visible = []
        
        for i, palm in enumerate(self.__palms):
            left_present = (palm['left'] != [None, None] and 
                        None not in palm['left'])
            right_present = (palm['right'] != [None, None] and 
                            None not in palm['right'])
            
            left_visible.append(left_present)
            right_visible.append(right_present)
            both_visible.append(left_present and right_present)
            
            # First frame with at least one hand
            if first_hand is None and (left_present or right_present):
                first_hand = i
            
            # First frame with both hands
            if first_with_both is None and (left_present and right_present):
                first_with_both = i
            
            # Last frame with both hands (keep updating)
            if left_present and right_present:
                last_with_both = i
            
            # Last frame with at least one hand (keep updating)
            if left_present or right_present:
                last_hand = i
        
        # Calculate derived metrics
        if first_with_both is not None and last_with_both is not None:
            length_with_both = last_with_both - first_with_both + 1
        else:
            length_with_both = 0
        
        if first_hand is not None and last_hand is not None:
            length_with_hands = last_hand - first_hand + 1
        else:
            length_with_hands = 0
        
        # Handle cases where hands are never visible
        if first_hand is None:
            first_hand = 0
        if last_hand is None:
            last_hand = total_frames - 1
        if first_with_both is None:
            first_with_both = 0
        if last_with_both is None:
            last_with_both = 0
        
        return Sigh_lenths(
            filepath=self.filepath,
            total_frames=total_frames,
            first_hand=first_hand,
            first_with_both=first_with_both,
            last_with_both=last_with_both,
            last_hand=last_hand,
            length_with_both=length_with_both,
            lenth_with_hands=length_with_hands
        )
            
    def findNonMissingValueClusters(self):
        '''Using the missing frames to find the clussers with no empty space
        return:
            left_clusters: list of clusters as array of tuples
            right_clusters: list of clusters as array of tuples
            formats - [(startinds, end),(start, end)]
            (a point i means that the boundry is between i and i + 1)
        '''
        left_clusters = []
        right_clusters = []
        
        missing = self.checkForMissingHands()
        left_in_cluster = None if (0, 'left') in missing else 0
        right_in_cluster = None if (0, 'right') in missing else 0
        for i in range(len(self.frames)):
            left_missing = (i, 'left') in missing
            right_missing = (i, 'right') in missing
            
            if left_missing:
                if left_in_cluster != None:
                    left_clusters.append((left_in_cluster, i-1))
                    left_in_cluster = None
            else:
                if left_in_cluster == None:
                    left_in_cluster = i
                
            if right_missing:
                if right_in_cluster != None:
                    right_clusters.append((right_in_cluster, i-1))
                    right_in_cluster = None
            else: 
                if right_in_cluster == None:
                    right_in_cluster = i
        
        return left_clusters, right_clusters

    def findHandDistancesFromMidpoint(self, frameIdx, midpoint=0.5, show_logs=False) -> dict:
        """
        Returns how far each hand sits from the screen midpoint, signed by side.
        Left hand: positive = left of midpoint (expected), negative = crossed over right
        Right hand: positive = right of midpoint (expected), negative = crossed over left
        Takes:
            frameIdx:  frame to compute distances for
            midpoint:  x-coordinate treated as centre (default 0.5 for MediaPipe normalised)
        Returns:
            {
                'left':  float | None,   # midpoint - left_x
                'right': float | None,   # right_x  - midpoint
            }
        """
        palms = self.findPalmCenters(frameIdx, show_logs=show_logs)

        result = {'left': None, 'right': None}

        left = palms.get('left')
        if left and left != [None, None] and None not in left:
            result['left'] = round(midpoint - left[0], 6)

        right = palms.get('right')
        if right and right != [None, None] and None not in right:
            result['right'] = round(right[0] - midpoint, 6)

        if show_logs:
            print(f"Frame {frameIdx} | midpoint={midpoint} | "
                f"left dist={result['left']}  right dist={result['right']}")

        return result

    def findAllHandDistancesFromMidpoint(self, midpoint=0.5, show_logs=False) -> list[dict]:
        """
        Runs findHandDistancesFromMidpoint across every frame.

        Returns:
            list of dicts, one per frame, each {'left': float|None, 'right': float|None}
        """
        if self.__palms is None:
            self.findAllPalmCenters()

        frames = self.data.get('frames', [])
        return [
            self.findHandDistancesFromMidpoint(i, midpoint=midpoint, show_logs=show_logs)
            for i in range(len(frames))
        ]

    def findAdaptiveMidpoint(self, show_logs=False) -> float:
        """
        Computes the empirical horizontal centre of the signer by averaging
        the x-coordinates of ALL visible palm centers across ALL frames.

        This accounts for signers who are not perfectly centred in frame,
        or for cameras with a compositional offset.

        Returns:
            float: the mean palm x-coordinate across all frames and both hands
        """
        if self.__palms is None:
            self.findAllPalmCenters()

        all_x = []
        for palm in self.__palms:
            for side in ('left', 'right'):
                pos = palm.get(side)
                if pos and pos != [None, None] and None not in pos:
                    all_x.append(pos[0])

        if not all_x:
            if show_logs:
                print("No visible palms found — falling back to midpoint=0.5")
            return 0.5

        adaptive = float(np.mean(all_x))

        if show_logs:
            print(f"Adaptive midpoint: {adaptive:.4f}  (computed from {len(all_x)} palm observations)")

        return adaptive

    def findAllHandDistancesFromAdaptiveMidpoint(self, show_logs=False) -> tuple[list[dict], float]:
        """
        Computes the adaptive midpoint from palm data, then returns distances
        for every frame using that midpoint.

        Returns:
            (distances, adaptive_midpoint)
            distances: list of {'left': float|None, 'right': float|None}, one per frame
            adaptive_midpoint: the empirically derived centre used
        """
        midpoint = self.findAdaptiveMidpoint(show_logs=show_logs)
        distances = self.findAllHandDistancesFromMidpoint(midpoint=midpoint, show_logs=show_logs)
        return distances, midpoint

    def flagAbnormalDistances(self, outlier_boundry=-0.1):
        distances, midpoint = self.findAllHandDistancesFromAdaptiveMidpoint()
        left_flags = []
        right_flags = []
        for indx, distance in enumerate(distances):
            if distance.get("left"):
                if distance.get("left") < outlier_boundry:
                    left_flags.append(indx)
            if distance.get("right"):
                if distance.get("right") < outlier_boundry:
                    right_flags.append(indx)
        return left_flags, right_flags
    
    def getKeyPointsAsLists(self):
        """Returns a 3d list holding the x and y coordinates of all 41 landmarks for
        each hand in each frame, with None for missing hands or landmarks.
        """
        keypoint_data = []
        for frame in self.frames:
            frame_keypoints = {'left': None, 'right': None}
            hands = frame.get('hands', {})
            for side in ('left', 'right'):
                landmarks = hands.get(side, []) if isinstance(hands, dict) else []
                if landmarks:
                    # Create a list of 41 landmarks initialized to None
                    side_keypoints = [[None, None] for _ in range(41)]
                    for lm in landmarks:
                        if isinstance(lm, dict):
                            landmark_id = lm.get('landmark_id')
                            x = lm.get('x')
                            y = lm.get('y')
                            if landmark_id is not None and 0 <= landmark_id < 41:
                                side_keypoints[landmark_id] = [x, y]
                    frame_keypoints[side] = side_keypoints
            keypoint_data.append(frame_keypoints)
        return keypoint_data
        
    
class KalmanKeyPointEstimator(KeyPointValidator):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.__filled_palms = None
        self.__estimation_flags = None
        self.__fps = 25  # Store FPS
        self.__accelerationsEstimated = None
        self.__momentumsEstimated = None
        
    def estimateMissingPalmCenters(self, fps=25, show_logs=False):
        """
        Fill missing palm centers using Kalman filtering.
        
        Kalman filter must predict() all frames, then update() if measurement exists
        """
        filled_centers = []
        estimation_flags = []
        
        # Get all palm centers first
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters(show_logs=show_logs)
        
        # Create Kalman filters with tuned parameters
        kfLeft = HandPositionKalmanFilter(
            fps=fps,
            process_noise=0.1,      # Increased - hands move unpredictably
            measurement_noise=0.005  # Slight measurement noise from MediaPipe
        )
        kfRight = HandPositionKalmanFilter(
            fps=fps,
            process_noise=0.1,
            measurement_noise=0.005
        )
        
        for i in range(len(self.frames)):
            palmCenters = self._KeyPointValidator__palms[i]
            
            filled_frame = {'left': [None, None], 'right': [None, None]}
            estimated = {'left': False, 'right': False}
            
            # ===== LEFT HAND =====
            left_measurement = palmCenters.get('left')
            left_has_measurement = (left_measurement is not None and 
                                   left_measurement != [None, None])
            
            if not kfLeft.initialized:
                # Need first measurement to initialize
                if left_has_measurement:
                    position = kfLeft.update(left_measurement)
                    filled_frame['left'] = position.tolist()
                    estimated['left'] = False
                else:
                    filled_frame['left'] = [None, None]
                    estimated['left'] = True
            else:
                # Filter is initialized - predict first, then update if measurement exists
                predicted_position = kfLeft.predict()  # Always predict
                
                if left_has_measurement:
                    # Measurement available - update with it
                    updated_position = kfLeft.update(left_measurement)
                    filled_frame['left'] = updated_position.tolist()
                    estimated['left'] = False
                    
                    if show_logs and i % 10 == 0:  # Log every 10 frames
                        pred = predicted_position
                        meas = left_measurement
                        upd = updated_position
                        print(f"Frame {i} LEFT: pred=[{pred[0]:.3f},{pred[1]:.3f}] "
                              f"meas=[{meas[0]:.3f},{meas[1]:.3f}] "
                              f"upd=[{upd[0]:.3f},{upd[1]:.3f}]")
                else:
                    # No measurement - use prediction
                    filled_frame['left'] = predicted_position.tolist()
                    estimated['left'] = True
            
            right_measurement = palmCenters.get('right')
            right_has_measurement = (right_measurement is not None and 
                                    right_measurement != [None, None])
            
            if not kfRight.initialized:
                if right_has_measurement:
                    position = kfRight.update(right_measurement)
                    filled_frame['right'] = position.tolist()
                    estimated['right'] = False
                else:
                    filled_frame['right'] = [None, None]
                    estimated['right'] = True
            else:
                predicted_position = kfRight.predict()
                
                if right_has_measurement:
                    updated_position = kfRight.update(right_measurement)
                    filled_frame['right'] = updated_position.tolist()
                    estimated['right'] = False
                else:
                    filled_frame['right'] = predicted_position.tolist()
                    estimated['right'] = True
            
            filled_centers.append(filled_frame)
            estimation_flags.append(estimated)
        
        self.__filled_palms = filled_centers
        self.__estimation_flags = estimation_flags
        
        if show_logs:
            # Calculate statistics
            left_estimated = sum(1 for e in estimation_flags if e['left'])
            right_estimated = sum(1 for e in estimation_flags if e['right'])
            print(f"\nEstimation summary:")
            print(f"  Left hand: {left_estimated}/{len(self.frames)} frames estimated "
                  f"({100*left_estimated/len(self.frames):.1f}%)")
            print(f"  Right hand: {right_estimated}/{len(self.frames)} frames estimated "
                  f"({100*right_estimated/len(self.frames):.1f}%)")
        
        return filled_centers, estimation_flags
    
    def getFilledPalmCenters(self):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters()
        return self.__filled_palms
    
    def estimateMissingMomentums(self, show_logs=False):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters(show_logs=show_logs)
        
        # Recompute momentums using filled palm centers
        momentums = []
        for i in range(len(self.__filled_palms)-1):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            momentum = self.findMomentum(palms, palms2, show_logs=show_logs)
            momentums.append(momentum)
        
        return momentums
    
    def getEstimatedMomentums(self):
        if self.__momentumsEstimated is None:
            self.__momentumsEstimated = self.estimateMissingMomentums()
        return self.__momentumsEstimated
    
    def estimateMissingAccelerations(self, show_logs=False):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters(show_logs=show_logs)
        accelerations = []
        for i in range(len(self.__filled_palms)-2):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            palms3 = self.__filled_palms[i+2]
            accelerations.append(self.computeAcceleration(palms, 
                                                          palms2, 
                                                          palms3, 
                                                          show_logs=show_logs))
        
        return accelerations
    
    def getEstimatedAccelerations(self):
        if self.__accelerationsEstimated is None:
            self.__accelerationsEstimated = self.estimateMissingAccelerations()
        return self.__accelerationsEstimated
    
    
class CubicSplineKeyPointInterpolator(KeyPointValidator):
    def __init__(self, filepath, method='cubic'):
        super().__init__(filepath)
        self.__filled_palms = None
        self.__estimation_flags = None
        self.__accelerationsEstimated = None
        self.__momentumsEstimated = None
        self.__hand_distances = None
        self.methods = ['cubic', 'pchip', 'linear']
        if method in self.methods:
            self.method = method

    def interpolateSequence(self, positions):
        '''
        Interpolates a sequence of 2D positions (with None for missing) using the specified method.
        Returns: A list of interpolated positions and a list of flags indicating which positions were interpolated.
        '''
        # Separate into known and unknown
        frames = np.arange(len(positions))
        known_frames = []
        known_x = []
        known_y = []
        
        for i, pos in enumerate(positions):
            if pos is not None and pos != [None, None] and None not in pos:
                known_frames.append(i)
                known_x.append(pos[0])
                known_y.append(pos[1])
        
        if len(known_frames) < 2:
            # Not enough points to interpolate
            return positions, [True] * len(positions)
        
        known_frames = np.array(known_frames)
        known_x = np.array(known_x)
        known_y = np.array(known_y)
        
        # Create interpolators
        if self.method == 'cubic':
            # Cubic spline - smooth and passes through all points
            interp_x = CubicSpline(known_frames, known_x, bc_type='natural')
            interp_y = CubicSpline(known_frames, known_y, bc_type='natural')
        elif self.method == 'pchip':
            # PCHIP - shape-preserving, no overshoots
            from scipy.interpolate import PchipInterpolator
            interp_x = PchipInterpolator(known_frames, known_x)
            interp_y = PchipInterpolator(known_frames, known_y)
        else:  # linear
            interp_x = interp1d(known_frames, known_x, kind='linear', 
                               fill_value='extrapolate')
            interp_y = interp1d(known_frames, known_y, kind='linear',
                               fill_value='extrapolate')
        
        # Fill all positions
        filled_positions = []
        was_interpolated = []
        
        for i in range(len(positions)):
            if i in known_frames:
                # Known measurement - use original
                idx = np.where(known_frames == i)[0][0]
                filled_positions.append([known_x[idx], known_y[idx]])
                was_interpolated.append(False)
            elif known_frames[0] <= i <= known_frames[-1]:
                # Within interpolation range
                x_interp = float(interp_x(i))
                y_interp = float(interp_y(i))
                filled_positions.append([x_interp, y_interp])
                was_interpolated.append(True)
            else:
                # Outside range - can't interpolate
                filled_positions.append([None, None])
                was_interpolated.append(True)
        
        return filled_positions, was_interpolated

    def getFilledPalmCenters(self):
        if self.__filled_palms is not None:
            return self.__filled_palms, self.__estimation_flags
        
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters()
        
        filled_palms = []
        estimation_flags = []
        
        for side in ['left', 'right']:
            # Extract sequence for this hand
            positions = [p[side] for p in self._KeyPointValidator__palms]
            filled_positions, was_interpolated = self.interpolateSequence(positions)
            
            # Store results
            for i in range(len(filled_positions)):
                if len(filled_palms) <= i:
                    filled_palms.append({'left': [None, None], 'right': [None, None]})
                    estimation_flags.append({'left': False, 'right': False})
                
                filled_palms[i][side] = filled_positions[i]
                estimation_flags[i][side] = was_interpolated[i]
        
        self.__filled_palms = filled_palms
        self.__estimation_flags = estimation_flags
        
        return filled_palms, estimation_flags
    
    def estimateMissingMomentums(self, show_logs=False):
        if self.__filled_palms is None:
            self.getFilledPalmCenters()
        
        momentums = []
        for i in range(len(self.__filled_palms)-1):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            momentum = self.findMomentum(palms, palms2, show_logs=show_logs)
            momentums.append(momentum)
        
        return momentums
    
    def getEstimatedMomentums(self):
        if self.__momentumsEstimated is None:
            self.__momentumsEstimated = self.estimateMissingMomentums()
        return self.__momentumsEstimated
    
    def estimateMissingAccelerations(self, show_logs=False):
        if self.__filled_palms is None:
            self.getFilledPalmCenters()
        
        accelerations = []
        for i in range(len(self.__filled_palms)-2):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            palms3 = self.__filled_palms[i+2]
            accelerations.append(self.computeAcceleration(palms, 
                                                          palms2, 
                                                          palms3, 
                                                          show_logs=show_logs))
        return accelerations
    
    def getEstimatedAccelerations(self):
        if self.__accelerationsEstimated is None:
            self.__accelerationsEstimated = self.estimateMissingAccelerations()
        return self.__accelerationsEstimated
    
    def findMovmentClusters(self, max_momentum=0.15):
        '''Uses momentum spikes to find the boundrys between clusters
           of hand frames with resmables movement 
           This is a marker for poptencaly problimatice frames

        returns:
            left_boundrys: list of boundry points
            right_boundrys: list of boundry points
            (points  i and i + 1 are the boundry between clusters
            and are both recorded)
        '''
        momentums = self.getEstimatedMomentums()
        left_boundrys = []
        right_boundrys = []
        
        for i in range(len(momentums) - 1):
            # Check left hand momentum
            left_mag_curr = momentums[i]['left']['magnitude']
            
            if left_mag_curr > max_momentum:
                left_boundrys.append(i)
                left_boundrys.append(i+1)
            
            # Check right hand momentum
            right_mag_curr = momentums[i]['right']['magnitude']
            
            if right_mag_curr > max_momentum:
                right_boundrys.append(i)
                right_boundrys.append(i+1)

        
        return left_boundrys, right_boundrys
    
    def findMovementRelativeByMAD(self, threshold):
        """ instead of hard thresholding for the movement, constructs a Gaussian distribution 
        for each the files momentum, then basing the threshold on a set amount of 
        MADs above the median
        
        takes:
            threshold: float of the standard deviations above the mean for outliers
        
        returns:
            2 lists holding indexes that have spiked
        """
        momentums = self.getEstimatedMomentums()
        left_boundrys = []
        right_boundrys = []

        left_magnitudes = np.array([m['left']['magnitude'] for m in momentums])
        right_magnitudes = np.array([m['right']['magnitude'] for m in momentums])
        
        # strip out 0 values to get a more accurate picture of the movement distribution
        stripped_left_magnitudes = left_magnitudes[left_magnitudes > 0]
        stripped_right_magnitudes = right_magnitudes[right_magnitudes > 0]
        
        # Robust location and scale estimators
        left_median = np.median(stripped_left_magnitudes)
        left_mad = np.median(np.abs(stripped_left_magnitudes - left_median)) * 1.4826
        
        right_median = np.median(stripped_right_magnitudes)
        right_mad = np.median(np.abs(stripped_right_magnitudes - right_median)) * 1.4826

        left_threshold = left_median + threshold * left_mad
        right_threshold = right_median + threshold * right_mad

        for i in range(len(momentums) - 1):
            if momentums[i]['left']['magnitude'] > left_threshold:
                left_boundrys.append(i)
                left_boundrys.append(i + 1)

            if momentums[i]['right']['magnitude'] > right_threshold:
                right_boundrys.append(i)
                right_boundrys.append(i + 1)

        return left_boundrys, right_boundrys
        
    def findMovementRelativeByPercentile(self, percentile):
        ''' instead of hard thresholding for the movement, constructs a Gaussian distribution
        for each the files momentum, then basing the threshold on a set amount of
        percentiles above the mean'''
        momentums = self.getEstimatedMomentums()
        left_boundrys = []
        right_boundrys = []
        
        left_magnitudes = np.array([m['left']['magnitude'] for m in momentums])
        right_magnitudes = np.array([m['right']['magnitude'] for m in momentums])
        
        left_threshold = np.percentile(left_magnitudes, percentile)
        right_threshold = np.percentile(right_magnitudes, percentile)
        
        for i in range(len(momentums) - 1):
            if momentums[i]['left']['magnitude'] > left_threshold:
                left_boundrys.append(i)
                left_boundrys.append(i + 1)

            if momentums[i]['right']['magnitude'] > right_threshold:
                right_boundrys.append(i)
                right_boundrys.append(i + 1)
                
        return left_boundrys, right_boundrys
    
    def findMovementRelativeByStdDev(self, num_std_dev):
        ''' instead of hard thresholding for the movement, constructs a Gaussian distribution
        for each the files momentum, then basing the threshold on a set amount of
        standard deviations above the mean'''
        momentums = self.getEstimatedMomentums()
        left_boundrys = []
        right_boundrys = []

        left_magnitudes = np.array([m['left']['magnitude'] for m in momentums])
        right_magnitudes = np.array([m['right']['magnitude'] for m in momentums])

        left_mean = np.mean(left_magnitudes)
        left_std = np.std(left_magnitudes)
        left_threshold = left_mean + num_std_dev * left_std

        right_mean = np.mean(right_magnitudes)
        right_std = np.std(right_magnitudes)
        right_threshold = right_mean + num_std_dev * right_std

        for i in range(len(momentums) - 1):
            if momentums[i]['left']['magnitude'] > left_threshold:
                left_boundrys.append(i)
                left_boundrys.append(i + 1)

            if momentums[i]['right']['magnitude'] > right_threshold:
                right_boundrys.append(i)
                right_boundrys.append(i + 1)

        return left_boundrys, right_boundrys
        
    def interpolateFullHands(self, start_frame=0, end_frame=None, show_logs=False):
        keypoint_data = self.getKeyPointsAsLists()
        end_frame = end_frame if end_frame is not None else len(keypoint_data) - 1

        filled_keypoints = []
        estimation_flags = []

        for side in ['left', 'right']:
            landmark_sequences = [[] for _ in range(21)]  # 21 not 41
            for frame in keypoint_data:
                for lm_id in range(21):
                    pos = frame[side][lm_id] if frame[side] else [None, None]
                    landmark_sequences[lm_id].append(pos)

            filled_landmarks = []
            flags_landmarks  = []
            for lm_id in range(21):
                filled_positions, was_interpolated = self.interpolateSequence(
                    landmark_sequences[lm_id]
                )
                filled_landmarks.append(filled_positions)
                flags_landmarks.append(was_interpolated)

            for i in range(len(keypoint_data)):
                if len(filled_keypoints) <= i:
                    filled_keypoints.append({
                        'left':  [[None, None] for _ in range(21)],
                        'right': [[None, None] for _ in range(21)]
                    })
                    estimation_flags.append({
                        'left':  [False] * 21,
                        'right': [False] * 21
                    })

                for lm_id in range(21):
                    # only fill within the requested frame range
                    if start_frame <= i <= end_frame:
                        filled_keypoints[i][side][lm_id] = filled_landmarks[lm_id][i]
                        estimation_flags[i][side][lm_id] = flags_landmarks[lm_id][i]

        return filled_keypoints, estimation_flags

    def detectRestPositionFrames(self,
                                y_threshold_relative=0.15,
                                proximity_threshold=0.15,
                                min_run_length=3):
        """
        Scans inward from edges to find preparation/retraction frames.

        Uses a relative y threshold derived from the signing region rather
        than an absolute coordinate, so detection works regardless of how
        high or low the signer sits in frame.

        Frames where only one hand is visible are treated as inconclusive
        and skipped during the scan rather than treated as non-rest.
        
        takes:
            y_threshold_relative: how far below the signing region bottom the rest threshold should be
            proximity_threshold: if both hands are present and closer than this, it's rest
            min_run_length: how many consecutive rest frames are needed to confirm a boundary
        returns:
            (first_stroke_frame, last_stroke_frame)
            
        """
        if self.__filled_palms is None:
            self.getFilledPalmCenters()

        lenths = self.getSignLenths()
        start  = lenths.first_hand
        end    = lenths.last_hand

        # ── derive relative y threshold from the both-hands signing region ────
        # collect y values only from frames where both hands are present
        # to get a clean picture of where active signing happens
        signing_y_values = []
        for i in range(lenths.first_with_both, lenths.last_with_both + 1):
            palms = self.__filled_palms[i]
            left  = palms.get('left')
            right = palms.get('right')
            if (left  and left  != [None, None] and None not in left and
                right and right != [None, None] and None not in right):
                signing_y_values.append(left[1])
                signing_y_values.append(right[1])

        if signing_y_values:
            # the bottom of the active signing space (highest y value = lowest on screen)
            signing_y_bottom = np.percentile(signing_y_values, 85)
            # rest threshold: meaningfully below the signing region
            y_threshold = signing_y_bottom + y_threshold_relative
        else:
            # fallback to absolute if no both-hands region found
            y_threshold = 0.75

        # frame classifier 

        def classify(i):
            """
            Returns:
                'rest'        — frame confidently looks like rest position
                'active'      — frame confidently looks like signing
                'inconclusive'— only one hand present, can't tell
            """
            palms = self.__filled_palms[i]
            left  = palms.get('left')
            right = palms.get('right')

            left_valid  = left  and left  != [None, None] and None not in left
            right_valid = right and right != [None, None] and None not in right

            if left_valid and right_valid:
                both_low  = left[1] > y_threshold and right[1] > y_threshold
                too_close = math.hypot(right[0] - left[0],
                                    right[1] - left[1]) < proximity_threshold
                return 'rest' if (both_low or too_close) else 'active'

            elif left_valid or right_valid:
                # one hand present — check that hand alone against y threshold
                hand = left if left_valid else right
                return 'rest' if hand[1] > y_threshold else 'inconclusive'

            else:
                return 'inconclusive'

        # scan forward from start
        first_stroke = start
        consecutive  = 0

        for i in range(start, end + 1):
            result = classify(i)
            if result == 'rest':
                consecutive += 1
            elif result == 'inconclusive':
                continue        # skip — don't reset counter, don't stop
            else:               # 'active'
                if consecutive >= min_run_length:
                    first_stroke = i
                break

        # ── scan backward from end ────────────────────────────────────────────
        last_stroke = end
        consecutive = 0

        for i in range(end, start - 1, -1):
            result = classify(i)
            if result == 'rest':
                consecutive += 1
            elif result == 'inconclusive':
                continue
            else:               # 'active'
                if consecutive >= min_run_length:
                    last_stroke = i
                break

        if first_stroke >= last_stroke:
            return start, end

        return first_stroke, last_stroke
    
    def findPalmDistances(self):
        """
        Calculates frame-by-frame distances between filled palm centers.
        
        For missing palms, uses the closest known location (last seen position if available).
        This allows measuring hand proximity even during transition frames where one hand
        may temporarily disappear.
        
        Returns:
            list of float: distance between left and right palms for each frame,
                          None for frames where neither palm position is available
        """
        if self.__hand_distances is not None:
            return self.__hand_distances
        
        filled_palms, _ = self.getFilledPalmCenters()
        distances = []
        
        # Track last known positions for handling edge cases
        last_left = None
        last_right = None
        
        for i, frame_palms in enumerate(filled_palms):
            left = frame_palms.get('left')
            right = frame_palms.get('right')
            
            # Check if positions are valid (not None and not [None, None])
            left_valid = left and left != [None, None] and None not in left
            right_valid = right and right != [None, None] and None not in right
            
            # Update last known positions
            if left_valid:
                last_left = left
            if right_valid:
                last_right = right
            
            # Use filled positions if valid, otherwise use last known
            left_pos = left if left_valid else last_left
            right_pos = right if right_valid else last_right
            
            # Calculate distance if both positions are available
            if left_pos and right_pos:
                distance = math.hypot(right_pos[0] - left_pos[0], right_pos[1] - left_pos[1])
                distances.append(distance)
            else:
                distances.append(None)
        
        self.__hand_distances = distances
        return distances
    
    def find_closest_distances(self):
        """
        Finds the closest pair of keypoints (one from left hand, one from right hand)
        in each frame and returns their distances.
        
        For each frame:
        - Extracts all valid keypoints from left hand (21 landmarks)
        - Extracts all valid keypoints from right hand (21 landmarks)
        - Computes distances between all left-right pairs
        - Returns the minimum distance
        
        Returns:
            list of float: minimum distance between closest left-right keypoint pair per frame,
                          None for frames where either hand has no valid keypoints
        """
        keypoint_data = self.getKeyPointsAsLists()
        closest_distances = []
        
        for frame_idx, frame_keypoints in enumerate(keypoint_data):
            left_landmarks = frame_keypoints.get('left')
            right_landmarks = frame_keypoints.get('right')
            
            # Ensure we have valid landmark sequences
            if not left_landmarks or not right_landmarks:
                closest_distances.append(None)
                continue
            
            # Collect valid left keypoints
            valid_left = []
            for lm in left_landmarks:
                if lm and lm != [None, None] and None not in lm:
                    valid_left.append(lm)
            
            # Collect valid right keypoints
            valid_right = []
            for lm in right_landmarks:
                if lm and lm != [None, None] and None not in lm:
                    valid_right.append(lm)
            
            # If either hand has no valid keypoints, skip this frame
            if not valid_left or not valid_right:
                closest_distances.append(None)
                continue
            
            # Find minimum distance between all left-right pairs
            min_distance = float('inf')
            for left_kp in valid_left:
                for right_kp in valid_right:
                    distance = math.hypot(right_kp[0] - left_kp[0], right_kp[1] - left_kp[1])
                    if distance < min_distance:
                        min_distance = distance
            
            closest_distances.append(min_distance)
        
        return closest_distances

    def getHandScales(self, smoothing_window=5):
        # Enforce odd window for medfilt
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        keypoint_data = self.getKeyPointsAsLists()
        n = len(keypoint_data)
        
        # Adjust kernel size if data is too small
        kernel_size = min(smoothing_window, n if n > 0 else 1)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(1, kernel_size)

        left_scales  = np.empty(n)
        right_scales = np.empty(n)

        for frame_idx, frame in enumerate(keypoint_data):
            for side, out_array in (('left', left_scales), ('right', right_scales)):
                landmarks = frame[side]  # list of 41 [x, y] entries; None if hand is missing
                
                # Handle missing hands (after cropping, some frames may have None)
                if landmarks is None or len(landmarks) < 13:
                    out_array[frame_idx] = np.nan
                    continue
                
                wrist  = landmarks[0]   # landmark 0 — wrist
                mid_tip = landmarks[12] # landmark 12 — middle fingertip
                
                # Check if landmarks are valid
                if wrist is None or mid_tip is None or None in wrist or None in mid_tip:
                    out_array[frame_idx] = np.nan
                    continue

                out_array[frame_idx] = math.hypot(
                    mid_tip[0] - wrist[0],
                    mid_tip[1] - wrist[1],
                )

        # Fill NaN values with forward-fill then backward-fill, or use mean as fallback
        left_scales = self._fillNaNSequence(left_scales)
        right_scales = self._fillNaNSequence(right_scales)

        # Median-filter each hand's scale sequence independently (only if kernel_size > 1)
        if kernel_size > 1:
            left_smoothed  = medfilt(left_scales,  kernel_size=kernel_size).astype(float)
            right_smoothed = medfilt(right_scales, kernel_size=kernel_size).astype(float)
        else:
            left_smoothed = left_scales.astype(float)
            right_smoothed = right_scales.astype(float)
        
        combined       = np.maximum(left_smoothed, right_smoothed)

        return {
            'left':     left_smoothed,
            'right':    right_smoothed,
            'combined': combined,
        }
    
    @staticmethod
    def _fillNaNSequence(sequence):
        """Fill NaN values in a sequence using forward-fill, backward-fill, then mean."""
        arr = np.array(sequence, dtype=float)
        
        # Forward fill
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        idx = np.maximum.accumulate(idx)
        arr[mask] = arr[idx[mask]]
        
        # If still NaN (all were NaN), use mean
        if np.any(np.isnan(arr)):
            valid_mean = np.nanmean(arr)
            if np.isnan(valid_mean):
                valid_mean = 1.0  # fallback default
            arr[np.isnan(arr)] = valid_mean
        
        return arr

    @staticmethod
    def _nearestFill(sequence):
        """Fills missing values with the temporally closest known value.
        
        No interpolation — each missing entry gets a copy of whichever 
        known entry is nearest in time. If equidistant, uses the earlier one.
        
        Example: [m, m, a, b, m, m, m, c, m, m, d, m]
              -> [a, a, a, b, b, b, c, c, c, d, d, d]
        
        takes:
            sequence: list where known entries are [x, y] and missing 
                      entries are None or [None, None]
        returns:
            list of same length with all missing entries filled
        """
        n = len(sequence)
        filled = [None] * n
        
        # Find indices of known values
        known = []
        for i, val in enumerate(sequence):
            if val is not None and val != [None, None] and None not in val:
                known.append(i)
                filled[i] = val
        
        if not known:
            return sequence  # nothing to fill with
        
        # For each missing position, find the nearest known index
        # Use two-pointer approach: track the closest known on each side
        ki = 0  # pointer into known[]
        for i in range(n):
            if filled[i] is not None:
                # advance pointer to stay at or just past i
                while ki < len(known) - 1 and known[ki] < i:
                    ki += 1
                continue
            
            # Find closest known index
            # Check the known index at and around ki
            best_idx = known[0]
            best_dist = abs(i - known[0])
            
            for k in known:
                dist = abs(i - k)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = k
                elif dist > best_dist and k > i:
                    break  # known is sorted, won't get closer
            
            filled[i] = sequence[best_idx]
        
        return filled
 
    def findHandOrderingByPalmCenterUsingNeighbourFilling(self, margin=0.0, show_logs=False):
        """Flags frames where the left palm center X > right palm center X.
        
        Compares the averaged palm landmark positions (landmarks 0,1,2,5,9,13,17).
        Missing hands are filled with their nearest known palm center position.
        
        takes:
            margin: minimum x-distance required to count as a violation.
                    0.0 = any crossing counts.
        returns:
            list of frame indices where left palm center X > right palm center X + margin
        """
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters()
        
        raw_palms = self._KeyPointValidator__palms
        
        # Extract and nearest-fill each side independently
        left_positions = [p.get('left', [None, None]) for p in raw_palms]
        right_positions = [p.get('right', [None, None]) for p in raw_palms]
        
        filled_left = self._nearestFill(left_positions)
        filled_right = self._nearestFill(right_positions)
        
        violations = []
        
        for i in range(len(raw_palms)):
            left = filled_left[i]
            right = filled_right[i]
            
            # Skip if a hand was never seen at all
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByPalmCenter(margin={margin}): "
                  f"{len(violations)} violations out of {len(raw_palms)} frames")
        
        return violations
 
    def findHandOrderingByWristUsingNeighbourFilling(self, margin=0.0, show_logs=False):
        """Flags frames where left wrist X > right wrist X.
        
        Compares landmark 0 (wrist) only. Missing hands are filled with 
        their nearest known wrist position.
        
        takes:
            margin: minimum x-distance required to count as a violation
        returns:
            list of frame indices where left wrist X > right wrist X + margin
        """
        keypoint_data = self.getKeyPointsAsLists()
        
        # Extract wrist positions per side
        left_wrists = []
        right_wrists = []
        
        for frame_kps in keypoint_data:
            # Left wrist
            if frame_kps.get('left') and frame_kps['left'][0]:
                lw = frame_kps['left'][0]
                if lw != [None, None] and None not in lw:
                    left_wrists.append(lw)
                else:
                    left_wrists.append([None, None])
            else:
                left_wrists.append([None, None])
            
            # Right wrist
            if frame_kps.get('right') and frame_kps['right'][0]:
                rw = frame_kps['right'][0]
                if rw != [None, None] and None not in rw:
                    right_wrists.append(rw)
                else:
                    right_wrists.append([None, None])
            else:
                right_wrists.append([None, None])
        
        filled_left = self._nearestFill(left_wrists)
        filled_right = self._nearestFill(right_wrists)
        
        violations = []
        
        for i in range(len(keypoint_data)):
            left = filled_left[i]
            right = filled_right[i]
            
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByWrist(margin={margin}): "
                  f"{len(violations)} violations out of {len(keypoint_data)} frames")
        
        return violations
 
    def findHandOrderingByExtremesUsingNeighbourFilling(self, margin=0.0, show_logs=False):
        """Flags frames where left hand's minimum X > right hand's maximum X.
        
        For each frame where a hand is visible, computes min_x (left) or 
        max_x (right) across all landmarks. Missing frames are filled with 
        the nearest known computed value.
        
        If even left's min-X exceeds right's max-X, the hands are completely 
        in the wrong order with no overlap.
        
        takes:
            margin: minimum x-distance required to count as a violation
        returns:
            list of frame indices where min_x(left) > max_x(right) + margin
        """
        keypoint_data = self.getKeyPointsAsLists()
        
        # Compute extremes per frame where data exists, None where missing
        left_min_xs = []
        right_max_xs = []
        
        for frame_kps in keypoint_data:
            # Left hand: compute min x across all valid landmarks
            if frame_kps.get('left'):
                valid_xs = [lm[0] for lm in frame_kps['left'] 
                           if lm and lm != [None, None] and None not in lm]
                if valid_xs:
                    left_min_xs.append([min(valid_xs), 0])  # wrap as [x, y] for _nearestFill
                else:
                    left_min_xs.append([None, None])
            else:
                left_min_xs.append([None, None])
            
            # Right hand: compute max x across all valid landmarks
            if frame_kps.get('right'):
                valid_xs = [lm[0] for lm in frame_kps['right'] 
                           if lm and lm != [None, None] and None not in lm]
                if valid_xs:
                    right_max_xs.append([max(valid_xs), 0])
                else:
                    right_max_xs.append([None, None])
            else:
                right_max_xs.append([None, None])
        
        filled_left = self._nearestFill(left_min_xs)
        filled_right = self._nearestFill(right_max_xs)
        
        violations = []
        
        for i in range(len(keypoint_data)):
            left = filled_left[i]
            right = filled_right[i]
            
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByExtremes(margin={margin}): "
                  f"{len(violations)} violations out of {len(keypoint_data)} frames")
        
        return violations
    
    def findHandOrderingByPalmCenterUsingInterpolation(self, margin=0.0, show_logs=False):
        """Flags frames where the left palm center X > right palm center X.
        
        Compares the averaged palm landmark positions (landmarks 0,1,2,5,9,13,17).
        Missing hands are filled with linear interpolation between known positions.
        
        takes:
            margin: minimum x-distance required to count as a violation
        returns:
            list of frame indices where left palm center X > right palm center X + margin
        """
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters()
        
        raw_palms = self._KeyPointValidator__palms
        
        # Extract and interpolate each side independently
        left_positions = [p.get('left', [None, None]) for p in raw_palms]
        right_positions = [p.get('right', [None, None]) for p in raw_palms]
        
        filled_left, _ = self.interpolateSequence(left_positions)
        filled_right, _ = self.interpolateSequence(right_positions)
        
        violations = []
        
        for i in range(len(raw_palms)):
            left = filled_left[i]
            right = filled_right[i]
            
            # Skip if a hand was never seen at all
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByPalmCenterUsingInterpolation(margin={margin}): "
                  f"{len(violations)} violations out of {len(raw_palms)} frames")
        
        return violations
    
    def findHandOrderingByWristUsingInterpolation(self, margin=0.0, show_logs=False):
        """Flags frames where left wrist X > right wrist X.
        
        Compares landmark 0 (wrist) only. Missing hands are filled with linear 
        interpolation between known wrist positions.
        
        takes:
            margin: minimum x-distance required to count as a violation
        returns:
            list of frame indices where left wrist X > right wrist X + margin
        """
        keypoint_data = self.getKeyPointsAsLists()
        
        # Extract wrist positions per side
        left_wrists = []
        right_wrists = []
        
        for frame_kps in keypoint_data:
            # Left wrist
            if frame_kps.get('left') and frame_kps['left'][0]:
                lw = frame_kps['left'][0]
                if lw != [None, None] and None not in lw:
                    left_wrists.append(lw)
                else:
                    left_wrists.append([None, None])
            else:
                left_wrists.append([None, None])
            
            # Right wrist
            if frame_kps.get('right') and frame_kps['right'][0]:
                rw = frame_kps['right'][0]
                if rw != [None, None] and None not in rw:
                    right_wrists.append(rw)
                else:
                    right_wrists.append([None, None])
            else:
                right_wrists.append([None, None])
        
        filled_left, _ = self.interpolateSequence(left_wrists)
        filled_right, _ = self.interpolateSequence(right_wrists)
        
        violations = []
        
        for i in range(len(keypoint_data)):
            left = filled_left[i]
            right = filled_right[i]
            
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByWristUsingInterpolation(margin={margin}): "
                  f"{len(violations)} violations out of {len(keypoint_data)} frames")
        
        return violations
    
    def findHandOrderingByExtremesUsingInterpolation(self, margin=0.0, show_logs=False):
        """Flags frames where left hand's minimum X > right hand's maximum X.
        
        For each frame where a hand is visible, computes min_x (left) or 
        max_x (right) across all landmarks. Missing frames are filled with 
        linear interpolation between known computed values.
        
        If even left's min-X exceeds right's max-X, the hands are completely 
        in the wrong order with no overlap.
        
        takes:
            margin: minimum x-distance required to count as a violation
        returns:
            list of frame indices where min_x(left) > max_x(right) + margin
        """
        keypoint_data = self.getKeyPointsAsLists()
        
        # Compute extremes per frame where data exists, None where missing
        left_min_xs = []
        right_max_xs = []
        
        for frame_kps in keypoint_data:
            # Left hand: compute min x across all valid landmarks
            if frame_kps.get('left'):
                valid_xs = [lm[0] for lm in frame_kps['left'] 
                           if lm and lm != [None, None] and None not in lm]
                if valid_xs:
                    left_min_xs.append([min(valid_xs), 0])  # wrap as [x, y] for interpolation
                else:
                    left_min_xs.append([None, None])
            else:
                left_min_xs.append([None, None])
            
            # Right hand: compute max x across all valid landmarks
            if frame_kps.get('right'):
                valid_xs = [lm[0] for lm in frame_kps['right'] 
                           if lm and lm != [None, None] and None not in lm]
                if valid_xs:
                    right_max_xs.append([max(valid_xs), 0])
                else:
                    right_max_xs.append([None, None])
            else:
                right_max_xs.append([None, None])
        
        filled_left, _ = self.interpolateSequence(left_min_xs)
        filled_right, _ = self.interpolateSequence(right_max_xs)
        
        violations = []
        
        for i in range(len(keypoint_data)):
            left = filled_left[i]
            right = filled_right[i]
            
            if (left is None or left == [None, None] or None in left or
                right is None or right == [None, None] or None in right):
                continue
            
            if left[0] > right[0] + margin:
                violations.append(i)
        
        if show_logs:
            print(f"findHandOrderingByExtremesUsingInterpolation(margin={margin}): "
                  f"{len(violations)} violations out of {len(keypoint_data)} frames")
        
        return violations
    
    def findAccelerationClusters(self, 
                                 margin=0.0, 
                                 show_logs=False, 
                                 inclusive=True, 
                                 interpolate_missing=False):
        '''
        When there are 3 simaltaneous peaks in the acceleration of the hands, 
        it is likely that media pipe hands has switched the hand labels.
        This function finds these peaks and returns the index of the frame 
        in the center all 3 frames.
        
        takes:
            margin: minimum acceleration required to count as a peak
            show_logs: if true, prints the number of clusters found and the total number of frames
            inclusive: if true, includes all frames in the center of 3 peaks 
                so if there are 6 peaks in a row, it will include frames 2,3,4,5
                if its false, it will only include the center frame of 3 peaks, so in the previous example 
                it would only include frame 2 (with sides 1, 2, 3) and 5 (with sides 4, 5, 6)
             
        returns:
            list of all susputions frame indexs 
        '''
        
        if interpolate_missing:
            accelerations = self.getEstimatedAccelerations()
        else:
            accelerations = self.getAccelerations()
        
        for acceleration in accelerations:
            accel = acceleration['acceleration']  # unwrap the nesting
            
            if accel['left'].get('est'):
                accel['left']['is_peak'] = False
            else:
                accel['left']['is_peak'] = accel['left']['acceleration'] > margin
                
            if accel['right'].get('est'):
                accel['right']['is_peak'] = False
            else:
                accel['right']['is_peak'] = accel['right']['acceleration'] > margin
        left_clusters = []
        right_clusters = []

        for i in range(1, len(accelerations) - 1):
            left_peaks = (accelerations[i-1]['acceleration']['left']['is_peak'],
                  accelerations[i]['acceleration']['left']['is_peak'],
                  accelerations[i+1]['acceleration']['left']['is_peak'])
            right_peaks = (accelerations[i-1]['acceleration']['right']['is_peak'], 
                            accelerations[i]['acceleration']['right']['is_peak'], 
                            accelerations[i+1]['acceleration']['right']['is_peak'])
            
            if all(left_peaks):
                if inclusive:
                    left_clusters.append(i+1)
                elif not (i-1 in left_clusters or i-2 in left_clusters):
                    left_clusters.append(i+1)
                
                    
            if all(right_peaks):
                if inclusive:
                    right_clusters.append(i+1)
                elif not (i-1 in right_clusters or i-2 in right_clusters):
                    right_clusters.append(i+1)
                    
        if show_logs:
            print(f"findAccelerationClusters(margin={margin}, inclusive={inclusive}): "
                  f"{len(left_clusters)} left clusters, {len(right_clusters)} right clusters out of {len(accelerations)} frames")
        
        return {
            'left': left_clusters,
            'right': right_clusters
        }