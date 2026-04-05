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
        # loads json file
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
        '''Uses relitive to'''
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
                                y_threshold_relative=0.15,  # rest is this far below signing space bottom
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

        left_scales  = np.empty(n)
        right_scales = np.empty(n)

        for frame_idx, frame in enumerate(keypoint_data):
            for side, out_array in (('left', left_scales), ('right', right_scales)):
                landmarks = frame[side]  # list of 41 [x, y] entries; all valid at this stage

                wrist  = landmarks[0]   # landmark 0 — wrist
                mid_tip = landmarks[12] # landmark 12 — middle fingertip

                out_array[frame_idx] = math.hypot(
                    mid_tip[0] - wrist[0],
                    mid_tip[1] - wrist[1],
                )

        # Median-filter each hand's scale sequence independently
        left_smoothed  = medfilt(left_scales,  kernel_size=smoothing_window).astype(float)
        right_smoothed = medfilt(right_scales, kernel_size=smoothing_window).astype(float)
        combined       = np.maximum(left_smoothed, right_smoothed)

        return {
            'left':     left_smoothed,
            'right':    right_smoothed,
            'combined': combined,
        }

if __name__ == "__main__":
    a = r'''
    from pathlib import Path
    from tabulate import tabulate
    
    # Define the corpus path
    corpus_path = Path(r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus')
    
    # Collect results
    results = []
    errors = []
    
    # Process each JSON file in the corpus
    json_files = sorted(corpus_path.glob('*.json'))
    
    print(f"\nProcessing {len(json_files)} sign files from Validated_interpolated_SubCorpus...")
    print("=" * 80)
    
    for json_file in json_files:
        sign_name = json_file.stem  # filename without .json extension
        
        try:
            # Instantiate interpolator and detect rest position frames
            interpolator = CubicSplineKeyPointInterpolator(str(json_file))
            first_stroke, last_stroke = interpolator.detectRestPositionFrames()
            
            # Calculate stroke duration
            stroke_duration = last_stroke - first_stroke + 1
            
            results.append({
                'Sign': sign_name,
                'First Stroke': first_stroke,
                'Last Stroke': last_stroke,
                'Duration': stroke_duration
            })
            
        except Exception as e:
            errors.append({
                'Sign': sign_name,
                'Error': str(e)
            })
            print(f"[ERROR] {sign_name}: {str(e)}")
    
    # Display results in table format
    print("\n" + "=" * 80)
    print("STROKE POSITION DETECTION RESULTS")
    print("=" * 80)
    if results:
        try:
            print(tabulate(results, headers='keys', tablefmt='grid', showindex=True))
        except ImportError:
            # Fallback if tabulate is not available
            print("\n{:<40} {:<15} {:<15} {:<10}".format('Sign', 'First Stroke', 'Last Stroke', 'Duration'))
            print("-" * 80)
            for row in results:
                print("{:<40} {:<15} {:<15} {:<10}".format(
                    row['Sign'], row['First Stroke'], row['Last Stroke'], row['Duration']
                ))
    
    # Display summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        durations = [r['Duration'] for r in results]
        print(f"Total signs processed: {len(results)}")
        print(f"Average stroke duration: {sum(durations) / len(durations):.1f} frames")
        print(f"Min stroke duration: {min(durations)} frames")
        print(f"Max stroke duration: {max(durations)} frames")
    
    # Display errors if any
    if errors:
        print("\n" + "=" * 80)
        print("PROCESSING ERRORS")
        print("=" * 80)
        for error in errors:
            print(f"  • {error['Sign']}: {error['Error']}")
    
    print("=" * 80 + "\n")
        
    
    
    a = 
    path = r'C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed\B\3e9dd7e5-f6a3-4b1f-9f29-9fba66f0b73c.json'
    path = r"C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed\B\acf7a090-7ece-488a-a6e8-f4df878629a9.json"
    path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Stratified_stratified - recursive level 1\1be98b34-0edc-41ee-871c-e592c0b4198f.json"
    validator = CubicSplineKeyPointInterpolator(path)
    
    print()
    
    left_missing, right_missing = validator.findNonMissingValueClusters()
    left_fast, right_fast = validator.findMovmentClusters()
    
    #print('visible values\n', left_missing, '\n\n', 'movment\n', left_fast)
    
    
    print(validator.estimateMissingMomentums(), "\n\n")
    
    print(validator.findMovmentClusters(max_momentum=0.1))
    
    
    
    
    frame = 66
    
    print("palm centers:")
    print(validator.findAllPalmCenters())
    
    print("\nmomentums:")
    print(validator.getMomentums())

    print("\naccelerations:")
    print(validator.getAccelerations())  # Use frame-1 since accelerations are one frame behind momentums
    
    print("\nFilled palm centers with estimation:")
    estimator = KalmanKeyPointEstimator(path)
    filled_palms, estimation_flags = estimator.estimateMissingPalmCenters()
    print(f"Total frames: {filled_palms[frame]}")
    
    
    estimator = KalmanKeyPointEstimator(path)
    print("\nFilled palm centers with estimation:")
    filled_palms, estimation_flags = estimator.estimateMissingPalmCenters(show_logs=False)
    print(f"Total frames: {len(filled_palms)}")
    
    print("\nEstimated momentums:")
    estimated_momentums = estimator.getEstimatedMomentums()
    print(f"Total momentums: {len(estimated_momentums)}")
    print("\nEstimated accelerations:")
    estimated_accelerations = estimator.getEstimatedAccelerations()
    print(f"Total accelerations: {len(estimated_accelerations)}")'''