import json
import math
import traceback
import os
import numpy as np

from validation_helpers import *
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
        
    def checkForSimultaneousHands(self, showLogs = False):
        frames = self.data.get('frames', [])
        found_error = False
        
        errors = []

        for frame in frames:
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
                    if showLogs:
                        print(
                            f"Simultaneous {side} hands detected in frame {frame_index} "
                            f"(timestamp={timestamp}): duplicate pairs={duplicate_pairs}"
                        )

                    errors.append((frame_index, side))
        return errors
    
    def checkForMissingHands(self, showLogs=False):
        frames = self.data.get('frames', [])
        found_error = False
        
        errors = []

        for frame in frames:
            frame_index = frame.get('frame_index')
            timestamp = frame.get('timestamp')
            hands = frame.get('hands', {})

            for side in ('left', 'right'):
                landmarks = hands.get(side, []) if isinstance(hands, dict) else []
                if not landmarks:
                    found_error = True
                    if showLogs:
                        print(f"missing {side} hand detected in frame {frame_index} (timestamp={timestamp})")
                    errors.append((frame_index, side))
        return errors
    
    def viewFrame(self, filepath, frame_index):

        frames = self.data.get('frames', [])
        
        if frame_index < 0 or frame_index >= len(frames):
            print(f"Frame index {frame_index} out of range. Total frames: {len(frames)}")
            return
        
        frame = frames[frame_index]
        print(json.dumps(frame, indent=2))
        
   # ------ setter functions for palm centers, momentum and acceleration --------
    
    def findPalmCenters(self, frameIdx, showLogs=False):
        # computes the average position of the palm points 
        # (landmark_id in [0, 1, 2, 5, 9, 13, 17]) for each hand
        # in the specified frame
        try:
            frames = self.data.get('frames', [])
            frame = frames[frameIdx]
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
                    if showLogs:
                        print(f"Frame {frameIdx}, {side}: Found {len(palm_points)} palm points, center: [{avg_x:.3f}, {avg_y:.3f}]")
                else:
                    palmCenters[side] = [None, None]
                    if showLogs:
                        print(f"Frame {frameIdx}, {side}: No palm points found (landmarks: {len(landmarks)})")
            
            return palmCenters
            
        except Exception as e:
            print(f"ERROR in findPalmCenters({frameIdx}): {e}")
            traceback.print_exc()
            return {}

    def findAllPalmCenters(self, showLogs=False):
        frames = self.data.get('frames', [])
        self.__palms = []
        
        for i in range(len(frames)):
            self.palmCenters = self.findPalmCenters(i, showLogs=showLogs)
            self.__palms.append(self.palmCenters)
        
        return self.__palms
    
    def findMomentum(self, palms, palms2, showLogs=False):
        
        if showLogs:
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

    def storeMomentum(self, showLogs=False):
        # stores the momentum values in an array as a class atribute
        frames = self.data.get('frames', [])
        self.__momentums = []
        for i in range(len(frames)-1):
            if self.__palms is None:
                self.findAllPalmCenters(showLogs=showLogs)
            
            palms = self.__palms[i] if i < len(self.__palms) \
                else self.findPalmCenters(i)
            palms2 = self.__palms[i+1] if i+1 < len(self.__palms) \
                else self.findPalmCenters(i+1, showLogs=showLogs)
        
            momentum = self.findMomentum(palms, palms2, showLogs=showLogs)
            self.__momentums.append(momentum)
            
    def computeAcceleration(self, palms, palms2, palms3, showLogs=False): 
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
        
        
        frames = self.data.get('frames', [])
        
        if not self.__momentums:
            self.storeMomentum()
        
        # Get momentums with estimation tracking
        
        momentum_1 = self.findMomentum(palms, palms2, showLogs=showLogs)
        momentum_2 = self.findMomentum(palms3, palms2, showLogs=showLogs)  # Use palms2 as second palm for momentum_2
        
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
        frames = self.data.get('frames', [])
        self.__accelerations = []
        for i in range(len(frames)-2):
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
        # return a Sigh_lenths object with filled values
        
        if self.__palms is None:
            self.findAllPalmCenters()
        
        frames = self.data.get('frames', [])
        total_frames = len(frames)
        
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
            


class KalmanKeyPointEstimator(KeyPointValidator):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.__filled_palms = None
        self.__estimation_flags = None
        self.__fps = 25  # Store FPS
        self.__accelerationsEstimated = None
        self.__momentumsEstimated = None
        
    def estimateMissingPalmCenters(self, fps=25, showLogs=False):
        """
        Fill missing palm centers using Kalman filtering.
        
        Kalman filter must predict() all frames, then update() if measurement exists
        """
        frames = self.data.get('frames', [])
        filled_centers = []
        estimation_flags = []
        
        # Get all palm centers first
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters(showLogs=showLogs)
        
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
        
        for i in range(len(frames)):
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
                    
                    if showLogs and i % 10 == 0:  # Log every 10 frames
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
        
        if showLogs:
            # Calculate statistics
            left_estimated = sum(1 for e in estimation_flags if e['left'])
            right_estimated = sum(1 for e in estimation_flags if e['right'])
            print(f"\nEstimation summary:")
            print(f"  Left hand: {left_estimated}/{len(frames)} frames estimated "
                  f"({100*left_estimated/len(frames):.1f}%)")
            print(f"  Right hand: {right_estimated}/{len(frames)} frames estimated "
                  f"({100*right_estimated/len(frames):.1f}%)")
        
        return filled_centers, estimation_flags
    
    def getFilledPalmCenters(self):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters()
        return self.__filled_palms
    
    def estimateMissingMomentums(self, showLogs=False):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters(showLogs=showLogs)
        
        # Recompute momentums using filled palm centers
        momentums = []
        for i in range(len(self.__filled_palms)-1):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            momentum = self.findMomentum(palms, palms2, showLogs=showLogs)
            momentums.append(momentum)
        
        return momentums
    
    def getEstimatedMomentums(self):
        if self.__momentumsEstimated is None:
            self.__momentumsEstimated = self.estimateMissingMomentums()
        return self.__momentumsEstimated
    
    def estimateMissingAccelerations(self, showLogs=False):
        if self.__filled_palms is None:
            self.estimateMissingPalmCenters(showLogs=showLogs)
        accelerations = []
        for i in range(len(self.__filled_palms)-2):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            palms3 = self.__filled_palms[i+2]
            accelerations.append(self.computeAcceleration(palms, 
                                                          palms2, 
                                                          palms3, 
                                                          showLogs=showLogs))
        
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
        self.methods = ['cubic', 'pchip', 'linear']
        if method in self.methods:
            self.method = method

    
    def interpolate_sequence(self, positions):
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
            return self.__filled_palms
        
        if self._KeyPointValidator__palms is None:
            self.findAllPalmCenters()
        
        filled_palms = []
        estimation_flags = []
        
        for side in ['left', 'right']:
            # Extract sequence for this hand
            positions = [p[side] for p in self._KeyPointValidator__palms]
            filled_positions, was_interpolated = self.interpolate_sequence(positions)
            
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
    
    def estimateMissingMomentums(self, showLogs=False):
        if self.__filled_palms is None:
            self.getFilledPalmCenters()
        
        momentums = []
        for i in range(len(self.__filled_palms)-1):
            palms = self.__filled_palms[i]
            palms2 = self.__filled_palms[i+1]
            momentum = self.findMomentum(palms, palms2, showLogs=showLogs)
            momentums.append(momentum)
        
        return momentums
    
    def getEstimatedMomentums(self):
        if self.__momentumsEstimated is None:
            self.__momentumsEstimated = self.estimateMissingMomentums()
        return self.__momentumsEstimated
    
    def estimateMissingAccelerations(self, showLogs=False):
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
                                                          showLogs=showLogs))
        
        return accelerations
    
    def getEstimatedAccelerations(self):
        if self.__accelerationsEstimated is None:
            self.__accelerationsEstimated = self.estimateMissingAccelerations()
        return self.__accelerationsEstimated

if __name__ == "__main__":
    path = r'C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed\B\3e9dd7e5-f6a3-4b1f-9f29-9fba66f0b73c.json'

    validator = KeyPointValidator(path)
    '''
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
    '''
    
    estimator = KalmanKeyPointEstimator(path)
    print("\nFilled palm centers with estimation:")
    filled_palms, estimation_flags = estimator.estimateMissingPalmCenters(showLogs=False)
    print(f"Total frames: {len(filled_palms)}")
    
    print("\nEstimated momentums:")
    estimated_momentums = estimator.getEstimatedMomentums()
    print(f"Total momentums: {len(estimated_momentums)}")
    print("\nEstimated accelerations:")
    estimated_accelerations = estimator.getEstimatedAccelerations()
    print(f"Total accelerations: {len(estimated_accelerations)}")