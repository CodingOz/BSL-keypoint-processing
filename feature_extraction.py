import numpy as np
import json


class FeatureExtraction:
    def __init__(self, corpus_path, feature_path):
        self.corpus_path = corpus_path
        self.feature_path = feature_path

        # holds all jsons in the corpus
        self.corpus = []
        self.load_corpus()

    def loadCorpus(self):
        with open(self.corpus_path, 'r') as f:
            for line in f:
                self.corpus.append(json.loads(line))

    def extractProximityFeatures(
            self,
            json_obj,
            include_labels=False,
            include_palms=False):
        '''Extract proximity features from a single JSON object.
        there are C(2, 42) = 861 pairs of proximitys per frame,
        C(2, 44) = 946 pairs of proximitys per frame if include_palms is True,
        there are 10 frames per video, so there are 8610 proximity features per video.

        returns: a numpy array of all proximity features for the given JSON object.
        shape:
            if include_labels are False: (num_frames, 861) of floats representing the euclidean distances for each pair of points
            if include_labels is True: list of frames with labels
        '''
        frames = json_obj['frames']
        proximity_features = []

        for frame_idx, frame in enumerate(frames):
            landmarks = []
            # Left hand landmarks
            if 'hands' in frame and 'left' in frame['hands']:
                for landmark in frame['hands']['left']:
                    landmarks.append([landmark['x'], landmark['y']])

            # Right hand landmarks
            if 'hands' in frame and 'right' in frame['hands']:
                for landmark in frame['hands']['right']:
                    landmarks.append([landmark['x'], landmark['y']])

            frame_proximity = []
            if landmarks:
                landmarks = np.array(landmarks)

                num_landmarks = landmarks.shape[0]
                for i in range(num_landmarks):
                    for j in range(i + 1, num_landmarks):
                        distance = np.linalg.norm(landmarks[i] - landmarks[j])
                        if include_labels:
                            frame_proximity.append([distance, i, j, frame_idx])
                        else:
                            frame_proximity.append(distance)

            proximity_features.append(frame_proximity)

        return np.array(
            proximity_features,
            dtype=object) if include_labels else np.array(proximity_features)

    def extractAngleFeatures(self, json_obj, include_labels=False):
        '''Extract angle features from hand landmarks.
        Calculates joint angles within each hand and inter-hand angles.

        Returns:
            if include_labels is False: array of shape (num_frames, 48)
                - 21 angles per hand (42 total) + 6 inter-hand angles
            if include_labels is True: array of shape (num_frames, 48, 2)
                - angles with frame indices
        '''

        # Define angle triplets for within-hand angles (point1, joint, point2)
        # These calculate the angle AT the joint between the two edges
        intra_hand_angles = [
            # Thumb: angles at joints along the chain
            (0, 1, 2), (1, 2, 3), (2, 3, 4),
            # Index: angles at joints
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            # Middle: angles at joints
            (0, 5, 9), (5, 9, 10), (9, 10, 11), (10, 11, 12),
            # Ring: angles at joints
            (5, 9, 13), (9, 13, 14), (13, 14, 15), (14, 15, 16),
            # Pinky: angles at joints
            (9, 13, 17), (13, 17, 18), (17, 18, 19), (18, 19, 20),
            # Wrist angles (between root edges)
            (1, 0, 5), (1, 0, 17), (5, 0, 17),
        ]

        inter_hand_pairs = [
            (0, 0),   # left wrist to right wrist
            (4, 4),   # left thumb tip to right thumb tip
            (8, 8),   # left index tip to right index tip
            (12, 12),  # left middle tip to right middle tip
            (16, 16),  # left ring tip to right ring tip
            (20, 20),  # left pinky tip to right pinky tip
        ]

        frames = json_obj['frames']
        angle_features = []

        for frame_idx, frame in enumerate(frames):
            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            frame_angles = []

            # Calculate intra-hand angles for left hand
            if left_landmarks is not None and len(left_landmarks) >= 21:
                for p1, joint, p2 in intra_hand_angles:
                    angle = self._calculate_angle(left_landmarks[p1],
                                                  left_landmarks[joint],
                                                  left_landmarks[p2])
                    frame_angles.append(angle)
            else:
                frame_angles.extend([0] * len(intra_hand_angles))

            # Calculate intra-hand angles for right hand
            if right_landmarks is not None and len(right_landmarks) >= 21:
                for p1, joint, p2 in intra_hand_angles:
                    angle = self._calculate_angle(right_landmarks[p1],
                                                  right_landmarks[joint],
                                                  right_landmarks[p2])
                    frame_angles.append(angle)
            else:
                frame_angles.extend([0] * len(intra_hand_angles))

            # Calculate inter-hand angles
            if (left_landmarks is not None and right_landmarks is not None and
                    len(left_landmarks) >= 21 and len(right_landmarks) >= 21):
                for left_idx, right_idx in inter_hand_pairs:
                    angle = self._calculate_angle(left_landmarks[left_idx],
                                                  # left wrist as reference
                                                  left_landmarks[0],
                                                  right_landmarks[right_idx])
                    frame_angles.append(angle)
            else:
                frame_angles.extend([0] * len(inter_hand_pairs))

            if include_labels:
                frame_with_labels = [[angle, frame_idx]
                                     for angle in frame_angles]
                angle_features.append(frame_with_labels)
            else:
                angle_features.append(frame_angles)

        return np.array(angle_features)

    @staticmethod
    def _calculateAngle(p1, vertex, p2):
        '''Calculate angle at vertex between vectors (p1->vertex) and (vertex->p2).

        Returns angle in radians.
        '''
        v1 = p1 - vertex
        v2 = p2 - vertex

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (mag1 * mag2)

        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Return angle in radians
        return np.arccos(cos_angle)

    def extract_feature_set_1_combined(self, json_obj):
        '''Feature Set 1 (Baseline - Largest):
        All proximity features (8610) + All angle features (480) = 9090 total inputs.
        Verbose approach for baseline accuracy comparison.
        '''
        proximity = self.extract_proximity_features(
            json_obj, include_labels=False)
        angles = self.extract_angle_features(json_obj, include_labels=False)

        proximity = np.array(proximity)
        angles = np.array(angles)

        combined = np.hstack([proximity, angles])

        result = []
        for frame in combined:
            frame_list = [float(x) for x in frame]
            result.append(frame_list)

        return result

    def extract_feature_set_2_proximity_only(self, json_obj):
        '''Feature Set 2: Only proximity features (8610 inputs).'''
        proximity = self.extract_proximity_features(
            json_obj, include_labels=False)

        result = []
        for frame in proximity:
            frame_list = [float(x) for x in frame]
            result.append(frame_list)

        return result

    def extract_feature_set_3_tips_and_palms_with_angles(self, json_obj):
        '''Feature Set 3:
        Distances of fingertips + palm centers + angles between finger key points.
        Uses 12 key points: wrist(0) and fingertips(4,8,12,16,20) per hand.
        '''
        frames = json_obj['frames']
        key_indices = [0, 4, 8, 12, 16, 20]
        features = []

        # Extract angle features once for all frames
        angles = self.extract_angle_features(json_obj, include_labels=False)

        for frame_idx, frame in enumerate(frames):
            frame_features = []

            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            if left_landmarks is not None and right_landmarks is not None:
                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        if i < j:
                            # Left hand distances
                            dist = np.linalg.norm(
                                left_landmarks[idx1] - left_landmarks[idx2])
                            frame_features.append(float(dist))

                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        if i < j:
                            # Right hand distances
                            dist = np.linalg.norm(
                                right_landmarks[idx1] - right_landmarks[idx2])
                            frame_features.append(float(dist))

                # Inter-hand distances
                for idx1 in key_indices:
                    for idx2 in key_indices:
                        dist = np.linalg.norm(
                            left_landmarks[idx1] - right_landmarks[idx2])
                        frame_features.append(float(dist))

                # Add angle features for this frame
                if frame_idx < len(angles):
                    frame_angles = [float(x) for x in angles[frame_idx]]
                    frame_features.extend(frame_angles)

            features.append(frame_features)

        return features

    def extract_feature_set_4_tips_and_palms_with_angles(self, json_obj):
        '''Feature Set 4:
        Angles and distances of 12 key points (wrist and fingertips: 0,4,8,12,16,20 per hand).
        No other angle data beyond the 12-point interactions.
        '''
        frames = json_obj['frames']
        key_indices = [0, 4, 8, 12, 16, 20]
        features = []

        for frame in frames:
            frame_features = []

            # Extract landmarks
            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            if left_landmarks is not None and right_landmarks is not None:
                # Distances
                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        if i < j:
                            dist_l = np.linalg.norm(
                                left_landmarks[idx1] - left_landmarks[idx2])
                            dist_r = np.linalg.norm(
                                right_landmarks[idx1] - right_landmarks[idx2])
                            dist_inter = np.linalg.norm(
                                left_landmarks[idx1] - right_landmarks[idx2])
                            frame_features.extend(
                                [float(dist_l), float(dist_r), float(dist_inter)])

                # Angles between the 12 points
                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        for k, idx3 in enumerate(key_indices):
                            if i < j < k:
                                angle_l = self._calculate_angle(
                                    left_landmarks[idx1], left_landmarks[idx2], left_landmarks[idx3])
                                angle_r = self._calculate_angle(
                                    right_landmarks[idx1], right_landmarks[idx2], right_landmarks[idx3])
                                frame_features.extend(
                                    [float(angle_l), float(angle_r)])

            features.append(frame_features)

        return features

    def extract_feature_set_5_tips_and_palms_distances_only(self, json_obj):
        '''Feature Set 5:
        Only distances of 12 key points (wrist and fingertips), no angle data.
        C(12,2) = 66 pairwise distances per hand pair combination.
        '''
        frames = json_obj['frames']
        key_indices = [0, 4, 8, 12, 16, 20]
        features = []

        for frame in frames:
            frame_features = []

            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            if left_landmarks is not None and right_landmarks is not None:
                # left
                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        if i < j:
                            dist = np.linalg.norm(
                                left_landmarks[idx1] - left_landmarks[idx2])
                            frame_features.append(float(dist))

                # right
                for i, idx1 in enumerate(key_indices):
                    for j, idx2 in enumerate(key_indices):
                        if i < j:
                            dist = np.linalg.norm(
                                right_landmarks[idx1] - right_landmarks[idx2])
                            frame_features.append(float(dist))

                # all pairs between left and right
                for idx1 in key_indices:
                    for idx2 in key_indices:
                        dist = np.linalg.norm(
                            left_landmarks[idx1] - right_landmarks[idx2])
                        frame_features.append(float(dist))

            features.append(frame_features)

        return features

    def extract_feature_set_6_extreme_minimal(self, json_obj):
        '''Feature Set 6 (Most Extreme/Minimal):
        Distances from index fingers (8) and palms (0) + one distance between pinkie fingertips (20).
        Approximately 163 features per frame × 10 frames = 1630 total.
        '''
        frames = json_obj['frames']
        focus_indices = [0, 8, 20]
        features = []

        for frame in frames:
            frame_features = []

            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            if left_landmarks is not None and right_landmarks is not None:
                # Distances from left index finger (8) to all right hand
                # landmarks
                for i in range(len(right_landmarks)):
                    dist = np.linalg.norm(
                        left_landmarks[8] - right_landmarks[i])
                    frame_features.append(float(dist))

                # Distances from left palm (0) to all right hand landmarks
                for i in range(len(right_landmarks)):
                    dist = np.linalg.norm(
                        left_landmarks[0] - right_landmarks[i])
                    frame_features.append(float(dist))

                # Distances from right index finger (8) to all left hand
                # landmarks
                for i in range(len(left_landmarks)):
                    dist = np.linalg.norm(
                        right_landmarks[8] - left_landmarks[i])
                    frame_features.append(float(dist))

                # Distances from right palm (0) to all left hand landmarks
                for i in range(len(left_landmarks)):
                    dist = np.linalg.norm(
                        right_landmarks[0] - left_landmarks[i])
                    frame_features.append(float(dist))

                # Distance between left and right pinkie fingertips
                # (orientation context)
                pinkie_distance = np.linalg.norm(
                    left_landmarks[20] - right_landmarks[20])
                frame_features.append(float(pinkie_distance))

            features.append(frame_features)

        return features

    def extract_feature_set_0_raw_coordinates(self, json_obj):
        '''Feature Set 0:
        Raw X and Y coordinates for each point.
        21 landmarks per hand × 2 hands × 2 coordinates = 84 features per frame.
        '''
        frames = json_obj['frames']
        features = []

        for frame in frames:
            frame_features = []

            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = [[lm['x'], lm['y']]
                                      for lm in frame['hands']['left']]
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = [[lm['x'], lm['y']]
                                       for lm in frame['hands']['right']]

            # Add left hand coordinates
            if left_landmarks is not None:
                for coord in left_landmarks:
                    frame_features.extend([float(coord[0]), float(coord[1])])
            else:
                frame_features.extend([0.0] * 42)  # 21 points × 2 coordinates

            # Add right hand coordinates
            if right_landmarks is not None:
                for coord in right_landmarks:
                    frame_features.extend([float(coord[0]), float(coord[1])])
            else:
                frame_features.extend([0.0] * 42)  # 21 points × 2 coordinates

            features.append(frame_features)

        return features

    def extract_feature_set_7_interhand_distances_only(self, json_obj):
        '''Feature Set 7:
        Inter-hand distances only (no intra-hand distances).
        Uses 12 key points: wrist(0) and fingertips(4,8,12,16,20) per hand.
        6 × 6 = 36 inter-hand distance features per frame.
        '''
        frames = json_obj['frames']
        key_indices = [0, 4, 8, 12, 16, 20]
        features = []

        for frame in frames:
            frame_features = []

            left_landmarks = None
            right_landmarks = None

            if 'hands' in frame:
                if 'left' in frame['hands'] and frame['hands']['left']:
                    left_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['left']])
                if 'right' in frame['hands'] and frame['hands']['right']:
                    right_landmarks = np.array(
                        [[lm['x'], lm['y']] for lm in frame['hands']['right']])

            if left_landmarks is not None and right_landmarks is not None:
                # All pairs between left and right hands (inter-hand distances
                # only)
                for idx1 in key_indices:
                    for idx2 in key_indices:
                        dist = np.linalg.norm(
                            left_landmarks[idx1] - right_landmarks[idx2])
                        frame_features.append(float(dist))

            features.append(frame_features)

        return features

    def extractAllProximityFeatures(self):
        self.features = []
        for json_obj in self.corpus:
            self.features.append(self.extract_proximity_features(json_obj))
        return self.features

    def extractAllAngleFeatures(self):
        features = []
        for json_obj in self.corpus:
            features.append(self.extract_angle_features(json_obj))
        return features

    def extract_all_feature_set_1(self):
        '''Batch extract Feature Set 1 (Combined + Angles): ~9090 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(self.extract_feature_set_1_combined(json_obj))
        return features

    def extract_all_feature_set_2(self):
        '''Batch extract Feature Set 2 (Proximity only): ~8610 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_2_proximity_only(json_obj))
        return features

    def extract_all_feature_set_3(self):
        '''Batch extract Feature Set 3 (Tips + Palms with angles): Mixed inputs.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_3_tips_and_palms_with_angles(json_obj))
        return features

    def extract_all_feature_set_4(self):
        '''Batch extract Feature Set 4 (12-point angles and distances): Mixed inputs.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_4_tips_and_palms_with_angles(json_obj))
        return features

    def extract_all_feature_set_5(self):
        '''Batch extract Feature Set 5 (12-point distances only): ~660 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_5_tips_and_palms_distances_only(json_obj))
        return features

    def extract_all_feature_set_6(self):
        '''Batch extract Feature Set 6 (Minimal/Extreme): ~1630 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_6_extreme_minimal(json_obj))
        return features

    def extract_all_feature_set_0(self):
        '''Batch extract Feature Set 0 (Raw Coordinates): 84 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_0_raw_coordinates(json_obj))
        return features

    def extract_all_feature_set_7(self):
        '''Batch extract Feature Set 7 (Inter-hand distances only): 144 inputs per sample.'''
        features = []
        for json_obj in self.corpus:
            features.append(
                self.extract_feature_set_7_interhand_distances_only(json_obj))
        return features
