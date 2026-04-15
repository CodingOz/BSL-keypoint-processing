import numpy as np
import json

class FeatureExraction:
    def __init__(self, corpus_path, feature_path):
        self.corpus_path = corpus_path
        self.feature_path = feature_path
        
        # holds all jsons in the corpus
        self.corpus = []
        self.load_corpus()
    
    def load_corpus(self):
        with open(self.corpus_path, 'r') as f:
            for line in f:
                self.corpus.append(json.loads(line))
                
    def extract_proximity_features(self, json_obj, include_labels=False, include_palms=False):
        '''Extract proximity features from a single JSON object.
        there are C(2, 42) = 861 pairs of proximitys per frame,
        C(2, 44) = 946 pairs of proximitys per frame if include_palms is True,
        there are 10 frames per video, so there are 8610 proximity features per video.
        
        returns: a numpy array of all proximity features for the given JSON object. 
        shape:
            if include_labels are False: (8610,) of floats representing the euclidean distances for each pair of points
            if include_labels is True: (8610, 3) where the second dimension contains 
                labels of the two joints in the pair and the frame index.
        '''
        frames = json_obj['frames']
        proximity_features = []
        
        for frame_idx, frame in enumerate(frames):
            # Extract all hand landmarks for this frame
            landmarks = []
            
            # Left hand landmarks
            if 'hands' in frame and 'left' in frame['hands']:
                for landmark in frame['hands']['left']:
                    landmarks.append([landmark['x'], landmark['y']])
            
            # Right hand landmarks
            if 'hands' in frame and 'right' in frame['hands']:
                for landmark in frame['hands']['right']:
                    landmarks.append([landmark['x'], landmark['y']])
            
            if not landmarks:
                continue
                
            landmarks = np.array(landmarks)
            
            # Calculate all pairwise distances
            num_landmarks = landmarks.shape[0]
            for i in range(num_landmarks):
                for j in range(i + 1, num_landmarks):
                    distance = np.linalg.norm(landmarks[i] - landmarks[j])
                    if include_labels:
                        proximity_features.append([distance, i, j, frame_idx])
                    else:
                        proximity_features.append(distance)
        
        return np.array(proximity_features) 
                
    def extract_all_proximity_features(self):
        self.features = []
        for json_obj in self.corpus:
            self.features.append(self.extract_proximity_features(json_obj))
        return self.features