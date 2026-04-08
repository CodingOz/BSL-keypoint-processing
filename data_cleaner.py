import os

from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from anomaly_detection import AnomalyDetection
import json
import copy

class DataCleaner:
    '''
    Class that handles the data cleaning pipeline
    '''
    def __init__(self, path):
        self.path = path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{path}': {e}")
            return 1
        self.frames = self.data.get('frames', [])
        self.validator = CubicSplineKeyPointInterpolator(path)
        self.anomaly_detector = AnomalyDetection(self.validator)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.current = json.load(f)
                self.versions = [copy.deepcopy(self.current)]
        except Exception as e:
            print(f"Failed to read/parse json file '{path}': {e}")            

    def detectAnomalousFrames(self, recursive_level=0, save_versions=False, show_logs=False):
        '''Uses anomaly_detection to detect frames with anomalous hand keypoint data
        then places these in metadata and reruns detection until non are found
        
        takes:
            recursive_level: number of recursions left to run (to prevent infinite loops)
            save_versions: whether to save versions of the data at each recursive level for later comparison
            show_logs: whether to print logs of the cleaning process
        returns:
            tuple of lists: (anomaly_frames_left, anomaly_frames_right)
        '''
        anomaly_frames_left, anomaly_frames_right = self.anomaly_detector.posisionAndFilledMovmentAnomalys(position_threshold=-0.1, 
                                                               movement_threshole=0.1, 
                                                               gap_size=5)
        if show_logs:
            print(f"Recursive level {recursive_level}: Detected {len(anomaly_frames_left)} anomalous frames in left hand and {len(anomaly_frames_right)} in right hand.")
            
        if len(anomaly_frames_left) == 0 and len(anomaly_frames_right) == 0:
            if show_logs:
                print("No anomalous frames detected by recursive level", recursive_level)
            return anomaly_frames_left, anomaly_frames_right
        
        else:
            for frame in anomaly_frames_left:
                self.current['metadata'].setdefault('left_anomalous_frames', []).append(self.current['frames'][frame])
                # remove anomalous hand keypoints from main dataset
                self.current['frames'][frame]['hands']['left'] = []
                            
            for frame in anomaly_frames_right:
                self.current['metadata'].setdefault('right_anomalous_frames', []).append(self.current['frames'][frame])
                # remove anomalous hand keypoints from main dataset
                self.current['frames'][frame]['hands']['right'] = []
        
        if save_versions:
            # saves version of data with anomalous frames removed to versions list
            self.versions.append(copy.deepcopy(self.current))
        
            
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.current, f, indent=2)
        self.validator = CubicSplineKeyPointInterpolator(self.path)
        self.anomaly_detector = AnomalyDetection(self.validator)
        
        if recursive_level > 0:
            recursive_level -= 1
            
            left, right = self.detectAnomalousFrames(
                recursive_level=recursive_level, 
                save_versions=save_versions, show_logs=show_logs)
            
            return anomaly_frames_left+left, anomaly_frames_right+right
            
        else:
            return anomaly_frames_left, anomaly_frames_right

    def getAllMetadata(self):
        '''returns all metadata from the current data'''
        return self.current.get('metadata', {})

    def detectDoubleHandAnomalousFrames(self, show_logs=False):
        """saves all Simultaneous hand anomalys
        removes and saves them to simultaneous_hands_frames in metadata """
        
        # insure json is up to date
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{self.path}': {e}")
            return 1
        self.frames = self.data.get('frames', [])
        
        cases = self.validator.checkForSimultaneousHands(show_logs=show_logs)

        for case in cases:
            frame_index, side = case
            
            frame = copy.deepcopy(self.frames[frame_index])
            self.data['frames'][frame_index]['hands'][side] = []
            self.data['metadata'].setdefault('simultaneous_hands_frames', []).append(frame)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def undoDetectDoubleHandAnomalousFrames(self, show_logs=False):
        """reverces the proccess of detectDoubleHandAnomalousFrames by reinserting the anomalous frames back into the main dataset and removing them from metadata
        
        this helps to valiate that all changes are trachible and reversible and that no data is lost in the cleaning process
        """
        # insure json is up to date
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{self.path}': {e}")
            return 1
        
        # Get saved simultaneous hands frames from metadata
        simultaneous_frames = self.data['metadata'].get('simultaneous_hands_frames', [])
        
        if not simultaneous_frames:
            if show_logs:
                print("No simultaneous hands frames found in metadata to restore")
            return 0
        
        restored_count = 0
        
        # Restore each saved frame back to the main dataset
        for saved_frame in simultaneous_frames:
            frame_index = None
            if 'frame_id' in saved_frame:
                for idx, frame in enumerate(self.data['frames']):
                    if frame.get('frame_id') == saved_frame['frame_id']:
                        frame_index = idx
                        break
            
            # If not found, try by frame_index metadata
            if frame_index is None and 'frame_index' in saved_frame:
                frame_index = saved_frame['frame_index']
            
            # Restore the hands data
            if frame_index is not None and frame_index < len(self.data['frames']):
                if 'hands' in saved_frame:
                    self.data['frames'][frame_index]['hands'] = saved_frame['hands']
                    restored_count += 1
        
        if 'simultaneous_hands_frames' in self.data['metadata']:
            del self.data['metadata']['simultaneous_hands_frames']
        
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
        
        if show_logs:
            print(f"Restored {restored_count} frames with simultaneous hand anomalies")
        return 0
        
            