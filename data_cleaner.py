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
            # anomalous frames stored in left and right frame anomaly lists in metadata
            # for later insertion validator use
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
        
            
        # resets validator and anomaly detector with updated data
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.current, f, indent=2)
        self.validator = CubicSplineKeyPointInterpolator(self.path)
        self.anomaly_detector = AnomalyDetection(self.validator)
        
        # reruns detection until no anomalous frames are found
        # counts down recursive level to prevent infinite loops
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

#path = r"C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed\B\acf7a090-7ece-488a-a6e8-f4df878629a9.json"

#cleaner = DataCleaner(path)

