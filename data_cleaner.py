from keypoint_validator import KeyPointValidator, CubicSplineKeyPointInterpolator, KalmanKeyPointEstimator
import json
import copy

class DataCleaner:
    def __init__(self, path):
        self.path = path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.current = json.load(f)
                self.validator = CubicSplineKeyPointInterpolator(path)

        except Exception as e:
            print(f"Failed to read/parse JSON file '{path}': {e}")
            return 1
        

       
       
    # ---------------------- operations on hands ------------------
    
    def deleteHandPoint(self, hand, duplicate=None):
        # hand is tuple (frame, side)
        # duplicate 
        pass
    
    
    
    
    
     
    def manageSimultaneousHands(self):
        duplicate_hands = self.validator.checkForSimultaneousHands()
        missing_hands = self.validator.checkForMissingHands()
        print(duplicate_hands, "\n\n")
        print(missing_hands)
        
        temp = copy.deepcopy(self.current)
        
        
        for hand in duplicate_hands:
            frame_idx, side = hand
            swapped = (frame_idx, 'right' if side == 'left' else 'left')
            if swapped in missing_hands:
                pass
            
                
                
                
        
                

path = r"C:\Users\Oscar Strong\Desktop\finalProgect\KeypointCorpus_unprocessed\B\acf7a090-7ece-488a-a6e8-f4df878629a9.json"

cleaner = DataCleaner(path)
cleaner.manageSimultaneousHands()

