
import numpy as np
import json
from copy import deepcopy

class SignDegradator:
    
    def __init__(self):
        pass
    
    def holePunch(self, sign_data, hole_size=5, available_area=None):# available_area is a tuple if (min_frame, max_frame)
        ''' 
        Removes a section of frames based on the hole size and available area
        returns the modified sign data and the hole data as a tuple (modified_sign_data, hole_data)
        
        takes: 
            sign_data: expected as a json dump of the sign data with format:
                {
                    "metadata": {...},
                    "frames": [frame1, frame2, ...]
                }
            hole_size: amount to be romoved
            avaidible_area: section of the frames to be tampered with
        
        returns:
            sign_data_copy: hand data with the hole removed
            hole_data: the data of the hole removed
            info: metadata bout where the hole was punched
        '''
        # deep copy to avoid modifying the original
        sign_data_copy = deepcopy(sign_data)
        frames = sign_data_copy["frames"]
        
        if not available_area:
            available_area = (0, len(frames))
            
        start_frame = np.random.randint(available_area[0], available_area[1] - hole_size)
        
        # extract the hole data (frames to be removed)
        hole_data = frames[start_frame:start_frame + hole_size]
        
        left_hand_count = 0
        right_hand_count = 0
        
        for frame in hole_data:
            hands = frame.get("hands", {})
            if hands.get("left_hand") and hands["left_hand"]:
                left_hand_count += 1
            if hands.get("right_hand") and hands["right_hand"]:
                right_hand_count += 1
        
        print(f"Of the {hole_size} frames in the hole, {left_hand_count} have left hand keypoints and {right_hand_count} have right hand keypoints.")
        info = {'size': hole_size,
                'start_frame': start_frame}
        # Remove the frames from the data
        modified_frames = frames[:start_frame] + frames[start_frame + hole_size:]
        sign_data_copy["frames"] = modified_frames
    
        return sign_data_copy, hole_data, info
    
    def frameSwap(self,  sign_data, hole_size=5, available_area=None):# available_area is a tuple if (min_frame, max_frame)):
        '''swaps the left and right hand data for a set number of frames
        
        takes: 
            sign_data: expected as a json dump of the sign data with format:
                {
                    "metadata": {...},
                    "frames": [frame1, frame2, ...]
                }
            hole_size: amount to be sitched
            avaidible_area: section of the frames to be tampered with
        
        returns:
            sign_data_copy: hand data with the swapped values
            info: metadata bout where the hole was swapped
        '''
        pass

degradator = SignDegradator()
