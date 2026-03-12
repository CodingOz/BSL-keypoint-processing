
import numpy as np
import json
from copy import deepcopy


class SignDegradator:
    
    def __init__(self):
        pass
    
    def holePunch(self, sign_data, hole_size=5, available_area=None): # available_area is a tuple if (min_frame, max_frame)
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
    
    def frameSwap(self,  sign_data, hole_size=5, available_area=None): # available_area is a tuple if (min_frame, max_frame)):
        '''swaps the left and right hand data for a set number of frames
        
        takes: 
            sign_data: expected as a json dump of the sign data with format:
                {
                    "metadata": {...},
                    "frames": [frame1, frame2, ...]
                }
            hole_size: amount to be switched
            available_area: section of the frames to be tampered with
        
        returns:
            sign_data_copy: hand data with the swapped values (info saved in metadata["testing_changes"])
        '''
        # deep copy to avoid modifying the original
        sign_data_copy = deepcopy(sign_data)
        frames = sign_data_copy["frames"]
        
        if not available_area:
            available_area = (0, len(frames))
        
        # Validate available_area and hole_size
        if available_area[1] - hole_size <= available_area[0]:
            # available_area is too small for the hole_size, adjust
            available_area = (available_area[0], available_area[0] + hole_size + 1)
        
        # Clamp to valid frame range
        available_area = (
            max(0, available_area[0]),
            min(len(frames), available_area[1])
        )
        
        # Keep finding a valid hole with most frames having at least one non-empty hand
        valid_hole_found = False
        max_attempts = 100
        attempts = 0
        best_start_frame = None
        best_frames_with_hands = 0
        
        while not valid_hole_found and attempts < max_attempts:
            try:
                start_frame = np.random.randint(available_area[0], available_area[1] - hole_size + 1)
            except ValueError:
                # available_area too small, pick any valid frame
                start_frame = available_area[0]
            
            # Check how many frames have at least one non-empty hand
            frames_with_hands = 0
            for i in range(start_frame, min(start_frame + hole_size, len(frames))):
                if i < len(frames):
                    hands = frames[i].get("hands", {})
                    left_hand = hands.get("left")
                    right_hand = hands.get("right")
                    if (left_hand and len(left_hand) > 0) or (right_hand and len(right_hand) > 0):
                        frames_with_hands += 1
            
            # Track the best hole found
            if frames_with_hands > best_frames_with_hands:
                best_frames_with_hands = frames_with_hands
                best_start_frame = start_frame
            
            # Consider it valid if most frames have at least one hand (more than half)
            if frames_with_hands > hole_size / 2:
                valid_hole_found = True
            
            attempts += 1
        
        # If no valid hole found, use the best one found
        if not valid_hole_found:
            start_frame = best_start_frame if best_start_frame is not None else available_area[0]
            print(f"Warning: Could not find hole with >50% hand coverage after {max_attempts} attempts. Using best found with {best_frames_with_hands}/{hole_size} frames.")
        else:
            start_frame = best_start_frame if best_start_frame is not None else available_area[0]
        
        # Ensure start_frame is valid
        start_frame = max(0, min(start_frame, len(frames) - hole_size))
        
        # Track non-empty hands
        left_hand_frames = []
        right_hand_frames = []
        
        # swap left and right hand data for the specified frames
        for i in range(start_frame, min(start_frame + hole_size, len(frames))):
            if i < len(frames):
                hands = frames[i].get("hands", {})
                left_hand = hands.get("left")
                right_hand = hands.get("right")
                
                # Track which frames have non-empty hands before swapping
                if left_hand and len(left_hand) > 0:
                    left_hand_frames.append(i)
                if right_hand and len(right_hand) > 0:
                    right_hand_frames.append(i)
                
                # swap the left and right hand data
                hands["left"], hands["right"] = hands.get("right"), hands.get("left")
        
        info = {
            'size': hole_size,
            'start_frame': start_frame,
            'left_hand_frames': left_hand_frames,
            'right_hand_frames': right_hand_frames
        }
        
        # Initialize testing_changes if it doesn't exist
        if "testing_changes" not in sign_data_copy["metadata"]:
            sign_data_copy["metadata"]["testing_changes"] = []
        
        # Append the info to testing_changes
        sign_data_copy["metadata"]["testing_changes"].append(info)
        
        return sign_data_copy

degradator = SignDegradator()