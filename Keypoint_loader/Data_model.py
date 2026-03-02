import numpy as np
import json
import os
import sys

# Add parent directory to path to import KeyPointValidator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keypoint_validator import KeyPointValidator, KalmanKeyPointEstimator, CubicSplineKeyPointInterpolator

class DataModel:
    def __init__(self):
        self.frames = []
        self.loaded_files = {}
        self.file_frame_counts = {}  # Track frame count for each file
        self.max_frames = 0
        self.palm_mode = None
        self.color_palette = [
            'r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange', 'purple', 
            'brown', 'pink', 'lime', 'navy', 'teal', 'olive'
        ]
        self.next_color_idx = 0

    def load_from_json(self, json_path, add=True, load_palms=True):
        """load json file and add or remove its data from frames"""
        filename = os.path.basename(json_path)
        
        if not add:
            # Remove file data
            self._remove_file_data(json_path)
            if filename in self.loaded_files:
                del self.loaded_files[filename]
            return
        
        # Load new file
        if filename in self.loaded_files:
            return
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Assign color to this file
        color = self.color_palette[self.next_color_idx % len(self.color_palette)]
        self.loaded_files[filename] = color
        self.next_color_idx += 1
        
        # Initialize frames list if empty
        num_frames = len(data["frames"])
        # Track frame count for this file
        self.file_frame_counts[filename] = num_frames
        if not self.frames:
            self.frames = [{} for _ in range(num_frames)]
            self.max_frames = num_frames
        else:
            # Extend frames list if this file has more frames than current max
            if num_frames > len(self.frames):
                self.frames.extend([{} for _ in range(num_frames - len(self.frames))])

            self.max_frames = max(self.max_frames, num_frames)
        
        # Add data to existing frames
        for frame_idx, frame_data in enumerate(data["frames"]):
            
            if "points_by_file" not in self.frames[frame_idx]:
                self.frames[frame_idx]["points_by_file"] = {}
                self.frames[frame_idx]["meta"] = {
                    "frame_index": frame_data["frame_index"],
                    "timestamp": frame_data["timestamp"]
                }
            
            points_by_cluster = {"left": [], "right": [], "pose": []}
            metadata_by_cluster = {"left": [], "right": [], "pose": []}
            
            # pose
            for lm in frame_data.get("pose", []):
                points_by_cluster["pose"].append([lm["x"], lm["y"]])
                metadata_by_cluster["pose"].append({
                    "cluster_id": lm.get("cluster_id"),
                    "landmark_id": lm.get("landmark_id"),
                    "x": lm.get("x"),
                    "y": lm.get("y"),
                    "z": lm.get("z"),
                    "visibility": lm.get("visibility")
                })
            
            # hands
            hands = frame_data.get("hands", {})
            for lm in hands.get("left", []):
                points_by_cluster["left"].append([lm["x"], lm["y"]])
                metadata_by_cluster["left"].append({
                    "cluster_id": lm.get("cluster_id"),
                    "landmark_id": lm.get("landmark_id"),
                    "x": lm.get("x"),
                    "y": lm.get("y"),
                    "z": lm.get("z")
                })
            
            for lm in hands.get("right", []):
                points_by_cluster["right"].append([lm["x"], lm["y"]])
                metadata_by_cluster["right"].append({
                    "cluster_id": lm.get("cluster_id"),
                    "landmark_id": lm.get("landmark_id"),
                    "x": lm.get("x"),
                    "y": lm.get("y"),
                    "z": lm.get("z")
                })
            
            self.frames[frame_idx]["points_by_file"][filename] = {
                "points_by_cluster": {
                    "left": self._to_2d(points_by_cluster["left"]),
                    "right": self._to_2d(points_by_cluster["right"]),
                    "pose": self._to_2d(points_by_cluster["pose"])
                },
                "metadata_by_cluster": metadata_by_cluster,
                "color": color
            }
        
        # load palm centers if requested
        if load_palms:
            try:
                validator = KeyPointValidator(json_path)
                self._load_palm_centers(validator, filename, color)
            except Exception as e:
                print(f"Warning: Could not load palm centers from {filename}: {e}")

    def _load_palm_centers(self, validator, filename, color):
        """load real palm centers from validator"""
        try:
            palm_centers = validator.getPalmCenters()
            
            for frame_idx, palms in enumerate(palm_centers):
                if frame_idx >= len(self.frames):
                    break
                
                if "palms_by_file" not in self.frames[frame_idx]:
                    self.frames[frame_idx]["palms_by_file"] = {}
                
                if filename not in self.frames[frame_idx]["palms_by_file"]:
                    self.frames[frame_idx]["palms_by_file"][filename] = {"color": color}
                
                palm_points = []
                palm_metadata = []
                
                for side in ['left', 'right']:
                    if side in palms and palms[side] != [None, None]:
                        palm_points.append(palms[side])
                        palm_metadata.append({'side': side, 'x': palms[side][0], 'y': palms[side][1]})
                
                # Merge with existing data, preserving color and estimated palms if they exist
                self.frames[frame_idx]["palms_by_file"][filename]["real_palms"] = self._to_2d(palm_points)
                self.frames[frame_idx]["palms_by_file"][filename]["palm_metadata"] = palm_metadata
        except Exception as e:
            print(f"error loading palm centers: {e}")
    
    def _load_estimated_palm_centers(self, filepath, filename, color):
        """load estimated palm centers from KalmanKeyPointEstimator"""
        try:
            estimator = KalmanKeyPointEstimator(filepath)
            filled_palms, estimation_flags = estimator.estimateMissingPalmCenters()
            
            for frame_idx, palms in enumerate(filled_palms):
                # extend frames if needed to accommodate this file's length
                while frame_idx >= len(self.frames):
                    self.frames.append({})
                
                if "palms_by_file" not in self.frames[frame_idx]:
                    self.frames[frame_idx]["palms_by_file"] = {}
                
                palm_points = []
                palm_metadata = []
                
                for side in ['left', 'right']:
                    if side in palms and palms[side] != [None, None]:
                        palm_points.append(palms[side])
                        palm_metadata.append({
                            'side': side,
                            'x': palms[side][0],
                            'y': palms[side][1],
                            'estimated': estimation_flags[frame_idx].get(side, False)
                        })
                
                if filename not in self.frames[frame_idx]["palms_by_file"]:
                    self.frames[frame_idx]["palms_by_file"][filename] = {"color": color}
                
                # merge with existing data, preserving color and real palms if they exist
                self.frames[frame_idx]["palms_by_file"][filename]["estimated_palms"] = self._to_2d(palm_points)
                self.frames[frame_idx]["palms_by_file"][filename]["estimated_palm_metadata"] = palm_metadata
            
            # update max_frames if this file has more frames
            if len(filled_palms) > self.max_frames:
                self.max_frames = len(filled_palms)
        except Exception as e:
            print(f"error loading estimated palm centers: {e}")
    
    def set_palm_mode(self, mode):
        """set palm display mode: None, 'real', 'kalman', 'cubic_spline', or 'pchip'"""
        self.palm_mode = mode
    
    def get_palm_mode(self):
        """get current palm display mode"""
        return self.palm_mode
    
    def load_palm_centers_by_type(self, filepath, filename, palm_type, color):
        """load palm centers by type: 'real', 'kalman', 'cubic_spline', or 'pchip'"""
        if palm_type == 'real':
            validator = KeyPointValidator(filepath)
            self._load_palm_centers(validator, filename, color)
        elif palm_type == 'kalman':
            estimator = KalmanKeyPointEstimator(filepath)
            self._load_kalman_palm_centers(estimator, filename, color)
        elif palm_type == 'cubic_spline':
            interpolator = CubicSplineKeyPointInterpolator(filepath, method='cubic')
            self._load_spline_palm_centers(interpolator, filename, color, 'cubic')
        elif palm_type == 'pchip':
            interpolator = CubicSplineKeyPointInterpolator(filepath, method='pchip')
            self._load_spline_palm_centers(interpolator, filename, color, 'pchip')
    
    def _load_kalman_palm_centers(self, estimator, filename, color):
        """load Kalman-estimated palm centers"""
        try:
            estimator.findAllPalmCenters()
            filled_palms, estimation_flags = estimator.estimateMissingPalmCenters()
            
            for frame_idx, palms in enumerate(filled_palms):
                # extend frames if needed
                while frame_idx >= len(self.frames):
                    self.frames.append({})
                
                if "palms_by_file" not in self.frames[frame_idx]:
                    self.frames[frame_idx]["palms_by_file"] = {}
                
                if filename not in self.frames[frame_idx]["palms_by_file"]:
                    self.frames[frame_idx]["palms_by_file"][filename] = {"color": color}
                
                palm_points = []
                palm_metadata = []
                
                for side in ['left', 'right']:
                    if side in palms and palms[side] != [None, None]:
                        palm_points.append(palms[side])
                        palm_metadata.append({
                            'side': side,
                            'x': palms[side][0],
                            'y': palms[side][1],
                            'estimated': estimation_flags[frame_idx].get(side, False)
                        })
                
                self.frames[frame_idx]["palms_by_file"][filename]["kalman_palms"] = self._to_2d(palm_points)
                self.frames[frame_idx]["palms_by_file"][filename]["kalman_metadata"] = palm_metadata
            
            if len(filled_palms) > self.max_frames:
                self.max_frames = len(filled_palms)
        except Exception as e:
            print(f"error loading Kalman palm centers: {e}")
    
    def _load_spline_palm_centers(self, interpolator, filename, color, method='cubic'):
        """load spline-interpolated palm centers (cubic or PCHIP)"""
        try:
            filled_palms, estimation_flags = interpolator.getFilledPalmCenters()
            method_key = f"{method}_spline_palms"
            method_meta_key = f"{method}_spline_metadata"
            
            for frame_idx, palms in enumerate(filled_palms):
                # extend frames if needed
                while frame_idx >= len(self.frames):
                    self.frames.append({})
                
                if "palms_by_file" not in self.frames[frame_idx]:
                    self.frames[frame_idx]["palms_by_file"] = {}
                
                if filename not in self.frames[frame_idx]["palms_by_file"]:
                    self.frames[frame_idx]["palms_by_file"][filename] = {"color": color}
                
                palm_points = []
                palm_metadata = []
                
                for side in ['left', 'right']:
                    if side in palms and palms[side] != [None, None]:
                        palm_points.append(palms[side])
                        palm_metadata.append({
                            'side': side,
                            'x': palms[side][0],
                            'y': palms[side][1],
                            'estimated': estimation_flags[frame_idx].get(side, False)
                        })
                
                self.frames[frame_idx]["palms_by_file"][filename][method_key] = self._to_2d(palm_points)
                self.frames[frame_idx]["palms_by_file"][filename][method_meta_key] = palm_metadata
            
            if len(filled_palms) > self.max_frames:
                self.max_frames = len(filled_palms)
        except Exception as e:
            print(f"Error loading {method.upper()} spline palm centers: {e}")

    def _remove_file_data(self, json_path):
        """remove data from a specific file"""
        filename = os.path.basename(json_path)
        
        for frame in self.frames:
            if "points_by_file" in frame and filename in frame["points_by_file"]:
                del frame["points_by_file"][filename]
            if "palms_by_file" in frame and filename in frame["palms_by_file"]:
                del frame["palms_by_file"][filename]
        
        # Remove frame count for this file and recalculate max_frames
        if filename in self.file_frame_counts:
            del self.file_frame_counts[filename]
        
        # Recalculate max_frames based on remaining loaded files
        if self.file_frame_counts:
            self.max_frames = max(self.file_frame_counts.values())
        else:
            self.max_frames = 0
            self.frames = []

    def _to_2d(self, points):
        arr = np.array(points, dtype=float)
        if arr.size == 0:
            return arr.reshape(0, 2)
        return arr.reshape(-1, 2)
    
    def frame_count(self):
        return self.max_frames

    def get_frame(self, index):
        if index >= len(self.frames):
            return {}
        return self.frames[index]
    
    def get_loaded_files(self):
        return list(self.loaded_files.keys())