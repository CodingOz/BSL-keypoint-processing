import json
from pathlib import Path
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
import numpy as np
from scipy.ndimage import gaussian_filter1d

class TemporalCropping:
    
    def crop_hands_resting_together(self, json_path, destination_path, show_logs=False):
        '''Crops frames from the start and end of the signing region where both hands are resting together
        using the detectRestPositionFrames to find start and end frames of the signing region 
        reseting the frame indexes to start at 0 and continuously increasing suquentially, and
        updating the metadata with the number of frames cropped from the start and end of the signing region
        
        this anomaly confuses the later 2 part pipeline so it is important to crop these frames out before the stroke phase cropping, 
        and to do it in a way that updates the frame indexes and metadata correctly
        
        takes:
            json_path: the path to the json file to crop
            destination_path: the path to save the cropped json file
            show_logs: whether to print logs of the cropping process'''

        interpolator = CubicSplineKeyPointInterpolator(json_path)
        start_frame, end_frame = interpolator.detectRestPositionFrames()
        
        if show_logs:
            print(f"{json_path}: Cropping frames from {start_frame} to {end_frame} where both hands are resting together.")
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Calculate frames cropped
        frames_cropped_start = start_frame
        original_total_frames = len(data['frames'])
        frames_cropped_end = original_total_frames - end_frame - 1
        
        # Extract cropped frames
        cropped_frames = data['frames'][start_frame:end_frame + 1]
        
        # Get FPS for timestamp recalculation
        fps = data['metadata'].get('fps', 25)
        
        # Reset frame indexes and recalculate timestamps
        for new_index, frame in enumerate(cropped_frames):
            frame['frame_index'] = new_index
            frame['timestamp'] = new_index / fps
        
        # Update metadata
        data['metadata']['frame_count'] = len(cropped_frames)
        data['metadata']['frames_cropped_start'] = frames_cropped_start
        data['metadata']['frames_cropped_end'] = frames_cropped_end
        
        # Update frames
        data['frames'] = cropped_frames
        
        # Ensure destination directory exists
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the cropped JSON file
        with open(destination_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if show_logs:
            print(f"Cropped JSON saved to {destination_path}")
            print(f"Frames cropped from start: {frames_cropped_start}")
            print(f"Frames cropped from end: {frames_cropped_end}")
            print(f"Original frame count: {original_total_frames}")
            print(f"New frame count: {len(cropped_frames)}")
    
    
    def crop_hands_resting_together_in_corpus(self, source_corpus, target_corpus, show_logs=False):
        '''Crops frames from the start and end of the signing region where both hands are resting together
        for all json files in a corpus using the crop_hands_resting_together method
        takes:
            source_corpus: the path to the corpus of json files to crop
            target_corpus: the location where the cropped json files are to be generated to
            show_logs: whether to print logs of the cropping process'''
        
        source_corpus = Path(source_corpus)
        target_corpus = Path(target_corpus)
        target_corpus.mkdir(parents=True, exist_ok=True)

        for json_path in source_corpus.rglob('*.json'):
            # mirror the subdirectory structure in the target corpus
            relative_path = json_path.relative_to(source_corpus)
            output_path   = target_corpus / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.crop_hands_resting_together(str(json_path), str(output_path), show_logs=show_logs) 


    def segment_sign_phases( self,
        palm_distances: np.ndarray,
        momentum_left: np.ndarray,
        momentum_right: np.ndarray,
        closest_distances: np.ndarray,
        *,
        # Stage 1 parameters
        smoothing_sigma: float = 2.5,
        rise_threshold_fraction: float = 0.25,
        min_sign_frames: int = 5,
        # Stage 2 parameters
        composite_halfmax_fraction: float = 0.50,
        refinement_search_window: int = 10,
    ) -> dict:
        """
        Segment a BSL fingerspelling recording into preparation, sign, and
        retraction phases.

        Parameters
        ----------
        palm_distances : array-like, shape (n_frames,)
            Per-frame distance between palm centres (normalised to image coords,
            so values are roughly in [0, 1]).
        momentum_left : array-like, shape (n_frames,)
            Per-frame velocity magnitude of the left palm.
        momentum_right : array-like, shape (n_frames,)
            Per-frame velocity magnitude of the right palm.
        closest_distances : array-like, shape (n_frames,)
            Per-frame minimum pairwise distance between any left and right
            hand keypoint.

        smoothing_sigma : float
            Standard deviation (in frames) of the Gaussian kernel applied to
            the momentum signal in Stage 1.  Increase for noisier recordings.
            Default 2.5.
        rise_threshold_fraction : float  (0 < value < 1)
            Stage 1 coarse boundary.  Walking outward from the valley minimum,
            the boundary is placed at the first frame where the normalised
            momentum rises above  min + rise_threshold_fraction * (max - min).
            Default 0.25 (25 % of the recording's dynamic momentum range).
        min_sign_frames : int
            Minimum acceptable width of the sign phase in frames.  If Stage 1
            or Stage 2 produce a narrower window it is expanded symmetrically
            to this width.  Default 5.
        composite_halfmax_fraction : float  (0 < value < 1)
            Stage 2 refinement.  The refined boundary is the outermost frame
            (within the search window) at which the composite score exceeds
            composite_halfmax_fraction * (local peak composite score).
            Default 0.50 (half-maximum criterion).
        refinement_search_window : int
            Number of frames on each side of the Stage 1 boundary that Stage 2
            is allowed to search.  Default 10.

        Returns
        -------
        dict with keys 'preparation', 'sign', 'retraction', each mapping to a
        (start_frame, end_frame) tuple of inclusive integer frame indices.
        The three phases tile the recording without overlap or gap.

        Raises
        ------
        ValueError
            If the input arrays have inconsistent lengths or the recording is
            too short for meaningful segmentation.
        """


        palm_dist= np.asarray(palm_distances,    dtype=float)
        mom_l= np.asarray(momentum_left,     dtype=float)
        mom_r= np.asarray(momentum_right,    dtype=float)
        closest= np.asarray(closest_distances, dtype=float)

        n = len(mom_l)
        if not (len(mom_r) == len(closest) == n):
            raise ValueError(
                f"All momentum/closest arrays must be the same length. "
                f"Got: momentum_left={len(mom_l)}, momentum_right={len(mom_r)}, "
                f"closest_distances={len(closest)}."
            )
        if n < 2 * min_sign_frames + 2:
            raise ValueError(
                f"Recording has only {n} frames, which is too short to segment. "
                f"Need at least {2 * min_sign_frames + 2} frames."
            )

        # Stage 1 - Coarse boundary detection via momentum valley
        sign_start_coarse, sign_end_coarse = self._stage1_valley_detection(
            mom_l, mom_r,
            smoothing_sigma=smoothing_sigma,
            rise_threshold_fraction=rise_threshold_fraction,
            min_sign_frames=min_sign_frames,
        )

        # Stage 2 - Boundary refinement via composite sign-presence score
        sign_start_refined, sign_end_refined = self._stage2_composite_refinement(
            mom_l, mom_r, closest,
            coarse_start=sign_start_coarse,
            coarse_end=sign_end_coarse,
            composite_halfmax_fraction=composite_halfmax_fraction,
            search_window=refinement_search_window,
            min_sign_frames=min_sign_frames,
        )

        # Assemble output phases
        # Phases tile the recording: [0 … sign_start-1] | [sign_start … sign_end]
        prep_start = 0
        prep_end   = sign_start_refined - 1
        ret_start  = sign_end_refined   + 1
        ret_end    = n - 1

        # Edge case: if the sign extends to frame 0 there is no preparation,
        # and if it extends to the last frame there is no retraction.
        if prep_end < prep_start:
            prep_start = prep_end = 0

        if ret_start > ret_end:
            ret_start = ret_end = n - 1

        return {
            "preparation": (int(prep_start), int(prep_end)),
            "sign":        (int(sign_start_refined), int(sign_end_refined)),
            "retraction":  (int(ret_start), int(ret_end)),
        }


    def _stage1_valley_detection(
        self,
        mom_l: np.ndarray,
        mom_r: np.ndarray,
        smoothing_sigma: float,
        rise_threshold_fraction: float,
        min_sign_frames: int,
    ) -> tuple[int, int]:
        """
        Find the coarse start and end of the sign phase as the deepest sustained
        valley in the smoothed, normalised combined momentum signal.

        Returns:
            (sign_start, sign_end) — inclusive integer frame indices.
        """
        n = len(mom_l)

        # Combine and smooth left and right momentum
        # Take the element-wise maximum so that movement of *either* hand is
        # captured.  (A signer may move mainly one hand during preparation.)
        combined = np.maximum(mom_l, mom_r)
        smoothed = gaussian_filter1d(combined, sigma=smoothing_sigma)

        # Normalise to [0, 1]
        sig_min = smoothed.min()
        sig_max = smoothed.max()
        sig_range = sig_max - sig_min

        if sig_range < 1e-9:
            # Degenerate case: flat signal (constant motion or no motion).
            # Return the full recording as a single "sign" phase.
            half = n // 2
            return (max(0, half - min_sign_frames), min(n - 1, half + min_sign_frames))

        normalised = (smoothed - sig_min) / sig_range   # in [0, 1]

        # find the valley minimum (centre of the sign)
        valley_centre = int(np.argmin(normalised))

        threshold = rise_threshold_fraction
        sign_start = valley_centre
        for i in range(valley_centre, -1, -1):
            if normalised[i] > threshold:
                sign_start = i
                break
        else:
            sign_start = 0 

        sign_end = valley_centre
        for i in range(valley_centre, n):
            if normalised[i] > threshold:
                sign_end = i
                break
        else:
            sign_end = n - 1

        sign_start, sign_end = self._enforce_min_width(
            sign_start, sign_end, min_sign_frames, n
        )

        return sign_start, sign_end


    def _stage2_composite_refinement(
        self,
        mom_l: np.ndarray,
        mom_r: np.ndarray,
        closest: np.ndarray,
        coarse_start: int,
        coarse_end: int,
        composite_halfmax_fraction: float,
        search_window: int,
        min_sign_frames: int,
    ) -> tuple[int, int]:
        """
        Refine the Stage 1 boundaries using a composite sign-presence score.

        The composite score is:
            sign_score(t) = (1 - M_norm(t)) * (1 - C_norm(t))

        Returns
        -------
        (sign_start_refined, sign_end_refined) — inclusive integer frame indices.
        """
        n = len(mom_l)
        
        # Normalise momentum globally within this recording
        combined_mom = np.maximum(mom_l, mom_r)
        mom_min, mom_max = combined_mom.min(), combined_mom.max()
        mom_range = mom_max - mom_min
        if mom_range < 1e-9:
            M_norm = np.zeros(n)
        else:
            M_norm = (combined_mom - mom_min) / mom_range

        # Normalise closest distance globally within this recording
        c_min, c_max = closest.min(), closest.max()
        c_range = c_max - c_min
        if c_range < 1e-9:
            C_norm = np.zeros(n)
        else:
            C_norm = (closest - c_min) / c_range

        composite = (1.0 - M_norm) * (1.0 - C_norm)

        # find peak composite score within the sign window
        sign_region = composite[coarse_start:coarse_end + 1]
        if len(sign_region) == 0:
            return coarse_start, coarse_end

        peak_score = sign_region.max()
        halfmax = composite_halfmax_fraction * peak_score

        if peak_score < 1e-9:
            # Degenerate: composite is zero everywhere in sign region
            return coarse_start, coarse_end

        # Search window spans [coarse_start - search_window, coarse_start + search_window]
        # Walk leftward from coarse_start; refined start is the leftmost frame
        # within the window where composite >= halfmax.
        search_left_start = max(0, coarse_start - search_window)
        search_left_end   = min(n - 1, coarse_start + search_window)

        refined_start = coarse_start
        for i in range(search_left_end, search_left_start - 1, -1):
            if composite[i] >= halfmax:
                refined_start = i
                break

        # Refine end boundary
        search_right_start = max(0, coarse_end - search_window)
        search_right_end   = min(n - 1, coarse_end + search_window)

        refined_end = coarse_end
        for i in range(search_right_start, search_right_end + 1):
            if composite[i] >= halfmax:
                refined_end = i

        if refined_start > refined_end:
            # Refinement crossed — fall back to coarse boundaries
            refined_start, refined_end = coarse_start, coarse_end

        refined_start, refined_end = self._enforce_min_width(
            refined_start, refined_end, min_sign_frames, n
        )

        return refined_start, refined_end


    def _enforce_min_width(self, start: int, end: int, min_frames: int, n: int) -> tuple[int, int]:
        """
        Expand (start, end) symmetrically if the window is narrower than
        min_frames, keeping indices within [0, n-1].
        """
        width = end - start + 1
        if width < min_frames:
            deficit = min_frames - width
            expand_left  = deficit // 2
            expand_right = deficit - expand_left
            start = max(0, start - expand_left)
            end   = min(n - 1, end + expand_right)
        return start, end


    def crop_to_stroke_phase(self, json_path, destination_path, show_logs=False):
        '''Crops frames from the start and end of the signing region to the stroke phase
        saving the spreperation and retraction phases as arrays in the metadata, and 
        updating the frame indexes to start at 0 and continuously increasing suquentially
        takes:
            json_path: the path to the json file to crop
            destination_path: the path to save the cropped json file
            show_logs: whether to print logs of the cropping process'''
        
        validator = CubicSplineKeyPointInterpolator(json_path)
        palm_distances = validator.findPalmDistances()
        closest_distances = validator.find_closest_distances()
        # removes the last frame to match the length of the momentums
        palm_distances = palm_distances[:-1]
        closest_distances = closest_distances[:-1]
        
        momentums = validator.getEstimatedMomentums()
        left_momentum = [momentum['left']['magnitude'] for momentum in momentums]
        right_momentum = [momentum['right']['magnitude'] for momentum in momentums]
        
        phases = self.segment_sign_phases(
            palm_distances,
            left_momentum,
            right_momentum,
            closest_distances)
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        frames = data['frames']
        
        # Extract phase boundaries
        prep_start, prep_end = phases['preparation']
        sign_start, sign_end = phases['sign']
        ret_start, ret_end = phases['retraction']
        
        # Extract frames for each phase
        preparation_frames = frames[prep_start:prep_end + 1]
        stroke_frames = frames[sign_start:sign_end + 1]
        retraction_frames = frames[ret_start:ret_end + 1]
        
        # Get FPS for timestamp recalculation
        fps = data['metadata'].get('fps', 25)
        
        # Reset frame indexes and recalculate timestamps for stroke frames
        for new_index, frame in enumerate(stroke_frames):
            frame['frame_index'] = new_index
            frame['timestamp'] = new_index / fps
        
        # Store preparation and retraction phases in metadata
        data['metadata']['preparation_phase'] = {
            'original_start_frame': prep_start,
            'original_end_frame': prep_end,
            'frame_count': len(preparation_frames),
            'frames': preparation_frames
        }
        data['metadata']['retraction_phase'] = {
            'original_start_frame': ret_start,
            'original_end_frame': ret_end,
            'frame_count': len(retraction_frames),
            'frames': retraction_frames
        }
        data['metadata']['stroke_phase'] = {
            'original_start_frame': sign_start,
            'original_end_frame': sign_end,
            'frame_count': len(stroke_frames)
        }
        
        # Update frame count and frames
        data['metadata']['frame_count'] = len(stroke_frames)
        data['frames'] = stroke_frames
               
        
        # Ensure destination directory exists
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the cropped JSON file
        with open(destination_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if show_logs:
            print(f"Stroke phase extracted and saved to {destination_path}")
            print(f"Preparation frames: {len(preparation_frames)} (frames {prep_start}-{prep_end})")
            print(f"Stroke frames: {len(stroke_frames)} (frames {sign_start}-{sign_end})")
            print(f"Retraction frames: {len(retraction_frames)} (frames {ret_start}-{ret_end})")
            
    def crop_to_stroke_phase_in_corpus(self, source_corpus, target_corpus, show_logs=False):
        '''Crops frames from the start and end of the signing region to the stroke phase
        for all json files in a corpus using the crop_to_stroke_phase method
        takes:
            source_corpus: the path to the corpus of json files to crop
            target_corpus: the location where the cropped json files are to be generated to
            show_logs: whether to print logs of the cropping process'''
        
        source_corpus = Path(source_corpus)
        target_corpus = Path(target_corpus)
        target_corpus.mkdir(parents=True, exist_ok=True)

        for json_path in source_corpus.rglob('*.json'):
            # mirror the subdirectory structure in the target corpus
            relative_path = json_path.relative_to(source_corpus)
            output_path   = target_corpus / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.crop_to_stroke_phase(str(json_path), str(output_path), show_logs=show_logs)

if __name__ == "__main__":
    source= r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus_Temporal_Cropping_level1"
    target= r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus_Temporal_Cropping_level2"
    cropper = TemporalCropping()
    cropper.crop_to_stroke_phase_in_corpus(source, target, show_logs=True)
    