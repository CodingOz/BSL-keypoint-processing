import json
import os
from scipy.interpolate import PchipInterpolator
import numpy as np

class TemporalNormalisor:

    def MakeNormalisedKeypointFile(self, source_path, target_path, frame_num):
        """
        Normalises a keypoint file to exactly `frame_num` frames using PCHIP
        interpolation along each landmark's 2-D trajectory.
        
        takes:
            source_path : str | Path
                Path to the cleaned keypoint JSON.
            target_path : str | Path
                Path at which to write the normalised JSON.
            frame_num : int
                Number of output frames

        Returns:
            dict: the output structure written to target_path.
        """
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_frames = data.get('frames', [])
        n_raw = len(raw_frames)

        # Extract per-landmark (x, y) sequences
        seqs_x = {side: np.empty((21, n_raw)) for side in ('left', 'right')}
        seqs_y = {side: np.empty((21, n_raw)) for side in ('left', 'right')}

        for frame_i, frame in enumerate(raw_frames):
            for side in ('left', 'right'):
                # Index landmarks by id for O(1) lookup regardless of list order
                lm_by_id = {
                    lm['landmark_id']: lm
                    for lm in frame.get('hands', {}).get(side, [])
                    if 'landmark_id' in lm
                }
                for lm_id in range(21):
                    lm = lm_by_id[lm_id]
                    seqs_x[side][lm_id, frame_i] = lm['x']
                    seqs_y[side][lm_id, frame_i] = lm['y']

        # PCHIP resampling
        source_t = np.arange(n_raw, dtype=float)
        target_t = np.linspace(0.0, float(n_raw - 1), frame_num)

        # resampled_x/y[side][lm_id] → 1-D array of length frame_num
        resampled_x = {side: np.empty((21, frame_num)) for side in ('left', 'right')}
        resampled_y = {side: np.empty((21, frame_num)) for side in ('left', 'right')}

        for side in ('left', 'right'):
            for lm_id in range(21):
                resampled_x[side][lm_id] = PchipInterpolator(
                    source_t, seqs_x[side][lm_id]
                )(target_t)
                resampled_y[side][lm_id] = PchipInterpolator(
                    source_t, seqs_y[side][lm_id]
                )(target_t)

        # Assemble output JSON structure
        frames_out = []
        for t_idx in range(frame_num):
            frames_out.append({
                'frame_index': t_idx,
                # 0.0 on first frame, 1.0 on last — position-in-sign feature
                # that is independent of frame_num choice
                'timestamp': round(t_idx / max(frame_num - 1, 1), 8),
                'hands': {
                    side: [
                        {
                            'landmark_id': lm_id,
                            'x': round(float(resampled_x[side][lm_id, t_idx]), 8),
                            'y': round(float(resampled_y[side][lm_id, t_idx]), 8),
                        }
                        for lm_id in range(21)
                    ]
                    for side in ('left', 'right')
                }
            })

        output = {
            'metadata': data.get('metadata', {}),
            'normalisation': {
                'method':              'pchip_uniform',
                'frame_num':           frame_num,
                'source_total_frames': n_raw,
                # >1.0 → downsampled; <1.0 → upsampled (more interpolation)
                'compression_ratio':   round(n_raw / frame_num, 4),
            },
            'frames': frames_out,
        }

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        return output
    
    def NormaliseCorpus(self, source_dir, target_dir, frame_num, show_logs=False):
        """
        Normalises all keypoint files in `source_dir` to `frame_num` frames and
        writes them to `target_dir` with the same filenames.

        takes:
            source_dir : str | Path
                Directory containing cleaned keypoint JSON files.
            target_dir : str | Path
                Directory at which to write normalised JSON files.
            frame_num : int
                Number of output frames for each file.
            show_logs : bool, optional
                If True, prints progress logs to console.
            """
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(source_dir):
            if filename.endswith('.json'):
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                self.MakeNormalisedKeypointFile(source_path, target_path, frame_num)
                if show_logs:
                    print(f"Normalised {filename} to {frame_num} frames.")
        