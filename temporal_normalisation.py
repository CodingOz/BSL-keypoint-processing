import json
import os
from scipy.interpolate import PchipInterpolator
import numpy as np


class TemporalNormalisor:

    def makeNormalisedKeypointFile(self, source_path, target_path, frame_num):
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_frames = data.get('frames', [])
        n_raw = len(raw_frames)

        # NaN where missing — we'll mask these out before fitting PCHIP
        seqs_x = {side: np.full((21, n_raw), np.nan) for side in ('left', 'right')}
        seqs_y = {side: np.full((21, n_raw), np.nan) for side in ('left', 'right')}

        for frame_i, frame in enumerate(raw_frames):
            for side in ('left', 'right'):
                for lm in frame.get('hands', {}).get(side, []) or []:
                    lm_id = lm.get('landmark_id')
                    if lm_id is None or not (0 <= lm_id < 21):
                        continue
                    seqs_x[side][lm_id, frame_i] = lm['x']
                    seqs_y[side][lm_id, frame_i] = lm['y']

        source_t = np.arange(n_raw, dtype=float)
        target_t = np.linspace(0.0, float(n_raw - 1), frame_num)

        resampled_x = {side: np.empty((21, frame_num)) for side in ('left', 'right')}
        resampled_y = {side: np.empty((21, frame_num)) for side in ('left', 'right')}

        for side in ('left', 'right'):
            for lm_id in range(21):
                mask = ~np.isnan(seqs_x[side][lm_id])
                n_valid = int(mask.sum())

                if n_valid >= 2:
                    # PCHIP through the valid frames only; extrapolate to fill ends
                    resampled_x[side][lm_id] = PchipInterpolator(
                        source_t[mask], seqs_x[side][lm_id][mask],
                        extrapolate=True,
                    )(target_t)
                    resampled_y[side][lm_id] = PchipInterpolator(
                        source_t[mask], seqs_y[side][lm_id][mask],
                        extrapolate=True,
                    )(target_t)
                elif n_valid == 1:
                    # Single observation — hold it constant across all output frames
                    idx = np.argmax(mask)
                    resampled_x[side][lm_id] = seqs_x[side][lm_id, idx]
                    resampled_y[side][lm_id] = seqs_y[side][lm_id, idx]
                else:
                    # Hand never seen — fall back to zeros and flag in metadata
                    resampled_x[side][lm_id] = 0.0
                    resampled_y[side][lm_id] = 0.0

        # rest of the function (frames_out, output dict, save) unchanged
        frames_out = []
        for t_idx in range(frame_num):
            frames_out.append({
                'frame_index': t_idx,
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
                'method': 'pchip_uniform_nan_robust',
                'frame_num': frame_num,
                'source_total_frames': n_raw,
                'compression_ratio': round(n_raw / frame_num, 4),
            },
            'frames': frames_out,
        }

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        return output
    
    
    def normaliseCorpus(
            self,
            source_dir,
            target_dir,
            frame_num,
            show_logs=False):
        """
        Normalises all keypoint files in `source_dir` to `frame_num` frames and
        writes them to `target_dir` with the same filenames, preserving nested
        directory structure.

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
        from pathlib import Path
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for source_path in source_dir.rglob('*.json'):
            # Preserve relative directory structure in target
            relative_path = source_path.relative_to(source_dir)
            target_path = target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            self.makeNormalisedKeypointFile(
                str(source_path), str(target_path), frame_num)
            if show_logs:
                print(f"Normalised {relative_path} to {frame_num} frames.")
