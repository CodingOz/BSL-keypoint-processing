import json
import os
import numpy as np


class SpatialNormalisor:
    """
    Applies a shared two-hand spatial normalisation to temporally-normalised
    keypoint files.

    Both steps are applied to ALL 42 landmarks (21 per hand) using a single
    shared reference frame derived from both hands together.  Per-hand
    normalisation is deliberately avoided because BSL letter identity is
    encoded in the RELATIVE geometry between the two hands — applying
    independent per-hand transforms would destroy the inter-hand contact and
    proximity information that distinguishes the signs.

    Reference frame
    ---------------
    Origin : midpoint of both wrists (landmark 0 of each hand).
             Subtracting this removes dependence on where the signer sits in
             frame (translation invariance).
    Scale  : mean of left and right wrist-to-middle-fingertip distances
             (landmark 0 to landmark 12 on each hand).
             Dividing by this removes dependence on camera distance and
             individual hand size, while preserving the ratio between each
             hand's internal geometry and the inter-hand distance (scale
             invariance with shared reference).

    Preconditions
    -------------
    - Source files must have already been temporally normalised so that both
      hands are fully present (21 landmarks each) in every frame.
    - The JSON schema must match the output of TemporalNormalisationKeypointAnalysis:
      { metadata, normalisation, frames: [ { frame_index, timestamp,
        hands: { left: [{landmark_id, x, y}, ...], right: [...] } } ] }
    """
    _WRIST_ID = 0   # landmark 0 — wrist
    _MID_TIP_ID = 12  # landmark 12 — middle fingertip

    def normalise(self, source_path, target_path):
        """
        Apply the full spatial normalisation (translation + scale) to a single
        file and write the result to target_path.

        Args:
            source_path (str | Path): temporally-normalised keypoint JSON.
            target_path (str | Path): path at which to write the result.

        Returns:
            dict: the output structure written to target_path.
        """
        data = self._load(source_path)
        frames_out, audit = self._normalise_frames(data['frames'])
        output = self._build_output(data, frames_out, audit)
        self._save(output, target_path)
        return output

    def normaliseCorpus(self, source_dir, target_dir):
        """
        Apply spatial normalisation to every *.json file in source_dir and
        write the results to target_dir, preserving nested directory structure.

        Args:
            source_dir (str | Path): directory containing temporally-normalised files.
            target_dir (str | Path): directory in which to write normalised files.

        Returns:
            list[dict]: list of (source_path, target_path, success, error) records.
        """
        from pathlib import Path
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        paths = sorted(source_dir.rglob('*.json'))
        if not paths:
            print(
                f"[SpatialNormalisation] No JSON files found in {source_dir}")
            return []

        results = []
        for src in paths:
            # Preserve relative directory structure in target
            relative_path = src.relative_to(source_dir)
            tgt = target_dir / relative_path
            tgt.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.normalise(str(src), str(tgt))
                results.append({'source': str(src), 'target': str(tgt),
                                'success': True, 'error': None})
                print(f"success: {relative_path}")
            except Exception as e:
                results.append({'source': str(src), 'target': str(tgt),
                                'success': False, 'error': str(e)})
                print(f"failed: {relative_path}  —  {e}")

        n_ok = sum(1 for r in results if r['success'])
        n_err = len(results) - n_ok
        print(
            f"\n[SpatialNormalisation] Done: {n_ok} succeeded, {n_err} failed.")
        return results

    def transform(self, source_path, target_path):
        """
        Translation step only: subtract the inter-wrist midpoint from every
        landmark so that the origin sits between the two wrists.

        Args:
            source_path (str | Path): temporally-normalised keypoint JSON.
            target_path (str | Path): path at which to write the result.

        Returns:
            dict: the output structure written to target_path.
        """
        data = self._load(source_path)
        frames_out, audit = self._translate_frames(data['frames'])
        output = self._build_output(data, frames_out, audit)
        self._save(output, target_path)
        return output

    def scale(self, source_path, target_path):
        """
        Scale step only: divide all landmarks by the shared mean hand size.
        Intended to be called on the output of transform(), not a raw file.

        Args:
            source_path (str | Path): translated keypoint JSON.
            target_path (str | Path): path at which to write the result.

        Returns:
            dict: the output structure written to target_path.
        """
        data = self._load(source_path)
        frames_out, audit = self._scale_frames(data['frames'])
        output = self._build_output(data, frames_out, audit)
        self._save(output, target_path)
        return output

    # ── private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _load(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _save(data, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _index_landmarks(frame):
        """
        Returns (left_dict, right_dict) where each dict maps landmark_id → lm.
        """
        return (
            {lm['landmark_id']: lm for lm in frame['hands']['left']},
            {lm['landmark_id']: lm for lm in frame['hands']['right']},
        )

    def _wrist_and_scale(self, left_lms, right_lms):
        """
        Compute origin (inter-wrist midpoint) and shared scale for one frame.

        Returns:
            origin
            shared_scale
            scale_left
            scale_right
        """
        W = self._WRIST_ID
        M = self._MID_TIP_ID

        left_wrist = np.array([left_lms[W]['x'], left_lms[W]['y']])
        right_wrist = np.array([right_lms[W]['x'], right_lms[W]['y']])
        left_mid = np.array([left_lms[M]['x'], left_lms[M]['y']])
        right_mid = np.array([right_lms[M]['x'], right_lms[M]['y']])

        origin = (left_wrist + right_wrist) / 2.0
        scale_left = float(np.linalg.norm(left_mid - left_wrist))
        scale_right = float(np.linalg.norm(right_mid - right_wrist))
        shared_scale = (scale_left + scale_right) / 2.0

        return origin, shared_scale, scale_left, scale_right

    def _normalise_frames(self, raw_frames):
        """
        Apply translation + scale in a single pass (most efficient).
        Returns (frames_out, audit_dict).
        """
        frames_out = []
        frame_scales = []

        for frame in raw_frames:
            left_lms, right_lms = self._index_landmarks(frame)
            origin, shared_scale, sl, sr = self._wrist_and_scale(
                left_lms, right_lms
            )
            frame_scales.append(shared_scale)

            frame_out = {
                'frame_index': frame['frame_index'],
                'timestamp': frame['timestamp'],
                'hands': {'left': [], 'right': []}
            }

            for side, lms_dict in (('left', left_lms), ('right', right_lms)):
                for lm_id in range(21):
                    lm = lms_dict[lm_id]
                    p = np.array([lm['x'], lm['y']])
                    p_norm = (p - origin) / shared_scale
                    frame_out['hands'][side].append({
                        'landmark_id': lm_id,
                        'x': round(float(p_norm[0]), 8),
                        'y': round(float(p_norm[1]), 8),
                    })

            frames_out.append(frame_out)

        audit = {
            'method': 'inter_wrist_midpoint_shared_scale',
            'origin': 'midpoint of left and right wrist (landmark 0)',
            'scale': 'mean of left and right wrist-to-middle-fingertip '
            '(landmark 12) distances; single shared divisor for '
            'both hands to preserve inter-hand geometry',
            'mean_frame_scale': round(float(np.mean(frame_scales)), 6),
            'min_frame_scale': round(float(np.min(frame_scales)), 6),
            'max_frame_scale': round(float(np.max(frame_scales)), 6),
        }
        return frames_out, audit

    def _translate_frames(self, raw_frames):
        """Translation step only — subtract inter-wrist midpoint."""
        frames_out = []
        for frame in raw_frames:
            left_lms, right_lms = self._index_landmarks(frame)
            origin, _, _, _ = self._wrist_and_scale(left_lms, right_lms)

            frame_out = {
                'frame_index': frame['frame_index'],
                'timestamp': frame['timestamp'],
                'hands': {'left': [], 'right': []}
            }
            for side, lms_dict in (('left', left_lms), ('right', right_lms)):
                for lm_id in range(21):
                    lm = lms_dict[lm_id]
                    p = np.array([lm['x'], lm['y']]) - origin
                    frame_out['hands'][side].append({
                        'landmark_id': lm_id,
                        'x': round(float(p[0]), 8),
                        'y': round(float(p[1]), 8),
                    })
            frames_out.append(frame_out)

        audit = {'method': 'translation_only',
                 'origin': 'midpoint of left and right wrist (landmark 0)'}
        return frames_out, audit

    def _scale_frames(self, raw_frames):
        """
        Scale step only — divide by shared hand size.
        Expects frames that have already been translated (wrists symmetric
        around zero), but works correctly on raw frames too.
        """
        frames_out = []
        frame_scales = []

        for frame in raw_frames:
            left_lms, right_lms = self._index_landmarks(frame)
            _, shared_scale, _, _ = self._wrist_and_scale(left_lms, right_lms)
            frame_scales.append(shared_scale)

            frame_out = {
                'frame_index': frame['frame_index'],
                'timestamp': frame['timestamp'],
                'hands': {'left': [], 'right': []}
            }
            for side, lms_dict in (('left', left_lms), ('right', right_lms)):
                for lm_id in range(21):
                    lm = lms_dict[lm_id]
                    frame_out['hands'][side].append({
                        'landmark_id': lm_id,
                        'x': round(lm['x'] / shared_scale, 8),
                        'y': round(lm['y'] / shared_scale, 8),
                    })
            frames_out.append(frame_out)

        audit = {
            'method': 'scale_only',
            'scale': 'mean of wrist-to-middle-fingertip distances',
            'mean_frame_scale': round(float(np.mean(frame_scales)), 6),
            'min_frame_scale': round(float(np.min(frame_scales)), 6),
            'max_frame_scale': round(float(np.max(frame_scales)), 6),
        }
        return frames_out, audit

    @staticmethod
    def _build_output(source_data, frames_out, spatial_audit):
        """
        Assemble the output dict, preserving all upstream audit blocks.
        """
        output = {
            'metadata': source_data.get('metadata', {}),
            'normalisation': source_data.get('normalisation', {}),
            'spatial_normalisation': spatial_audit,
            'frames': frames_out,
        }
        # If the source already had a spatial_normalisation block, stack it
        # so the history is never silently overwritten.
        existing = source_data.get('spatial_normalisation')
        if existing:
            output['spatial_normalisation_previous'] = existing
        return output
