import json
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator, Sigh_lenths
from pathlib import Path


class KeypointInterpolator:
    ''' manages the the interpolation of keypoints'''

    def SimpleCubicSplineKeyPointGenerator(self, json_path, show_logs=False):
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        interpolator = CubicSplineKeyPointInterpolator(json_path)
        frames  = json_data['frames']
        lenths  = interpolator.getSignLenths()

        # fill both hands across the full signing region
        first = lenths.first_hand
        last  = lenths.last_hand

        missing = interpolator.checkForMissingHands()
        missing_left  = {idx for idx, side in missing
                        if side == 'left'  and first <= idx <= last}
        missing_right = {idx for idx, side in missing
                        if side == 'right' and first <= idx <= last}

        filled_keypoints, estimation_flags = interpolator.interpolateFullHands(
            first, last, show_logs=show_logs
        )

        cluster_id = {'left': 0, 'right': 1}

        for i, frame in enumerate(frames):
            if i < first or i > last:
                continue

            for side in ('left', 'right'):
                is_missing = i in (missing_left if side == 'left' else missing_right)
                if not is_missing:
                    continue

                landmarks = [
                    {
                        'cluster_id':  cluster_id[side],
                        'landmark_id': lm_id,
                        'x':           filled_keypoints[i][side][lm_id][0],
                        'y':           filled_keypoints[i][side][lm_id][1],
                        'z':           0.0,
                    }
                    for lm_id in range(21)
                    if filled_keypoints[i][side][lm_id] not in (None, [None, None])
                ]

                if landmarks:
                    frame['hands'][side] = landmarks
                    frame.setdefault('interpolation', [])
                    if side not in frame['interpolation']:
                        frame['interpolation'].append(side)

        return json_data

    def SimpleCubicSplineCorpusGenerator(self, source_corpus, target_corpus):
        '''Uses SimpleCubicSplineKeyPointGenerator to generate a new corpus of interpolated keypoints.
        takes:
            source_corpus: the path to the corpus of uninterpolated keypoints
            target_corpus: the location where the interpolated keypoints are to be generated to
        '''
        source_corpus = Path(source_corpus)
        target_corpus = Path(target_corpus)
        target_corpus.mkdir(parents=True, exist_ok=True)

        processed = 0
        errors    = 0

        for json_path in source_corpus.rglob('*.json'):
            # mirror the subdirectory structure in the target corpus
            relative_path = json_path.relative_to(source_corpus)
            output_path   = target_corpus / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                interpolated = self.SimpleCubicSplineKeyPointGenerator(str(json_path))

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(interpolated, f, indent=2)

                print(f"success: {relative_path}")
                processed += 1

            except Exception as e:
                print(f"failed on: {relative_path}: {e}")
                errors += 1

        print(f"\nDone — {processed} files interpolated, {errors} errors")
            
if __name__ == "__main__":
    
    interpolate = KeypointInterpolator()
    target_corpus = r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus'
    source_corpus = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_SubCorpus"
    interpolate.SimpleCubicSplineCorpusGenerator(source_corpus, target_corpus)