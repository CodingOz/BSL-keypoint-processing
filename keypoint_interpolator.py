import json
from copy import deepcopy
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from pathlib import Path


class KeypointInterpolator:
    ''' manages the the interpolation of keypoints'''

    def simpleCubicSplineKeyPointGenerator(self, json_path, show_logs=False):
        '''interpolates all keypoints of the '''
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        interpolator = CubicSplineKeyPointInterpolator(json_path)
        frames = json_data['frames']
        lenths = interpolator.getSignLengths()

        # fill both hands across the full signing region
        first = lenths.first_hand
        last = lenths.last_hand

        missing = interpolator.checkForMissingHands()
        missing_left = {idx for idx, side in missing
                        if side == 'left' and first <= idx <= last}
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
                is_missing = i in (
                    missing_left if side == 'left' else missing_right)
                if not is_missing:
                    continue

                landmarks = [
                    {
                        'cluster_id': cluster_id[side],
                        'landmark_id': lm_id,
                        'x': filled_keypoints[i][side][lm_id][0],
                        'y': filled_keypoints[i][side][lm_id][1],
                        'z': 0.0,
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

    def simpleCubicSplineCorpusGenerator(self, source_corpus, target_corpus):
        '''Uses simpleCubicSplineKeyPointGenerator to generate a new corpus of interpolated keypoints.
        takes:
            source_corpus: the path to the corpus of uninterpolated keypoints
            target_corpus: the location where the interpolated keypoints are to be generated to
        '''
        source_corpus = Path(source_corpus)
        target_corpus = Path(target_corpus)
        target_corpus.mkdir(parents=True, exist_ok=True)

        processed = 0
        errors = 0

        for json_path in source_corpus.rglob('*.json'):
            # mirror the subdirectory structure in the target corpus
            relative_path = json_path.relative_to(source_corpus)
            output_path = target_corpus / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                interpolated = self.simpleCubicSplineKeyPointGenerator(
                    str(json_path))

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(interpolated, f, indent=2)

                print(f"success: {relative_path}")
                processed += 1

            except Exception as e:
                print(f"failed on: {relative_path}: {e}")
                errors += 1

        print(f"\nDone — {processed} files interpolated, {errors} errors")

    def estimateHandsEnds(
            self,
            json_path,
            target_path=None,
            hand_speed=0.05,
            max_estimates=5,
            show_logs=False):
        '''checks if there are prolonged periods at the start or end of the video where one hand is missing and the other hand is present,
        and if so, estimates at most 5 frames at the start and end of the video where the missing hand is estimated to be present,

        This is does by assuming that a left hand will move slowly left after the last frame where it is present,
        and a right hand will move slowly right after the last frame where it is present, and vice versa for the start of the video.
        '''
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        frames = json_data['frames']
        validator = CubicSplineKeyPointInterpolator(json_path)
        lenths = validator.getSignLengths()

        # looks at does the start first
        gap = lenths.first_with_both - lenths.first_hand
        if show_logs:
            print(f"Gap at the start of the video: {gap} frames.")
        if gap > 0:
            # checks which hand is missing
            left = frames[lenths.first_hand]['hands'].get('left')
            right = frames[lenths.first_hand]['hands'].get('right')
            if left and not right:
                if show_logs:
                    print(f"Right hand is missing at the start of the video.")
                missing = 'right'
                direction = 1
            elif right and not left:
                if show_logs:
                    print(f"Left hand is missing at the start of the video.")
                missing = 'left'
                direction = -1
            last_know = frames[lenths.first_with_both]['hands'].get(missing)
            current = deepcopy(last_know)
            if gap <= max_estimates:
                if show_logs:
                    print(
                        f"Estimating {gap} frames at the start of the video where one hand is missing but the other hand is present.")

                i = lenths.first_with_both - 1
                while i >= lenths.first_hand and gap > 0:

                    for landmark in current:
                        landmark['x'] += hand_speed * direction

                    frames[i]['hands'][missing] = deepcopy(current)

                    i -= 1
                    gap -= 1

            elif gap >= max_estimates:
                stop = lenths.first_with_both - max_estimates  # fixed boundary
                i = lenths.first_with_both - 1
                while i >= stop and gap > 0:
                    for landmark in current:
                        landmark['x'] += hand_speed * direction
                    frames[i]['hands'][missing] = deepcopy(current)
                    i -= 1
                    gap -= 1

        # looks at the end next
        gap = lenths.last_hand - lenths.last_with_both
        if show_logs:
            print(f"Gap at the end of the video: {gap} frames.")
        if gap > 0:
            # checks which hand is missing
            left = frames[lenths.last_hand]['hands'].get('left')
            right = frames[lenths.last_hand]['hands'].get('right')
            if left and not right:
                if show_logs:
                    print(f"Right hand is missing at the end of the video.")
                missing = 'right'
                direction = 1
            elif right and not left:
                if show_logs:
                    print(f"Left hand is missing at the end of the video.")
                missing = 'left'
                direction = -1
            last_know = frames[lenths.last_with_both]['hands'].get(missing)
            current = deepcopy(last_know)
            if gap <= max_estimates:
                if show_logs:
                    print(
                        f"Estimating {gap} frames at the end of the video where one hand is missing but the other hand is present.")

                i = lenths.last_with_both + 1
                while i <= lenths.last_hand and gap > 0:

                    for landmark in current:
                        landmark['x'] += hand_speed * direction

                    frames[i]['hands'][missing] = deepcopy(current)

                    i += 1
                    gap -= 1

            elif gap >= max_estimates:

                stop = lenths.last_with_both + max_estimates  # fixed boundary
                i = lenths.last_with_both + 1
                while i <= stop and gap > 0:
                    for landmark in current:
                        landmark['x'] += hand_speed * direction
                    frames[i]['hands'][missing] = deepcopy(current)
                    i += 1
                    gap -= 1

        if target_path:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
        return json_data

    def estimateHandsEndsCorpusGenerator(
            self,
            source_corpus,
            target_corpus,
            hand_speed=0.05,
            max_estimates=5,
            show_logs=False):
        '''Uses estimateHandsEnds to generate a new corpus of keypoints with estimated hand presence at the start and end of videos.
        takes:
            source_corpus: the path to the corpus of uninterpolated keypoints
            target_corpus: the location where the interpolated keypoints are to be generated to
            hand_speed: the speed at which the hand is estimated to move in x direction (default 0.05)
            max_estimates: the maximum number of frames at the start and end of the video where hand presence is estimated (default 5)
            show_logs: whether to print logs about which hands are missing and how many frames are being estimated (default False)
        '''
        source_corpus = Path(source_corpus)
        target_corpus = Path(target_corpus)
        target_corpus.mkdir(parents=True, exist_ok=True)

        num = sum(1 for _ in source_corpus.rglob('*.json'))
        print(f"starting interpolation of {num} JSON files")

        processed = 0
        errors = 0

        for json_path in source_corpus.rglob('*.json'):
            # mirror the subdirectory structure in the target corpus
            relative_path = json_path.relative_to(source_corpus)
            output_path = target_corpus / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                estimated = self.estimateHandsEnds(
                    str(json_path), show_logs=show_logs)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(estimated, f, indent=2)

                print(f"success: {relative_path}")
                processed += 1

            except Exception as e:
                print(f"failed on: {relative_path}: {e}")
                errors += 1


if __name__ == "__main__":
    interpolate = KeypointInterpolator()
    source_corpus = r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus'
    target_corpus = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_interpolated_SubCorpus_ends_estimated"
    interpolate.estimateHandsEndsCorpusGenerator(
        source_corpus, target_corpus, show_logs=True)
