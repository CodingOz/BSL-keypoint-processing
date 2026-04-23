import numpy as np
from copy import deepcopy


class SignDegradator:

    def __init__(self):
        pass

    def holePunch(self, sign_data, hole_size=5, available_area=None):
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
        sign_data_copy = deepcopy(sign_data)
        frames = sign_data_copy["frames"]

        if not available_area:
            available_area = (0, len(frames))

        start_frame = np.random.randint(
            available_area[0], available_area[1] - hole_size)

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
        modified_frames = frames[:start_frame] + \
            frames[start_frame + hole_size:]
        sign_data_copy["frames"] = modified_frames

        return sign_data_copy, hole_data, info

    def frameSwap(self, sign_data, hole_size=5, available_area=None):
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
        sign_data_copy = deepcopy(sign_data)
        frames = sign_data_copy["frames"]

        if not available_area:
            available_area = (0, len(frames))

        if available_area[1] - hole_size <= available_area[0]:
            available_area = (
                available_area[0],
                available_area[0] + hole_size + 1)

        available_area = (
            max(0, available_area[0]),
            min(len(frames), available_area[1])
        )

        valid_hole_found = False
        max_attempts = 100
        attempts = 0
        best_start_frame = None
        best_frames_with_hands = 0

        while not valid_hole_found and attempts < max_attempts:
            try:
                start_frame = np.random.randint(
                    available_area[0], available_area[1] - hole_size + 1)
            except ValueError:
                start_frame = available_area[0]

            frames_with_hands = 0
            for i in range(
                start_frame, min(
                    start_frame + hole_size, len(frames))):
                if i < len(frames):
                    hands = frames[i].get("hands", {})
                    left_hand = hands.get("left")
                    right_hand = hands.get("right")
                    if (left_hand and len(left_hand) > 0) or (
                            right_hand and len(right_hand) > 0):
                        frames_with_hands += 1

            if frames_with_hands > best_frames_with_hands:
                best_frames_with_hands = frames_with_hands
                best_start_frame = start_frame

            if frames_with_hands > hole_size / 2:
                valid_hole_found = True

            attempts += 1

        if not valid_hole_found:
            start_frame = best_start_frame if best_start_frame is not None else available_area[
                0]
            print(
                f"Warning: Could not find hole with >50% hand coverage after {max_attempts} attempts. Using best found with {best_frames_with_hands}/{hole_size} frames.")
        else:
            start_frame = best_start_frame if best_start_frame is not None else available_area[
                0]

        start_frame = max(0, min(start_frame, len(frames) - hole_size))

        left_hand_frames = []
        right_hand_frames = []

        for i in range(start_frame, min(start_frame + hole_size, len(frames))):
            if i < len(frames):
                hands = frames[i].get("hands", {})
                left_hand = hands.get("left")
                right_hand = hands.get("right")

                if left_hand and len(left_hand) > 0:
                    left_hand_frames.append(i)
                if right_hand and len(right_hand) > 0:
                    right_hand_frames.append(i)

                # IMPORTANT: Always swap, even if one or both hands have no data
                # Swapping None→None or []→[] still counts as degradation in
                # metadata
                hands["left"], hands["right"] = hands.get(
                    "right"), hands.get("left")

        # DEBUG: Log results
        frames_with_data = len(left_hand_frames) + len(right_hand_frames)
        print(
            f"      frameSwap result: {len(left_hand_frames)} left, {len(right_hand_frames)} right frames with data (expected {hole_size})")
        if frames_with_data < hole_size:
            print(
                f"      INFO: Only {frames_with_data}/{hole_size} frames had actual hand data, but all {hole_size} were still swapped")

        info = {
            'size': hole_size,
            'start_frame': start_frame,
            'frames_affected': max(
                len(left_hand_frames),
                len(right_hand_frames)),
            'left_hand_frames': left_hand_frames,
            'right_hand_frames': right_hand_frames,
            'swap_type': 'symmetric'}

        if "testing_changes" not in sign_data_copy["metadata"]:
            sign_data_copy["metadata"]["testing_changes"] = []

        sign_data_copy["metadata"]["testing_changes"].append(info)

        return sign_data_copy

    def singleHandSwap(self, sign_data, hole_size=1, available_area=None,
                       swap_direction=None):
        '''Simulates a realistic MediaPipe error: one hand's data appears in
        the wrong slot while the original slot becomes empty.

        In real MediaPipe failures, the tracker sees one hand, labels it
        incorrectly, and misses the other hand entirely. This creates a
        frame where e.g. the right hand's keypoints appear in the left
        slot and the right slot is empty — NOT a symmetric swap.

        takes:
            sign_data: json sign data with format {"metadata": {...}, "frames": [...]}
            hole_size: number of consecutive frames to corrupt
            available_area: tuple (min_frame, max_frame) restricting placement
            swap_direction: 'left_to_right' or 'right_to_left' or None (random).
                - 'left_to_right': left hand data moves to right slot, left becomes empty
                - 'right_to_left': right hand data moves to left slot, right becomes empty

        returns:
            sign_data_copy: corrupted data with info in metadata["testing_changes"]
        '''
        sign_data_copy = deepcopy(sign_data)
        frames = sign_data_copy["frames"]

        if not available_area:
            available_area = (0, len(frames))

        if available_area[1] - hole_size <= available_area[0]:
            available_area = (
                available_area[0],
                available_area[0] + hole_size + 1)

        available_area = (
            max(0, available_area[0]),
            min(len(frames), available_area[1])
        )

        # DEBUG: Log available area
        print(
            f"      singleHandSwap: available_area={available_area}, hole_size={hole_size}, total_frames={len(frames)}")

        if swap_direction is None:
            left_count = 0
            right_count = 0
            for i in range(
                available_area[0], min(
                    available_area[1], len(frames))):
                hands = frames[i].get("hands", {})
                if hands.get('left') and len(hands['left']) > 0:
                    left_count += 1
                if hands.get('right') and len(hands['right']) > 0:
                    right_count += 1

            if left_count == 0 and right_count == 0:
                swap_direction = np.random.choice(
                    ['left_to_right', 'right_to_left'])
            elif left_count == 0:
                swap_direction = 'right_to_left'
            elif right_count == 0:
                swap_direction = 'left_to_right'
            else:
                swap_direction = 'right_to_left' if np.random.random() < 0.73 else 'left_to_right'

        if swap_direction == 'left_to_right':
            source_side = 'left'
            target_side = 'right'
        else:
            source_side = 'right'
            target_side = 'left'

        # Find a good start frame where the source hand has data
        valid_hole_found = False
        max_attempts = 100
        attempts = 0
        best_start_frame = None
        best_source_coverage = 0

        while not valid_hole_found and attempts < max_attempts:
            try:
                start_frame = np.random.randint(available_area[0], max(
                    available_area[0] + 1, available_area[1] - hole_size + 1))
            except ValueError:
                start_frame = available_area[0]

            # Count frames where the source hand has data
            source_coverage = 0
            for i in range(
                start_frame, min(
                    start_frame + hole_size, len(frames))):
                hands = frames[i].get("hands", {})
                source_data = hands.get(source_side)
                if source_data and len(source_data) > 0:
                    source_coverage += 1

            if source_coverage > best_source_coverage:
                best_source_coverage = source_coverage
                best_start_frame = start_frame

            if source_coverage >= max(1, hole_size // 2):
                valid_hole_found = True

            attempts += 1

        if not valid_hole_found:
            start_frame = best_start_frame if best_start_frame is not None else available_area[
                0]
        else:
            start_frame = best_start_frame if best_start_frame is not None else available_area[
                0]

        start_frame = max(0, min(start_frame, len(frames) - hole_size))

        affected_frames = []

        for i in range(start_frame, min(start_frame + hole_size, len(frames))):
            if i < len(frames):
                hands = frames[i].get("hands", {})
                source_data = hands.get(source_side)

                # IMPORTANT: Always perform the swap, even if source_data is empty or None
                # This ensures consistent degradation across all frames in the range.
                # If source has no data, swapping empty→target and None→empty
                # still counts as degradation.
                affected_frames.append(i)
                # Move source hand data to the wrong slot (may be empty list or
                # None)
                hands[target_side] = source_data
                # Source slot becomes empty (hand "disappeared")
                hands[source_side] = []

        # DEBUG: Log results
        print(
            f"        singleHandSwap result: {len(affected_frames)} frames affected (expected {hole_size})")
        if len(affected_frames) < hole_size:
            print(
                f"        WARNING: Expected {hole_size} affected frames but only got {len(affected_frames)}")

        info = {
            'size': hole_size,
            'start_frame': start_frame,
            'frames_affected': len(affected_frames),
            'left_hand_frames': list(affected_frames),
            'right_hand_frames': list(affected_frames),
            'swap_type': 'single_hand',
            'swap_direction': swap_direction,
        }

        if "testing_changes" not in sign_data_copy["metadata"]:
            sign_data_copy["metadata"]["testing_changes"] = []

        sign_data_copy["metadata"]["testing_changes"].append(info)

        return sign_data_copy


degradator = SignDegradator()
