import os
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from anomaly_detection import AnomalyDetection
import json
import copy


class DataCleaner:
    '''
    Class that handles the data cleaning pipeline
    '''

    def __init__(self, path, target_path=None):
        self.path = path
        self.target_path = target_path if target_path is not None else path

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{path}': {e}")
            return 1
        self.frames = self.data.get('frames', [])
        self.validator = CubicSplineKeyPointInterpolator(path)
        self.anomaly_detector = AnomalyDetection(self.validator)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.current = json.load(f)
                self.versions = [copy.deepcopy(self.current)]
        except Exception as e:
            print(f"Failed to read/parse json file '{path}': {e}")

    def detectAnomalousFrames(self,
                              num_std_dev=1.8,
                              position_threshold=-0.08,
                              gap_size=4,
                              margin=0.00,
                              recursive_level=0,
                              save_versions=False,
                              show_logs=False):
        '''Detects anomalous frames using ordering_wrist_neighbour + stddev
        intersection, places them in metadata, and optionally reruns detection
        until none are found. Writes only to self.target_path.

        takes:
            num_std_dev: stddev threshold for movement detection
            position_threshold: position threshold for the intersection
            gap_size: gap-fill size for movement detection
            margin: margin for the wrist ordering violation check
            recursive_level: number of recursions left to run
            save_versions: whether to save versions at each recursive level
            show_logs: whether to print logs of the cleaning process
        returns:
            tuple of lists: (anomaly_frames_left, anomaly_frames_right)
        '''
        anomaly_frames_left, anomaly_frames_right = \
            self.anomaly_detector.position_and_filled_movement_and_acceleration_anomalys(
                movement_threshold=0.11,
                position_threshold=-0.08,
                gap_size=4
            )
        if show_logs:
            print(
                f"Recursive level {recursive_level}: "
                f"Detected {len(anomaly_frames_left)} anomalous frames in left hand "
                f"and {len(anomaly_frames_right)} in right hand.")

        if len(anomaly_frames_left) == 0 and len(anomaly_frames_right) == 0:
            if show_logs:
                print(
                    f"No anomalous frames detected at recursive level {recursive_level}")
            # Still need to write the file even if nothing found, so the target
            # exists and the recursion's prior changes persist
            self._writeCurrentToTarget()
            return anomaly_frames_left, anomaly_frames_right

        for frame in anomaly_frames_left:
            self.current['metadata'].setdefault(
                'left_anomalous_frames', []).append(
                self.current['frames'][frame])
            self.current['frames'][frame]['hands']['left'] = []

        for frame in anomaly_frames_right:
            self.current['metadata'].setdefault(
                'right_anomalous_frames', []).append(
                self.current['frames'][frame])
            self.current['frames'][frame]['hands']['right'] = []

        if save_versions:
            self.versions.append(copy.deepcopy(self.current))

        self._writeCurrentToTarget()

        # Rebuild the validator from the target file so the next recursive pass
        # sees the changes we just made, without touching the source
        self.validator = CubicSplineKeyPointInterpolator(self.target_path)
        self.anomaly_detector = AnomalyDetection(self.validator)

        if recursive_level > 0:
            left, right = self.detectAnomalousFrames(
                num_std_dev=num_std_dev,
                position_threshold=position_threshold,
                gap_size=gap_size,
                margin=margin,
                recursive_level=recursive_level - 1,
                save_versions=save_versions,
                show_logs=show_logs,
            )
            return anomaly_frames_left + left, anomaly_frames_right + right

        return anomaly_frames_left, anomaly_frames_right

    def _writeCurrentToTarget(self):
        '''Ensures target directory exists, then writes self.current to it.'''
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
        with open(self.target_path, 'w', encoding='utf-8') as f:
            json.dump(self.current, f, indent=2)

    def getAllMetadata(self):
        '''returns all metadata from the current data'''
        return self.current.get('metadata', {})

    def detectDoubleHandAnomalousFrames(self, show_logs=False):
        """saves all Simultaneous hand anomalys
        removes and saves them to simultaneous_hands_frames in metadata """

        # insure json is up to date
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{self.path}': {e}")
            return 1
        self.frames = self.data.get('frames', [])

        cases = self.validator.checkForSimultaneousHands(show_logs=show_logs)

        for case in cases:
            frame_index, side = case

            frame = copy.deepcopy(self.frames[frame_index])
            self.data['frames'][frame_index]['hands'][side] = []
            self.data['metadata'].setdefault(
                'simultaneous_hands_frames', []).append(frame)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def undoDetectDoubleHandAnomalousFrames(self, show_logs=False):
        """reverces the proccess of detectDoubleHandAnomalousFrames by reinserting the anomalous frames back into the main dataset and removing them from metadata

        this helps to valiate that all changes are trachible and reversible and that no data is lost in the cleaning process
        """
        # insure json is up to date
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Failed to read/parse JSON file '{self.path}': {e}")
            return 1

        # Get saved simultaneous hands frames from metadata
        simultaneous_frames = self.data['metadata'].get(
            'simultaneous_hands_frames', [])

        if not simultaneous_frames:
            if show_logs:
                print("No simultaneous hands frames found in metadata to restore")
            return 0

        restored_count = 0

        # Restore each saved frame back to the main dataset
        for saved_frame in simultaneous_frames:
            frame_index = None
            if 'frame_id' in saved_frame:
                for idx, frame in enumerate(self.data['frames']):
                    if frame.get('frame_id') == saved_frame['frame_id']:
                        frame_index = idx
                        break

            # If not found, try by frame_index metadata
            if frame_index is None and 'frame_index' in saved_frame:
                frame_index = saved_frame['frame_index']

            # Restore the hands data
            if frame_index is not None and frame_index < len(
                    self.data['frames']):
                if 'hands' in saved_frame:
                    self.data['frames'][frame_index]['hands'] = saved_frame['hands']
                    restored_count += 1

        if 'simultaneous_hands_frames' in self.data['metadata']:
            del self.data['metadata']['simultaneous_hands_frames']

        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

        if show_logs:
            print(
                f"Restored {restored_count} frames with simultaneous hand anomalies")
        return 0


def cleanCorpus(source_root, target_root,
                num_std_dev=1.6,
                position_threshold=-0.08,
                gap_size=4,
                margin=0.03,
                recursive_level=3,
                show_logs=False):
    '''Walks a corpus directory and runs DataCleaner on every JSON file,
    preserving the relative folder structure in the target directory.

    takes:
        source_root: root directory of the source corpus
        target_root: root directory where cleaned files will be written
        num_std_dev, position_threshold, gap_size, margin: detection parameters
        recursive_level: how many recursive cleaning passes to run per file
        show_logs: verbose output
    returns:
        dict with counts of files processed, errors, and total anomalies removed
    '''
    stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_left_anomalies': 0,
        'total_right_anomalies': 0,
        'failed_files': [],
    }

    for dirpath, _, filenames in os.walk(source_root):
        for fname in filenames:
            if not fname.lower().endswith('.json'):
                continue
            # Skip non-sign files like dominance_labels.json at the root
            if dirpath == source_root:
                continue

            source_file = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(source_file, source_root)
            target_file = os.path.join(target_root, rel_path)

            if show_logs:
                print(f"Cleaning {rel_path}")

            try:
                cleaner = DataCleaner(
                    path=source_file, target_path=target_file)
                left, right = cleaner.detectAnomalousFrames(
                    num_std_dev=num_std_dev,
                    position_threshold=position_threshold,
                    gap_size=gap_size,
                    margin=margin,
                    recursive_level=recursive_level,
                    show_logs=False,
                )
                stats['files_processed'] += 1
                stats['total_left_anomalies'] += len(left)
                stats['total_right_anomalies'] += len(right)
            except Exception as e:
                stats['files_failed'] += 1
                stats['failed_files'].append((rel_path, str(e)))
                print(f"  FAILED {rel_path}: {e}")

    print("\n=== Corpus cleaning complete ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files failed:    {stats['files_failed']}")
    print(f"Total left anomalies removed:  {stats['total_left_anomalies']}")
    print(f"Total right anomalies removed: {stats['total_right_anomalies']}")

    return stats


if __name__ == "__main__":
    cleanCorpus(
        source_root=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\UNPROCCESSED_KEYPOINTS_V1",
        target_root=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\CLEANED_KEYPOINTS_V1",
        show_logs=True,
    )
