import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import sys
from pathlib import Path
import shutil
import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent))
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from anomaly_detection import AnomalyDetection
from data_cleaner import DataCleaner
from copy import deepcopy


# Result types
@dataclass
class FrameLevelMetrics:
    """
    Confusion-matrix counts at the individual frame level.
    A frame is a positive if it was actually swapped on that side.
    """
    true_positives:  int = 0   # swapped frame, correctly flagged
    false_positives: int = 0   # clean frame, incorrectly flagged
    true_negatives:  int = 0   # clean frame, correctly not flagged
    false_negatives: int = 0   # swapped frame, missed

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total else 0.0

    def __repr__(self):
        return (f"TP={self.true_positives} FP={self.false_positives} "
                f"TN={self.true_negatives} FN={self.false_negatives} | "
                f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}")


@dataclass
class FileResult:
    """Detection results for a single file under a single method/param configuration."""
    filepath:       str
    method_name:    str
    params:         dict

    # ground truth — derived from metadata["testing_changes"]
    ground_truth_left:  list[int] = field(default_factory=list)
    ground_truth_right: list[int] = field(default_factory=list)
    total_frames:       int = 0

    # detector output
    detected_left:  list[int] = field(default_factory=list)
    detected_right: list[int] = field(default_factory=list)

    # per-side metrics
    metrics_left:  FrameLevelMetrics = field(default_factory=FrameLevelMetrics)
    metrics_right: FrameLevelMetrics = field(default_factory=FrameLevelMetrics)


@dataclass
class MethodSummary:
    """Aggregated metrics across all files for one method/param configuration."""
    method_name: str
    params:      dict
    per_file:    list[FileResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, FrameLevelMetrics]:
        """Sum confusion matrix counts across all files, both sides."""
        agg = {'left': FrameLevelMetrics(), 'right': FrameLevelMetrics(), 'combined': FrameLevelMetrics()}
        for fr in self.per_file:
            for side, attr in [('left', 'metrics_left'), ('right', 'metrics_right')]:
                m = getattr(fr, attr)
                agg[side].true_positives  += m.true_positives
                agg[side].false_positives += m.false_positives
                agg[side].true_negatives  += m.true_negatives
                agg[side].false_negatives += m.false_negatives
        # combined = sum of both sides
        for side in ('left', 'right'):
            agg['combined'].true_positives  += agg[side].true_positives
            agg['combined'].false_positives += agg[side].false_positives
            agg['combined'].true_negatives  += agg[side].true_negatives
            agg['combined'].false_negatives += agg[side].false_negatives
        return agg


# Main test runner
class RunDetectionTests:
    """
    Evaluates AnomalyDetection methods against a degraded corpus.

    Ground truth is read from metadata["testing_changes"] which frameSwap()
    writes for every swap: {start_frame, size, left_hand_frames, right_hand_frames}.

    For each (method, param_grid) combination the runner:
        1. Loads each degraded JSON
        2. Derives the set of actually-swapped frame indices per side
        3. Runs the detector to get flagged frame indices per side
        4. Computes TP / FP / TN / FN at the frame level
    """

    # Registry of detection strategies.
    # Each entry: method_name -> callable(detector, **params) -> (left_flags, right_flags)
    METHODS: dict[str, Callable] = {
        'movement_only': lambda det, **p: det.movementAnomalys(**p),

        'position_only': lambda det, **p: det.posisionAnomalys(**p),

        'filled_movement': lambda det, **p: det.filledMovementAnomalys(**p),
        
        'ordering_palm_center_neighbour': lambda det, **p: (
            det.handOrderingAnomalysByPalmCenterUsingNeighbourFilling(**p)
        ),
 
        'ordering_wrist_neighbour': lambda det, **p: (
            det.handOrderingAnomalysByWristUsingNeighbourFilling(**p)
        ),
 
        'ordering_extremes_neighbour': lambda det, **p: (
            det.handOrderingAnomalysByExtremesUsingNeighbourFilling(**p)
        ),
        
        'ordering_palm_center_interpolation': lambda det, **p: (
            det.handOrderingAnomalysByPalmCenterUsingInterpolation(**p)
        ),
 
        'ordering_wrist_interpolation': lambda det, **p: (
            det.handOrderingAnomalysByWristUsingInterpolation(**p)
        ),
 
        'ordering_extremes_interpolation': lambda det, **p: (
            det.handOrderingAnomalysByExtremesUsingInterpolation(**p)
        ),


        # --- relative movement variants ---
        'filled_relative_mad': lambda det, **p: (
            det.filledMovementAnomalysByMAD(**p)
        ),

        'filled_relative_percentile': lambda det, **p: (
            det.filledMovementAnomalysByPercentile(**p)
        ),

        'filled_relative_stddev': lambda det, **p: (
            det.filledMovementAnomalysByStdDev(**p)
        ),
        
        'position_and_filled_movement': lambda det, **p: (
            det.posisionAndFilledMovmentAnomalys(**p)
        ),
        
        'position_and_filled_movement_by_stddev': lambda det, **p: (
            det.posisionAndFilledMovmentAnomalysByStdDev(**p)
        ),
        
        'position_and_filled_movement_by_mad': lambda det, **p: (
            det.posisionAndFilledMovmentAnomalysByMAD(**p)
        ),
        
        'position_and_filled_movement_by_percentile': lambda det, **p: (
            det.posisionAndFilledMovmentAnomalysByPercentile(**p)
        ),
        
        'position_and_stddev_intersection': lambda det, **p: (
            det.posisionAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_palm_center_neighbour_stddev_intersection': lambda det, **p: (
            det.OrderingByPalmsWithNeighbourFillingAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_wrist_neighbour_stddev_intersection': lambda det, **p: (
            det.OrderingByWristsWithNeighbourFillingAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_extremes_neighbour_stddev_intersection': lambda det, **p: (
            det.OrderingByExtremesWithNeighbourFillingAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_palm_center_interpolation_stddev_intersection': lambda det, **p: (
            det.OrderingByPalmsWithInterpolationAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_wrist_interpolation_stddev_intersection': lambda det, **p: (
            det.OrderingByWristsWithInterpolationAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_extremes_interpolation_stddev_intersection': lambda det, **p: (
            det.OrderingByExtremesWithInterpolationAndFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_palm_center_neighbour_stddev_union': lambda det, **p: (
            det.OrderingByPalmsWithNeighbourFillingOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_wrist_neighbour_stddev_union': lambda det, **p: (
            det.OrderingByWristsWithNeighbourFillingOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_extremes_neighbour_stddev_union': lambda det, **p: (
            det.OrderingByExtremesWithNeighbourFillingOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_palm_center_interpolation_stddev_union': lambda det, **p: (
            det.OrderingByPalmsWithInterpolationOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_wrist_interpolation_stddev_union': lambda det, **p: (
            det.OrderingByWristsWithInterpolationOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'ordering_extremes_interpolation_stddev_union': lambda det, **p: (
            det.OrderingByExtremesWithInterpolationOrFilledMovmentByStdDevIntersection(**p)
        ),
        
        'acceleration_only': lambda det, **p: det.AccelerationAnomalys(**p),

        
        'position_and_filled_movement_and_acceleration': lambda det, **p: (
            det.position_and_filled_movement_and_acceleration_anomalys(**p)
        ),
        
        'appearance_disappearance': lambda det, **p: (
            det.findAppearanceDisappearanceSwaps(**p)
        )
    }
    
    # Default parameter grids — each is a list of kwarg dicts to trial
    DEFAULT_PARAM_GRIDS: dict[str, list[dict]] = {
        'movement_only': [
            {'threshold': t} for t in [0.15, 0.16, 0.17, 0.18, 0.19]
        ],
        'position_only': [
            {'threshold': t} for t in [-0.06, -0.07, -0.08, -0.09, -0.10]
        ],
        'filled_movement': [
            {'threshold': t, 'gap_size': g}
            for t in [0.09, 0.10, 0.11, 0.12, 0.13]
            for g in [3, 4, 5]
        ],
        'position_and_filled_movement': [
            {'movement_threshole': mt, 'position_threshold': pt, 'gap_size': g}
            for mt in [0.10, 0.11, 0.12, 0.13]
            for pt in [-0.07, -0.08, -0.09, -0.10]
            for g in [2, 3]
        ],
        # --- relative movement variants ---
        'filled_relative_mad': [
            {'threshold': t, 'gap_size': g}
            for t in [5.2, 5.3, 5.4, 5.5, 5.6]
            for g in [3, 4, 5]
        ],
        'filled_relative_percentile': [
            {'percentile': p, 'gap_size': g}
            for p in [96, 97, 98, 99]
            for g in [3, 4, 5]
        ],
        'filled_relative_stddev': [
            {'num_std_dev': s, 'gap_size': g}
            for s in [2.1, 2.2, 2.3, 2.4]
            for g in [3, 4, 5]
        ],
        
        'position_and_filled_movement_by_stddev': [
            {'movement_threshole': mt, 'position_threshold': pt, 'gap_size': g, 'num_std_dev': s}
            for mt in [0.07, 0.08, 0.09, 0.10]
            for pt in [-0.07, -0.08, -0.09, -0.10]
            for g in [2, 3]
            for s in [2.0, 2.1, 2.2, 2.3]
        ],
        
        'position_and_filled_movement_by_mad': [
            {'movement_threshole': mt, 'position_threshold': pt, 'gap_size': g, 'threshold': t}
            for mt in [0.07, 0.08, 0.09, 0.10]
            for pt in [-0.07, -0.08, -0.09, -0.10]
            for g in [2, 3]
            for t in [5.3, 5.4, 5.5, 5.6]
        ],

        'position_and_filled_movement_by_percentile': [ 
            {'movement_threshole': mt, 'position_threshold': pt, 'gap_size': g, 'percentile': p}
            for mt in [0.07, 0.08, 0.09, 0.10]
            for pt in [-0.07, -0.08, -0.09, -0.10]
            for g in [3, 4, 5]
            for p in [97, 98, 99]
        ],
        
        'position_and_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g}
            for s in [1.1, 1.2, 1.3, 1.4, 1.5]
            for pt in [-0.02, -0.03, -0.04, -0.05]
            for g in [3, 4, 5]
        ],
        'ordering_palm_center_neighbour': [
            {'margin': m}
            for m in [0.06, 0.07, 0.08, 0.09]
        ],
        'ordering_wrist_neighbour': [
            {'margin': m}
            for m in [0.06, 0.07, 0.08, 0.09]
        ],
        'ordering_extremes_neighbour': [
            {'margin': m}
            for m in [0.0, 0.01, 0.02, 0.03]
        ],
        'ordering_palm_center_interpolation': [
            {'margin': m}
            for m in [0.06, 0.07, 0.08, 0.09]
        ],
        'ordering_wrist_interpolation': [
            {'margin': m}
            for m in [0.06, 0.07, 0.08, 0.09]
        ],
        'ordering_extremes_interpolation': [
            {'margin': m}
            for m in [0.0, 0.01, 0.02, 0.03]
        ],
        'ordering_palm_center_neighbour_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.003, 0.005, 0.01]
        ],
        'ordering_wrist_neighbour_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.003, 0.005, 0.01]
        ],
        'ordering_extremes_neighbour_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.0, 0.005, 0.01]
        ],
        'ordering_palm_center_interpolation_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.003, 0.005, 0.01]
        ],
        'ordering_wrist_interpolation_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.003, 0.005, 0.01]
        ],
        'ordering_extremes_interpolation_stddev_intersection': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [1.2, 1.3, 1.4]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.0, 0.005, 0.01]
        ],
        'ordering_palm_center_neighbour_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.03, 0.04, 0.05]
        ],
        'ordering_wrist_neighbour_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.03, 0.04, 0.05]
        ],
        'ordering_extremes_neighbour_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.0, 0.01, 0.02]
        ],
        'ordering_palm_center_interpolation_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.0, 0.01, 0.02]
        ],
        'ordering_wrist_interpolation_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [2, 3]
            for m in [0.01, 0.02, 0.03]
        ],
        'ordering_extremes_interpolation_stddev_union': [
            {'num_std_dev': s, 'position_threshold': pt, 'gap_size': g, 'margin': m}
            for s in [2.1, 2.2, 2.3]
            for pt in [-0.08, -0.09, -0.10]
            for g in [4, 5]
            for m in [0.0, 0.01, 0.02]
        ],
        
        'acceleration_only': [
            {'threshold': t, 'inclusive': inc, 'interpolate_missing': interp}
            for t in [0.05, 0.06,0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
            for inc in [True, False]
            for interp in [True, False]
        ],
        
        'position_and_filled_movement_and_acceleration': [
            {'position_threshold': pt, 'movement_threshold': mt, 'acceleration_threshold': at, 'gap_size': g}
            for pt in [-0.08, -0.09, -0.10]
            for mt in [0.09, 0.10, 0.11]
            for at in [0.14, 0.15, 0.16]
            for g in [2, 3]
        ],
        
        'appearance_disappearance': [
            {'max_gap': g, 'distance_threshold': d}
            for g in [1, 2, 3]
            for d in [0.05, 0.10, 0.15, 0.20, 0.30]
        ]
    }

    def __init__(self, corpus_path: str, param_grids: dict[str, list[dict]] | None = None, show_logs=False):
        """
        takes:
            corpus_path:  path to the degraded corpus directory
            param_grids:  optional override — dict mapping method name to list
                          of kwarg dicts.  Falls back to DEFAULT_PARAM_GRIDS.
        """
        self.corpus_path = corpus_path
        self.param_grids = param_grids or self.DEFAULT_PARAM_GRIDS
        self.files = self._load_files()
        if show_logs:
            print(f"Loaded {len(self.files)} degraded files from {corpus_path}")

    # file loading
    def _load_files(self) -> list[str]:
        paths = []
        for root, _, filenames in os.walk(self.corpus_path):
            for fname in filenames:
                if fname.lower().endswith('.json'):
                    paths.append(os.path.join(root, fname))
        return sorted(paths)

    @staticmethod
    def _extract_ground_truth(filepath: str) -> tuple[list[int], list[int], int]:
        """
        Reads metadata["testing_changes"] and returns:
            (swapped_left_frames, swapped_right_frames, total_frames)

        A frame is 'swapped left' if it appears in left_hand_frames of any swap entry,
        meaning the data that was originally right-hand data is now in the left slot.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_frames = len(data.get('frames', []))
        changes = data.get('metadata', {}).get('testing_changes', [])

        swapped_left:  set[int] = set()
        swapped_right: set[int] = set()

        for change in changes:
            start = change['start_frame']
            size  = change['size']
            swap_range = set(range(start, start + size))

            # left_hand_frames = frames that had a left hand present when swapped
            # these frames now have wrong-side data in the left slot
            for fi in change.get('left_hand_frames', []):
                if fi in swap_range:
                    swapped_left.add(fi)

            for fi in change.get('right_hand_frames', []):
                if fi in swap_range:
                    swapped_right.add(fi)

        return sorted(swapped_left), sorted(swapped_right), total_frames

    # metrics
    @staticmethod
    def _compute_metrics(
        detected:     list[int],
        ground_truth: list[int],
        total_frames: int,
    ) -> FrameLevelMetrics:
        detected_set = set(detected)
        truth_set    = set(ground_truth)
        all_frames   = set(range(total_frames))

        tp = len(detected_set & truth_set)
        fp = len(detected_set - truth_set)
        fn = len(truth_set - detected_set)
        tn = len((all_frames - detected_set) - truth_set)

        return FrameLevelMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )

    def run(
    self,
    methods: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, MethodSummary]:
        """
        Run all specified methods across all corpus files.
        File loading is the outer loop so each validator is built only once
        and reused across every method/param combination.
        """
        methods = methods or list(self.METHODS.keys())
        summaries: dict[str, MethodSummary] = {}

        # Pre-create a MethodSummary for every (method, params) key
        for method_name in methods:
            if method_name not in self.METHODS:
                print(f"Unknown method '{method_name}' — skipping")
                continue
            for params in self.param_grids.get(method_name, [{}]):
                key = f"{method_name}::{params}"
                summaries[key] = MethodSummary(method_name=method_name, params=params)

        # Outer loop: each file is loaded and validated ONCE
        for file_idx, filepath in enumerate(self.files):
            if verbose:
                print(f"[{file_idx + 1}/{len(self.files)}] {os.path.basename(filepath)}")

            try:
                gt_left, gt_right, total_frames = self._extract_ground_truth(filepath)
                validator = CubicSplineKeyPointInterpolator(filepath)
                detector = AnomalyDetection(validator)
            except Exception as e:
                print(f"  ERROR loading {filepath}: {e}")
                continue

            # Inner loop: every method/param combo reuses the same detector
            for method_name in methods:
                if method_name not in self.METHODS:
                    continue

                detector_fn = self.METHODS[method_name]
                param_grid = self.param_grids.get(method_name, [{}])

                for params in param_grid:
                    key = f"{method_name}::{params}"

                    try:
                        detected_left, detected_right = detector_fn(detector, **params)

                        file_result = FileResult(
                            filepath=filepath,
                            method_name=method_name,
                            params=params,
                            ground_truth_left=gt_left,
                            ground_truth_right=gt_right,
                            total_frames=total_frames,
                            detected_left=list(detected_left),
                            detected_right=list(detected_right),
                            metrics_left=self._compute_metrics(
                                detected_left, gt_left, total_frames),
                            metrics_right=self._compute_metrics(
                                detected_right, gt_right, total_frames),
                        )
                        summaries[key].per_file.append(file_result)

                    except Exception as e:
                        import traceback
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        print(f"    ERROR [{method_name}] {params}: {error_msg}")
                        traceback.print_exc()

        return summaries# reporting
    
    
    @staticmethod
    def print_summary(summaries: dict[str, 'MethodSummary']):
        """Pretty-print aggregated results for every method/param combination."""
        print(f"\n{'═'*90}")
        print(f"{'METHOD':<40} {'SIDE':<10} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} "
              f"{'PREC':>7} {'REC':>7} {'F1':>7}")
        print(f"{'═'*90}")

        for key, summary in sorted(summaries.items()):
            agg = summary.aggregate()
            for side in ('left', 'right', 'combined'):
                m = agg[side]
                label = f"{summary.method_name} {summary.params}" if side == 'left' else ''
                print(f"{label:<40} {side:<10} "
                      f"{m.true_positives:>6} {m.false_positives:>6} "
                      f"{m.true_negatives:>6} {m.false_negatives:>6} "
                      f"{m.precision:>7.3f} {m.recall:>7.3f} {m.f1:>7.3f}")
            print(f"{'-'*90}")

    @staticmethod
    def best_params(
        summaries: dict[str, 'MethodSummary'],
        metric: str = 'f1',
        side:   str = 'combined',
    ) -> dict[str, tuple[dict, float]]:
        """
        For each method, return the param config that maximised the chosen metric.

        takes:
            metric: 'f1' | 'precision' | 'recall' | 'accuracy'
            side:   'left' | 'right' | 'combined'

        returns:
            {method_name: (best_params_dict, best_metric_value)}
        """
        best: dict[str, tuple[dict, float]] = {}

        for key, summary in summaries.items():
            agg = summary.aggregate()
            score = getattr(agg[side], metric)

            method = summary.method_name
            if method not in best or score > best[method][1]:
                best[method] = (summary.params, score)

        print(f"\nBest params per method (metric={metric}, side={side}):")
        for method, (params, score) in best.items():
            print(f"  {method:<35} {params}:  {metric}={score:.4f}")

        return best
    
    @staticmethod
    def performance_by_swap_size(
        summaries: dict[str, 'MethodSummary'],
        methods:   list[str] = ('position_only', 
                                'filled_movement', 
                                'position_and_filled_movement', 
                                'filled_relative_mad', 
                                'filled_relative_percentile', 
                                'filled_relative_stddev',
                                'position_and_filled_movement_by_stddev',
                                'position_and_filled_movement_by_mad',
                                'position_and_filled_movement_by_percentile',
                                'position_and_stddev_intersection''ordering_palm_center',
                                'ordering_wrist',
                                'ordering_extremes'),
        metric:    str = 'f1',
        side:      str = 'combined',
    ):
        """
        For each specified method, finds the best param config (by aggregate metric),
        then breaks down TP/FP/TN/FN and derived metrics per swap size.

        Swap size is read from metadata["testing_changes"] in each file's ground truth —
        a frame is attributed to a swap size based on which change entry it came from.

        takes:
            summaries: output of run()
            methods:   method names to include
            metric:    metric used to select best params per method
            side:      'left' | 'right' | 'combined'
        """

        # ── find best param config per method ─────────────────────────────────────
        best_keys: dict[str, str] = {}   # method_name -> summary key
        for key, summary in summaries.items():
            if summary.method_name not in methods:
                continue
            agg   = summary.aggregate()
            score = getattr(agg[side], metric)
            prev  = best_keys.get(summary.method_name)
            if prev is None or score > getattr(summaries[prev].aggregate()[side], metric):
                best_keys[summary.method_name] = key

        # ── helper: attribute each ground-truth frame to its swap size ─────────────
        @staticmethod
        def _frames_by_swap_size(filepath: str) -> dict[int, dict[str, set[int]]]:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            changes = data.get('metadata', {}).get('testing_changes', [])
            by_size: dict[int, dict[str, set[int]]] = {}
            for change in changes:
                size       = change['size']
                start      = change['start_frame']
                swap_range = set(range(start, start + size))
                if size not in by_size:
                    by_size[size] = {'left': set(), 'right': set()}
                for fi in change.get('left_hand_frames', []):
                    if fi in swap_range:
                        by_size[size]['left'].add(fi)
                for fi in change.get('right_hand_frames', []):
                    if fi in swap_range:
                        by_size[size]['right'].add(fi)
            return by_size

        # ── accumulate confusion matrices per method per swap size ─────────────────
        all_swap_sizes = sorted({1, 2, 3, 4, 5})  # known from degradation patterns

        # structure: {method_name: {swap_size: FrameLevelMetrics}}
        breakdown: dict[str, dict[int, FrameLevelMetrics]] = {
            m: {s: FrameLevelMetrics() for s in all_swap_sizes}
            for m in best_keys
        }

        for method_name, key in best_keys.items():
            summary = summaries[key]

            for file_result in summary.per_file:
                by_size = _frames_by_swap_size(file_result.filepath)

                detected_left  = set(file_result.detected_left)
                detected_right = set(file_result.detected_right)

                for swap_size, sides in by_size.items():
                    if swap_size not in breakdown[method_name]:
                        continue

                    m = breakdown[method_name][swap_size]

                    for side_key, truth_set in sides.items():
                        detected = detected_left if side_key == 'left' else detected_right

                        if side == 'combined' or side == side_key:
                            m.true_positives  += len(detected & truth_set)
                            m.false_negatives += len(truth_set - detected)
                            # FP/TN per swap size isn't meaningful at frame level
                            # (FPs come from clean frames unrelated to this swap)
                            # so we track them globally and note it in output

        # ── print table ───────────────────────────────────────────────────────────
        col_w = 12
        print(f"\n{'═' * 80}")
        print(f"Performance by Swap Size  (metric={metric}, side={side})")
        print(f"Best configs selected by aggregate {metric}")
        print(f"{'═' * 80}")

        for method_name, key in best_keys.items():
            summary = summaries[key]
            print(f"\n  {method_name}  {summary.params}")
            print(f"  {'Size':<8} {'TP':>6} {'FN':>6} {'Recall':>8}  {'Files':>6}")
            print(f"  {'-' * 40}")

            for swap_size in all_swap_sizes:
                m         = breakdown[method_name][swap_size]
                recall    = m.recall  # TP / (TP + FN) — meaningful per swap size
                total_pos = m.true_positives + m.false_negatives
                # count how many files had swaps of this size
                n_files = sum(
                    1 for fr in summary.per_file
                    if swap_size in _frames_by_swap_size(fr.filepath)
                )
                print(f"  {swap_size:<8} {m.true_positives:>6} {m.false_negatives:>6} "
                    f"{recall:>8.3f}  {n_files:>6}")

            print(f"\n  Note: precision/F1 per swap size omitted — FPs arise from clean frames")
            print(f"        unattributable to any single swap. Use aggregate table for F1.")

        print(f"\n{'═' * 80}")

def run_recursive_data_cleaning(
    recursive_range: tuple = (0, 1, 2, 3, 4, 5),
    show_logs:       bool  = False
):
    '''goes through each file in the "path/to/Testing_Corpus_Stratified_stratified - recursive level i" 
    for recursive_range (i, i+1, etc) and runs the data cleaner's detectAnomalousFrames with recursive_level=i
    note that this does not test methods/paramiters of detectAnomalousFrames, 
    it just uses the default position_threshold=-0.1, movement_threshole=0.1, gap_size=5
    
    takes:
        recursive_range: tuple of ints, the recursive levels to run e.g. (0, 1, 2)
        show_logs: whether to print logs of the cleaning process
        
    returns: 
        summaries: tuple of (recursive_level, method_summary_dict) based on analysis of 
        "path/to/Testing_Corpus_Stratified_stratified - recursive level i" for each recursive level in recursive_range
        found: list of the detected anomalous frames for each file in each recursive level, as returned by detectAnomalousFrames
    '''
    
    directories = [r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Stratified_stratified - recursive level " + str(i) for i in recursive_range]
    
    level_summaries = {
        level: MethodSummary(
            method_name='recursive_cleaning',
            params={'recursive_level': level}
        )
        for level in recursive_range
    }    
    found = []
    
    for level, directory in zip(recursive_range, directories):
        print(f"\nRunning recursive data cleaning on directory: {directory}")
        
        
        # loops through each file in the Testing_Corpus_Stratified_stratified - recursive level i
        
        root_path = Path(directory)
        for path in root_path.rglob('*.json'): 
            cleaner = DataCleaner(path=str(path))
            found.append(cleaner.detectAnomalousFrames(
                recursive_level=level))
    
            # compears the metadata of 'left_anomalous_frames' and 'right_anomalous_frames' with the 'testing_changes' metadata to calculate the true positives, false positives, true negatives and false negatives for each recursive level
            # and appends the results to summaries as a tuple of (recursive_level, method_summary_dict)
            
            # get the ground truth swapped frames from testing_changes
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            changes = data.get('metadata', {}).get('testing_changes', [])
            gt_left = set()
            gt_right = set()
            
            for change in changes:
                start = change['start_frame']
                size  = change['size']
                swap_range = set(range(start, start + size))
                gt_left.update(fi for fi in change.get('left_hand_frames', []) if fi in swap_range)
                gt_right.update(fi for fi in change.get('right_hand_frames', []) if fi in swap_range)
            
            # get the detected anomalous frames from metadata
            meta = cleaner.getAllMetadata()
            detected_left  = set(f['frame_index'] for f in meta.get('left_anomalous_frames', []))
            detected_right = set(f['frame_index'] for f in meta.get('right_anomalous_frames', []))
            total_frames   = len(data.get('frames', []))

            # calculate metrics
            metrics_left  = RunDetectionTests._compute_metrics(detected_left,  gt_left,  total_frames)
            metrics_right = RunDetectionTests._compute_metrics(detected_right, gt_right, total_frames)

            level_summaries[level].per_file.append(FileResult(
                filepath=str(path),
                method_name='recursive_cleaning_level_' + str(level),
                params={'recursive_level': level},
                metrics_left=metrics_left,
                metrics_right=metrics_right,
                ground_truth_left=list(gt_left),
                ground_truth_right=list(gt_right),
                detected_left=list(detected_left),
                detected_right=list(detected_right),
                total_frames=total_frames
            ))

    return [s for s in level_summaries.values()], found

if __name__ == "__main__":
    uniform_corpus_path=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Uniform_uniform"
    simple_corpus_path=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Stratified_stratified"
    gaussian_corpus_path=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Test_corpus_with_gaussian_using_momentum_heuristics"
    simple_corpus_path_copy=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Stratified_stratified - Copy"
    ground_truth_based_corpus_paths=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_from_gound_truth_distribution_stratified"
    runner = RunDetectionTests(
        corpus_path=ground_truth_based_corpus_paths
    )

    # run all methods with default param grids
    summaries = runner.run(verbose=False)
    runner.best_params(summaries, metric='f1', side='combined')
    
    runner.performance_by_swap_size(summaries, metric='f1', side='combined')
    
    runner.best_params(summaries, metric='recall', side='combined')
    
    runner.performance_by_swap_size(summaries, metric='recall', side='combined')
    
    
    with open(r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Error_detection_summaries.json", 'w', encoding='utf-8') as f:
        json.dump([dataclasses.asdict(s) for s in summaries.values()], f, indent=2)
    
    
    #runner.print_summary(summaries)
    
    # print(summaries)
    
    # saves sumaries 
    # with open(r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\recursive_cleaning_summaries.json", 'w', encoding='utf-8') as f:
    #    json.dump([dataclasses.asdict(s) for s in summaries], f, indent=2)

    '''
    print("Gaussian placement corpus:")
    RunDetectionTests.performance_by_swap_size(summaries, metric='f1', side='combined')

    print("\n\nUniform placement corpus:")
    runner_uniform = RunDetectionTests(corpus_path=uniform_corpus_path)
    summaries_uniform = runner_uniform.run(verbose=False)
    RunDetectionTests.performance_by_swap_size(summaries_uniform, metric='f1', side='combined')

    print("\n\nStratified placement corpus:")
    runner_stratified = RunDetectionTests(corpus_path=simple_corpus_path)
    summaries_stratified = runner_stratified.run(verbose=False)
    RunDetectionTests.performance_by_swap_size(summaries_stratified, metric='f1', side='combined')
    '''