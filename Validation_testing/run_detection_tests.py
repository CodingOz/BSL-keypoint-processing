import json
import os
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Validators.keypoint_validator import CubicSplineKeyPointInterpolator
from anomaly_detection import AnomalyDetection


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

        'position_and_filled_movement': lambda det, **p: (
            det.posisionAndFilledMovmentAnomalys(**p)
        ),
    }

    # Default parameter grids — each is a list of kwarg dicts to trial
    DEFAULT_PARAM_GRIDS: dict[str, list[dict]] = {
        'movement_only': [
            {'threshold': t} for t in [0.05, 0.10, 0.15, 0.20, 0.30]
        ],
        'position_only': [
            {'threshold': t} for t in [-0.05, -0.10, -0.15, -0.20, -0.30]
        ],
        'filled_movement': [
            {'threshold': t, 'gap_size': g}
            for t in [0.10, 0.15, 0.20]
            for g in [3, 5, 8]
        ],
        'position_and_filled_movement': [
            {'movement_threshole': mt, 'position_threshold': pt, 'gap_size': g}
            for mt in [0.10, 0.15, 0.20]
            for pt in [-0.10, -0.15]
            for g in [3, 5]
        ],
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

        takes:
            methods:  list of method names to run (default: all in METHODS)
            verbose:  print progress

        returns:
            dict mapping method_name -> MethodSummary
            (one MethodSummary per (method, param) combination)
        """
        methods = methods or list(self.METHODS.keys())
        # key: f"{method_name}::{param_repr}"
        summaries: dict[str, MethodSummary] = {}

        for method_name in methods:
            if method_name not in self.METHODS:
                print(f"Unknown method '{method_name}' — skipping")
                continue

            detector_fn  = self.METHODS[method_name]
            param_grid   = self.param_grids.get(method_name, [{}])

            for params in param_grid:
                key = f"{method_name}::{params}"
                summary = MethodSummary(method_name=method_name, params=params)

                for filepath in self.files:
                    if verbose:
                        print(f"  [{method_name}] {os.path.basename(filepath)}  params={params}")

                    try:
                        gt_left, gt_right, total_frames = self._extract_ground_truth(filepath)

                        validator = CubicSplineKeyPointInterpolator(filepath)
                        detector  = AnomalyDetection(validator)

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
                        summary.per_file.append(file_result)

                    except Exception as e:
                        print(f"    ERROR on {filepath}: {e}")

                summaries[key] = summary

        return summaries

    # reporting
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
            print(f"  {method:<35} {params}  →  {metric}={score:.4f}")

        return best
    
    
if __name__ == "__main__":
    runner = RunDetectionTests(
        corpus_path=r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\swapped_hands_corpus"
    )

    # run all methods with default param grids
    summaries = runner.run(verbose=False)

    # print the full table
    RunDetectionTests.print_summary(summaries)

    # find best threshold per method by F1
    RunDetectionTests.best_params(summaries, metric='f1', side='combined')
