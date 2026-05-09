"""
Sweep harness for the BSL feature-set evaluation.

Drives 5-fold-CNN.py across the experimental matrix and appends every result
to a single results.jsonl file.  Re-runnable: by default it skips any
(feature_set x keypoint_version x pipeline_config x augment x seed) combo
already present in results.jsonl.

Pipeline-stage taxonomy
-----------------------
The six pipeline stages decompose into two categories:

* Structural prerequisites: cleaning_10finger, interpolation,
  temporal_normalisation.  Removing any of these breaks the CNN's fixed-
  shape input contract (NaNs in the tensor, malformed per-hand feature
  vectors, or variable-length sequences).  Their contribution is justified
  theoretically rather than by ablation.

* Quality enhancers: cleaning_mislabel, temporal_cropping,
  spatial_normalisation.  The pipeline runs without each of these; the
  data the CNN sees is simply dirtier.  These are the stages that admit
  empirical ablation, and the corpora for each ablation are pre-generated
  under feature_corpuses_V2_without_<stage>/.

Usage examples
--------------
    python run_sweep.py --mode main
    python run_sweep.py --mode stage_ablation
    python run_sweep.py --mode aug_ablation
    python run_sweep.py --mode seed_stability
    python run_sweep.py --mode all
"""
import argparse
import json
import os
import subprocess
import sys
from itertools import product

# ---------------------------------------------------------------------------
# Configuration: corpus roots and the experimental matrix.
# Update these paths to match your filesystem.
# ---------------------------------------------------------------------------

_FEATURES_BASE = (
    r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\features'
)

CORPUS_ROOT_V1                 = os.path.join(_FEATURES_BASE, 'feature_corpuses_V1')
CORPUS_ROOT_V2_FULL            = os.path.join(_FEATURES_BASE, 'feature_corpuses_V2')
CORPUS_ROOT_V2_NO_MISLABEL     = os.path.join(_FEATURES_BASE, 'feature_corpuses_V2_without_cleaning')
CORPUS_ROOT_V2_NO_CROPPING     = os.path.join(_FEATURES_BASE, 'feature_corpuses_V2_without_cropping')
CORPUS_ROOT_V2_NO_SPATIAL_NORM = os.path.join(_FEATURES_BASE, 'feature_corpuses_V2_without_spacal_norm')

# Map short feature_set IDs to the corpus folder name produced by
# generate_corpuses.py. Update if the folders are ever renamed.
FEATURE_SET_FOLDERS = {
    'v0': 'v0_raw_coordinates_840',
    'v1': 'v1_all_proximity_and_angles_9090',
    'v2': 'v2_proximity_only_8610',
    'v3': 'v3_tips_palms_with_angles_mixed_1140',
    'v4': 'v4_12points_angles_and_distance_850',
    'v5': 'v5_12points_distances_only_660',
    'v6': 'v6_minimal_index_palm_pinkie_850',
    'v7': 'v7_interhand_distances_only_360',
}

ALL_FEATURE_SETS = list(FEATURE_SET_FOLDERS.keys())

# All six pipeline stages, in the order they appear in the formal pipeline.
# Used only to construct full-pipeline pipeline_config dicts.
ALL_PIPELINE_STAGES = [
    'cleaning_10finger',
    'cleaning_mislabel',
    'interpolation',
    'temporal_cropping',
    'temporal_normalisation',
    'spatial_normalisation',
]

# Stages whose contribution can be measured by ablation. The other three
# are structural prerequisites of the CNN and are not ablatable.
ABLATABLE_STAGES = [
    'cleaning_mislabel',
    'temporal_cropping',
    'spatial_normalisation',
]

# Map an ablated-stage name to the corpus root that contains the corpora
# generated with that stage disabled. Stage ablation is only available for
# V2 keypoints; V1 has only the full-pipeline corpus.
V2_ABLATION_ROOTS = {
    'cleaning_mislabel':     CORPUS_ROOT_V2_NO_MISLABEL,
    'temporal_cropping':     CORPUS_ROOT_V2_NO_CROPPING,
    'spatial_normalisation': CORPUS_ROOT_V2_NO_SPATIAL_NORM,
}

DEFAULT_PIPELINE = {stage: True for stage in ALL_PIPELINE_STAGES}

CNN_SCRIPT = '5-fold-CNN.py'
RESULTS_FILE = 'results.jsonl'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identify_ablation(pipeline_config):
    """Return the single ablated stage name, or None if pipeline is full.

    Raises ValueError if more than one ablatable stage is disabled, or if
    a structural-prerequisite stage is disabled (which the corpora do not
    support).
    """
    disabled = [s for s in ALL_PIPELINE_STAGES if not pipeline_config.get(s, True)]
    if not disabled:
        return None
    structural = [s for s in disabled if s not in ABLATABLE_STAGES]
    if structural:
        raise ValueError(
            f"Cannot ablate structural prerequisite stage(s): {structural}. "
            f"Only {ABLATABLE_STAGES} are ablatable.")
    if len(disabled) > 1:
        raise ValueError(
            f"Multiple stages disabled: {disabled}. Only one ablation per run "
            f"is supported.")
    return disabled[0]


def corpus_path(feature_set, keypoint_version, pipeline_config):
    """Resolve the corpus_data.json path for a (fs, kp_version, config) triple.

    For V1: only the full-pipeline corpus exists; raises if any stage is
    disabled.
    For V2: routes to the matching ablation corpus when exactly one
    ablatable stage is disabled, otherwise to the full V2 corpus.
    """
    folder = FEATURE_SET_FOLDERS[feature_set]
    ablated = _identify_ablation(pipeline_config)

    if keypoint_version == 'V1':
        if ablated is not None:
            raise ValueError(
                f"V1 stage-ablation corpora do not exist; cannot ablate "
                f"{ablated} on V1.")
        root = CORPUS_ROOT_V1
    elif keypoint_version == 'V2':
        root = (V2_ABLATION_ROOTS[ablated] if ablated is not None
                else CORPUS_ROOT_V2_FULL)
    else:
        raise ValueError(f"Unknown keypoint_version: {keypoint_version}")

    return os.path.join(root, folder, 'corpus_data.json')


def load_existing_results(results_file):
    """Return list of records already in results_file. Empty list if absent."""
    if not os.path.exists(results_file):
        return []
    out = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def already_run(records, feature_set, keypoint_version, pipeline_config,
                augment, seed):
    """Check whether a matching record already exists."""
    for r in records:
        if (r.get('feature_set') == feature_set
                and r.get('keypoint_version') == keypoint_version
                and r.get('pipeline_config') == pipeline_config
                and r.get('augmentation', {}).get('enabled') == augment
                and r.get('training', {}).get('seed') == seed):
            return True
    return False


def run_one(feature_set, keypoint_version, pipeline_config, augment, seed,
            notes='', extra_args=None, dry_run=False):
    """Invoke the CNN script for a single (filtered) configuration."""
    try:
        path = corpus_path(feature_set, keypoint_version, pipeline_config)
    except ValueError as e:
        print(f"  [skip] {e}")
        return False

    if not os.path.exists(path) and not dry_run:
        print(f"  [skip] corpus missing: {path}")
        return False

    cmd = [
        sys.executable, CNN_SCRIPT,
        '--corpus', path,
        '--keypoint_version', keypoint_version,
        '--seed', str(seed),
        '--results_file', RESULTS_FILE,
    ]
    if not augment:
        cmd.append('--no_augment')
    # Pass --no_<stage> for each disabled stage so the recorded
    # pipeline_config in the result reflects what was actually run.
    for stage in ALL_PIPELINE_STAGES:
        if not pipeline_config.get(stage, True):
            cmd.append(f'--no_{stage}')
    if notes:
        cmd.extend(['--notes', notes])
    if extra_args:
        cmd.extend(extra_args)

    print('  $', ' '.join(cmd))
    if dry_run:
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Run-set generators
# ---------------------------------------------------------------------------

def iter_main_matrix(feature_sets, keypoint_versions, seeds):
    """Yield the main feature-set x kp-version x seed runs (full pipeline, aug ON)."""
    for fs, kv, seed in product(feature_sets, keypoint_versions, seeds):
        yield {
            'feature_set': fs,
            'keypoint_version': kv,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'notes': 'main_matrix',
        }


def iter_stage_ablation(feature_sets, seed=42):
    """Yield V2 runs with each ablatable stage individually disabled.

    Stage ablation is V2-only by construction (the ablation corpora are
    only generated for V2 keypoints).
    """
    for fs in feature_sets:
        for stage in ABLATABLE_STAGES:
            cfg = dict(DEFAULT_PIPELINE)
            cfg[stage] = False
            yield {
                'feature_set': fs,
                'keypoint_version': 'V2',
                'pipeline_config': cfg,
                'augment': True,
                'seed': seed,
                'notes': f'stage_ablation:no_{stage}',
            }


def iter_aug_ablation(feature_sets, keypoint_version='V2', seed=42):
    """Yield no-augmentation runs for each feature set."""
    for fs in feature_sets:
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': False,
            'seed': seed,
            'notes': 'aug_ablation',
        }


def iter_seed_stability(feature_sets, keypoint_version='V2',
                        seeds=(0, 1, 2, 3, 4)):
    """Yield multi-seed stability runs for the top performers."""
    for fs, seed in product(feature_sets, seeds):
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'notes': 'seed_stability',
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['main', 'stage_ablation', 'aug_ablation',
                                 'seed_stability', 'all'],
                        help='Which sweep to run')
    parser.add_argument('--feature_set', action='append', default=None,
                        help='Restrict to specific feature_set(s). Repeatable. '
                             'Default: all eight feature sets.')
    parser.add_argument('--keypoint_version', action='append', default=None,
                        help="Restrict to V1 / V2. Repeatable. "
                             "Default: both for `main`, V2 only for ablations.")
    parser.add_argument('--seed', type=int, action='append', default=None,
                        help='Override seed list. Repeatable.')
    parser.add_argument('--results_file', type=str, default='results.jsonl')
    parser.add_argument('--force', action='store_true',
                        help='Re-run experiments that already have a record.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the command lines without running them.')
    args = parser.parse_args()

    global RESULTS_FILE
    RESULTS_FILE = args.results_file

    feature_sets = args.feature_set or ALL_FEATURE_SETS
    keypoint_versions = args.keypoint_version or ['V1', 'V2']
    seeds = args.seed or [42]

    # Build the run list
    runs = []
    if args.mode in ('main', 'all'):
        runs.extend(iter_main_matrix(feature_sets, keypoint_versions, seeds))
    if args.mode in ('stage_ablation', 'all'):
        runs.extend(iter_stage_ablation(feature_sets))
    if args.mode in ('aug_ablation', 'all'):
        runs.extend(iter_aug_ablation(feature_sets))
    if args.mode in ('seed_stability', 'all'):
        # Default seed_stability to v5/v7 only unless feature_sets explicitly
        # given, since this mode is intended to bound noise on the top
        # performers rather than on every variant.
        fs_for_stab = (args.feature_set or ['v7', 'v5'])
        runs.extend(iter_seed_stability(fs_for_stab))

    # Skip combinations already done
    existing = load_existing_results(RESULTS_FILE)
    print(f"Loaded {len(existing)} existing records from {RESULTS_FILE}")

    queued = []
    skipped = 0
    for r in runs:
        if not args.force and already_run(
                existing, r['feature_set'], r['keypoint_version'],
                r['pipeline_config'], r['augment'], r['seed']):
            skipped += 1
            continue
        queued.append(r)

    print(f"\nTotal planned runs: {len(runs)}")
    print(f"  Already complete:  {skipped}")
    print(f"  To execute:        {len(queued)}\n")

    if not queued:
        print("Nothing to do.")
        return

    # Execute
    n_ok = 0
    n_fail = 0
    for i, r in enumerate(queued, 1):
        cfg_str = ','.join(s for s, on in r['pipeline_config'].items()
                           if not on) or 'full'
        print(f"\n[{i}/{len(queued)}] {r['feature_set']} kp{r['keypoint_version']} "
              f"aug={r['augment']} seed={r['seed']} pipeline={cfg_str}")
        ok = run_one(
            feature_set=r['feature_set'],
            keypoint_version=r['keypoint_version'],
            pipeline_config=r['pipeline_config'],
            augment=r['augment'],
            seed=r['seed'],
            notes=r['notes'],
            dry_run=args.dry_run,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nSweep complete: {n_ok} succeeded, {n_fail} failed")


if __name__ == '__main__':
    main()