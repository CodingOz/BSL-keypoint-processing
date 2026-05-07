"""
Sweep harness for the BSL feature-set evaluation.

Drives 5-fold-CNN.py across the experimental matrix and appends every result
to a single results.jsonl file. Designed to be re-runnable: by default it skips
any (feature_set x keypoint_version x pipeline_config x augment x seed) combo
that already has a record in results.jsonl, so a crash mid-sweep doesn't cost
the runs that already finished.

Usage examples:
    python run_sweep.py --mode main
    python run_sweep.py --mode stage_ablation --feature_set v7
    python run_sweep.py --mode aug_ablation --feature_set v7
    python run_sweep.py --mode stability --feature_set v7 --feature_set v5
    python run_sweep.py --mode all
"""
import argparse
import json
import os
import subprocess
import sys
from itertools import product

# ---------------------------------------------------------------------------
# Configuration: paths and the experimental matrix.
# Edit the two CORPUS_ROOT_* paths to match your filesystem.
# ---------------------------------------------------------------------------

CORPUS_ROOT_V1 = (
    r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing'
    r'\features\feature_corpuses_V1'
)
CORPUS_ROOT_V2 = (
    r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing'
    r'\features\feature_corpuses_V2'
)

# Map short feature_set IDs to the corpus folder name produced by
# generate_corpuses.py. Update if you ever rename the folders.
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

# Pipeline stages, in the order they appear in formalProcessingPipeline.py.
# Used by stage-ablation runs.
PIPELINE_STAGES = [
    'cleaning_10finger',
    'cleaning_mislabel',
    'interpolation',
    'temporal_cropping',
    'temporal_normalisation',
    'spatial_normalisation',
]

DEFAULT_PIPELINE = {stage: True for stage in PIPELINE_STAGES}

CNN_SCRIPT = '5-fold-CNN.py'
RESULTS_FILE = 'results.jsonl'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corpus_path(feature_set, keypoint_version):
    """Resolve the corpus_data.json path for a (feature_set, kp_version) pair."""
    root = CORPUS_ROOT_V1 if keypoint_version == 'V1' else CORPUS_ROOT_V2
    folder = FEATURE_SET_FOLDERS[feature_set]
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
    path = corpus_path(feature_set, keypoint_version)
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
    for stage in PIPELINE_STAGES:
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


def iter_stage_ablation(feature_sets, keypoint_version='V2', seed=42):
    """Yield runs with each pipeline stage individually disabled, plus baseline."""
    for fs in feature_sets:
        # Baseline (full pipeline) - already covered by main_matrix but included
        # here as an explicit member of the ablation series for self-contained
        # analysis; the de-dup check will skip it if main_matrix ran first.
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'notes': 'stage_ablation_baseline',
        }
        for stage in PIPELINE_STAGES:
            cfg = dict(DEFAULT_PIPELINE)
            cfg[stage] = False
            yield {
                'feature_set': fs,
                'keypoint_version': keypoint_version,
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
                             'Default: all eight for `main`, top performers '
                             'for the ablations.')
    parser.add_argument('--keypoint_version', action='append', default=None,
                        help="Restrict to V1 / V2. Repeatable. Default: both.")
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
    if args.mode == 'stage_ablation':
        # Default to top performers if no explicit feature_set given
        fs_for_ablation = (args.feature_set or ['v7', 'v5'])
        runs.extend(iter_stage_ablation(fs_for_ablation))
    if args.mode in ('aug_ablation', 'all'):
        fs_for_aug = (args.feature_set or ['v7', 'v5', 'v2'])
        runs.extend(iter_aug_ablation(fs_for_aug))
    if args.mode in ('seed_stability', 'all'):
        fs_for_stab = (args.feature_set or ['v7', 'v5'])
        runs.extend(iter_seed_stability(fs_for_stab))

    # Skip ones already done
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