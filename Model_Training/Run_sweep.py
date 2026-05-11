"""
Sweep harness for the BSL feature-set evaluation.

Drives 5-fold-CNN.py across the experimental matrix and appends every result
to a single results.jsonl file.  Re-runnable: by default it skips any
(feature_set x keypoint_version x pipeline_config x augment x seed x cv_mode)
combo already present in results.jsonl.

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

Cross-validation modes
----------------------
Two modes are supported and recorded in each result:

* stratified (default): sample-level stratified k-fold. Used for the headline
  cross-feature-set comparison and for all stage / aug / seed-stability
  ablations.
* signer_disjoint: leave-one-signer-out. Used to quantify the accuracy drop
  between within-signer and signer-independent generalisation. Defaults to v5
  and v7 only since this analysis is intended to bound generalisation on the
  top performers rather than on every variant.

Usage examples
--------------
    python run_sweep.py --mode main
    python run_sweep.py --mode stage_ablation
    python run_sweep.py --mode aug_ablation
    python run_sweep.py --mode seed_stability
    python run_sweep.py --mode signer_disjoint
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
CORPUS_ROOT_V2_WITH_SIGNER_IDS = os.path.join(_FEATURES_BASE, 'feature_corpuses_V2_with_signer_ids')

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

ALL_PIPELINE_STAGES = [
    'cleaning_10finger',
    'cleaning_mislabel',
    'interpolation',
    'temporal_cropping',
    'temporal_normalisation',
    'spatial_normalisation',
]

ABLATABLE_STAGES = [
    'cleaning_mislabel',
    'temporal_cropping',
    'spatial_normalisation',
]

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
    """Return the single ablated stage name, or None if pipeline is full."""
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


def corpus_path(feature_set, keypoint_version, pipeline_config, cv_mode='stratified'):
    """Resolve the corpus_data.json path for a (fs, kp_version, config) triple."""
    folder = FEATURE_SET_FOLDERS[feature_set]
    ablated = _identify_ablation(pipeline_config)

    if keypoint_version == 'V1':
        if ablated is not None:
            raise ValueError(
                f"V1 stage-ablation corpora do not exist; cannot ablate "
                f"{ablated} on V1.")
        root = CORPUS_ROOT_V1
    elif keypoint_version == 'V2':
        if cv_mode == 'signer_disjoint':
            # Signer-disjoint requires the corpus regenerated with explicit
            # signer_id fields. Stage ablation is not currently supported in
            # signer-disjoint mode.
            if ablated is not None:
                raise ValueError(
                    f"Stage ablation under signer_disjoint mode requires "
                    f"regenerated ablation corpora with signer IDs, which "
                    f"are not currently available.")
            root = CORPUS_ROOT_V2_WITH_SIGNER_IDS
        else:
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
                augment, seed, cv_mode):
    """Check whether a matching record already exists.

    cv_mode is part of the dedup key so a stratified run does not block a
    signer_disjoint run on the same (feature_set, kv, ...) tuple from
    executing. Records written by older versions of the CNN script that
    predate the cv_mode field are treated as 'stratified' for the dedup
    check, since that was the only mode the older script supported.
    """
    for r in records:
        rec_cv_mode = r.get('cv_mode') or r.get('training', {}).get(
            'cv_mode', 'stratified')
        if (r.get('feature_set') == feature_set
                and r.get('keypoint_version') == keypoint_version
                and r.get('pipeline_config') == pipeline_config
                and r.get('augmentation', {}).get('enabled') == augment
                and r.get('training', {}).get('seed') == seed
                and rec_cv_mode == cv_mode):
            return True
    return False


def run_one(feature_set, keypoint_version, pipeline_config, augment, seed,
            cv_mode='stratified', notes='', extra_args=None, dry_run=False):
    """Invoke the CNN script for a single (filtered) configuration."""
    try:
        path = corpus_path(feature_set, keypoint_version, pipeline_config, cv_mode=cv_mode)
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
        '--cv_mode', cv_mode,
        '--results_file', RESULTS_FILE,
    ]
    if not augment:
        cmd.append('--no_augment')
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
    for fs, kv, seed in product(feature_sets, keypoint_versions, seeds):
        yield {
            'feature_set': fs,
            'keypoint_version': kv,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'cv_mode': 'stratified',
            'notes': 'main_matrix',
        }


def iter_stage_ablation(feature_sets, seed=42):
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
                'cv_mode': 'stratified',
                'notes': f'stage_ablation:no_{stage}',
            }


def iter_aug_ablation(feature_sets, keypoint_version='V2', seed=42):
    for fs in feature_sets:
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': False,
            'seed': seed,
            'cv_mode': 'stratified',
            'notes': 'aug_ablation',
        }


def iter_seed_stability(feature_sets, keypoint_version='V2',
                        seeds=(0, 1, 2, 3, 4)):
    for fs, seed in product(feature_sets, seeds):
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'cv_mode': 'stratified',
            'notes': 'seed_stability',
        }


def iter_signer_disjoint(feature_sets, keypoint_version='V2', seed=42):
    """Yield leave-one-signer-out runs for the requested feature sets.

    Signer-disjoint cross-validation answers a different question from the
    main matrix: not 'how accurate is each variant' but 'how much of the
    headline accuracy survives when training and validation never share a
    signer'. The default scope is v5 and v7, the two top performers under
    stratified CV, since the analysis is concerned with the generalisation
    of the headline finding rather than with re-ranking the eight variants.
    """
    for fs in feature_sets:
        yield {
            'feature_set': fs,
            'keypoint_version': keypoint_version,
            'pipeline_config': dict(DEFAULT_PIPELINE),
            'augment': True,
            'seed': seed,
            'cv_mode': 'signer_disjoint',
            'notes': 'signer_disjoint',
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['main', 'stage_ablation', 'aug_ablation',
                                 'seed_stability', 'signer_disjoint', 'all'],
                        help='Which sweep to run')
    parser.add_argument('--feature_set', action='append', default=None,
                        help='Restrict to specific feature_set(s). Repeatable. '
                             'Default depends on mode.')
    parser.add_argument('--keypoint_version', action='append', default=None,
                        help="Restrict to V1 / V2. Repeatable.")
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

    runs = []
    if args.mode in ('main', 'all'):
        runs.extend(iter_main_matrix(feature_sets, keypoint_versions, seeds))
    if args.mode in ('stage_ablation', 'all'):
        runs.extend(iter_stage_ablation(feature_sets))
    if args.mode in ('aug_ablation', 'all'):
        runs.extend(iter_aug_ablation(feature_sets))
    if args.mode in ('seed_stability', 'all'):
        fs_for_stab = (args.feature_set or ['v7', 'v5'])
        runs.extend(iter_seed_stability(fs_for_stab))
    if args.mode in ('signer_disjoint', 'all'):
        # Defaults to v5/v7 when no feature_set is explicitly given. Pass
        # --feature_set repeatedly to expand the scope to other variants.
        fs_for_signer = (args.feature_set or ['v5', 'v7'])
        runs.extend(iter_signer_disjoint(fs_for_signer))

    existing = load_existing_results(RESULTS_FILE)
    print(f"Loaded {len(existing)} existing records from {RESULTS_FILE}")

    queued = []
    skipped = 0
    for r in runs:
        if not args.force and already_run(
                existing, r['feature_set'], r['keypoint_version'],
                r['pipeline_config'], r['augment'], r['seed'],
                r.get('cv_mode', 'stratified')):
            skipped += 1
            continue
        queued.append(r)

    print(f"\nTotal planned runs: {len(runs)}")
    print(f"  Already complete:  {skipped}")
    print(f"  To execute:        {len(queued)}\n")

    if not queued:
        print("Nothing to do.")
        return

    n_ok = 0
    n_fail = 0
    for i, r in enumerate(queued, 1):
        cfg_str = ','.join(s for s, on in r['pipeline_config'].items()
                           if not on) or 'full'
        print(f"\n[{i}/{len(queued)}] {r['feature_set']} kp{r['keypoint_version']} "
              f"aug={r['augment']} seed={r['seed']} "
              f"cv={r.get('cv_mode', 'stratified')} pipeline={cfg_str}")
        ok = run_one(
            feature_set=r['feature_set'],
            keypoint_version=r['keypoint_version'],
            pipeline_config=r['pipeline_config'],
            augment=r['augment'],
            seed=r['seed'],
            cv_mode=r.get('cv_mode', 'stratified'),
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