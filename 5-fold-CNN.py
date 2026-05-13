"""
1D CNN with k-fold cross-validation for BSL sign classification.

Cross-validation modes:
* `stratified` (default): stratified K-fold on samples, preserving the per-class
  distribution within each fold. Used for the headline cross-feature-set
  comparison and all stage/aug ablations.
* `signer_disjoint`: leave-one-signer-out cross-validation, in which each fold's
  validation set is all recordings from a single signer and the training set is
  all recordings from the remaining signers. Used to measure the accuracy drop
  between within-signer and signer-independent generalisation.

Architecture: 1D convolutions along the temporal axis (10 frames). Features are
treated as input channels, time as the spatial dimension.
"""
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import copy
import os
import random
import time
import subprocess
from datetime import datetime, timezone


class BSLDataset(Dataset):
    """BSL sign language dataset with optional on-the-fly augmentation."""

    def __init__(self, features, labels, augment=False, noise_std=0.03,
                 time_shift_max=1, scale_range=(0.95, 1.05)):
        """
        takes:
            features: np.ndarray of shape (N, num_features, num_frames)
                      already scaled and transposed.
            labels:   np.ndarray of int class indices.
            augment:  whether to apply random augmentation (training only).
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.scale_range = scale_range

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augment:
            # additive Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std

            # Random temporal shift
            shift = np.random.randint(-self.time_shift_max,
                                      self.time_shift_max + 1)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

            # random per-frame scaling
            lo, hi = self.scale_range
            scale = torch.empty(1, x.shape[-1]).uniform_(lo, hi)
            x = x * scale

        return x, y


class BSLConv1DNet(nn.Module):
    """
    Small 1D CNN for temporal sign classification.

    Input shape:  (batch, in_channels, 10), features as channels, frames as time
    Output shape: (batch, num_classes)

    Architecture:
        Conv1D(in_ch, 32, k=3, pad=1) to BN to ReLU to Dropout
        Conv1D(32, 64, k=3, pad=1) to BN to ReLU to Dropout
        Conv1D(64, 64, k=3, pad=1) to BN to ReLU to Dropout
        Global Average Pooling to collapses time dim
        Linear(64, num_classes)
    """

    def __init__(self, in_channels, num_classes, dropout=0.35):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Global average pooling collapses time
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)       # (batch, 64, time)
        x = self.gap(x).squeeze(-1)  # (batch, 64)
        x = self.classifier(x)       # (batch, num_classes)
        return x


class EarlyStopping:
    """Stop training when validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_model = None
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience  # stop if patience exceeded


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / \
        total, np.array(all_preds), np.array(all_labels)


def _set_seed(seed):
    """Seed all RNGs that affect training reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_git_commit():
    """Return short git SHA of the working tree, or 'unknown' if unavailable."""
    try:
        sha = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return sha if sha else 'unknown'
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return 'unknown'


def _count_parameters(model):
    """Total trainable parameter count of the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _measure_inference_time(model, val_loader, device, n_warmup=3, n_runs=10):
    """Mean per-sample inference time in milliseconds, measured on the val set.

    Performs n_warmup untimed passes (to allow CUDA to settle and any lazy
    initialisation to complete), then n_runs timed passes whose total time is
    averaged over the number of samples.
    """
    model.eval()
    total_samples = 0
    for _, y in val_loader:
        total_samples += y.size(0)

    with torch.no_grad():
        for _ in range(n_warmup):
            for X, _ in val_loader:
                X = X.to(device)
                _ = model(X)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            for X, _ in val_loader:
                X = X.to(device)
                _ = model(X)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    if total_samples == 0:
        return 0.0
    return (elapsed / (n_runs * total_samples)) * 1000.0


def _extract_signer_id(sample):
    """Extract a stable signer identifier from a corpus sample.

    The corpus generator records the source JSON filename per sample under
    the 'filename' field. The filename stem is the signer ID by construction:
    each participant submitted recordings under a single UUID, so the same
    stem appearing under multiple sign-letter folders represents the same
    signer recording different letters, and distinct stems represent
    distinct signers (the 'manual-N' files included). This function
    therefore prefers an explicit 'signer_id' field if the corpus has been
    regenerated with one, but falls back to deriving the ID from 'filename'
    on existing corpora.
    """
    # Prefer an explicit signer_id field if present.
    for key in ('signer_id', 'participant_id', 'source_uuid', 'uuid'):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    meta = sample.get('metadata') or {}
    for key in ('signer_id', 'participant_id', 'source_uuid', 'uuid'):
        if key in meta and meta[key] is not None:
            return str(meta[key])
    # Fall back to deriving from the filename stem, which the corpus
    # generator already records per sample.
    if 'filename' in sample and sample['filename']:
        stem = os.path.splitext(str(sample['filename']))[0]
        if stem:
            return stem
    return None


def load_corpus(path, require_signer_id=False):
    """Load corpus_data.json and return (features, labels, signer_ids).

    `signer_ids` is a 1D array of strings of length N. If require_signer_id is
    True and any sample lacks a signer identifier, an informative error is
    raised. If require_signer_id is False, missing signer IDs are returned as
    None entries in the array, and the caller is free to ignore them when
    signer information is not needed.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    features = []
    labels = []
    signer_ids = []
    missing = 0

    for sample in data:
        features.append(sample['features'])
        labels.append(sample['sign'])
        sid = _extract_signer_id(sample)
        if sid is None:
            missing += 1
        signer_ids.append(sid)

    if require_signer_id and missing > 0:
        raise ValueError(
            f"Signer-disjoint cross-validation requested, but {missing} of "
            f"{len(data)} samples in {path} have no recoverable signer ID. "
            f"The corpus generator needs to record one of "
            f"'signer_id', 'participant_id', 'source_uuid', or 'uuid' "
            f"per sample (top-level or under 'metadata').")

    features = np.array(features, dtype=np.float32)
    return features, np.array(labels), np.array(signer_ids, dtype=object)


def prepare_fold(X_all, y_encoded, train_idx, val_idx):
    """
    Scale features per-fold (fit on train, transform both).
    Transpose to (N, channels, time) for Conv1D.
    """
    n_frames = X_all.shape[1]
    n_feats = X_all.shape[2]

    X_train_flat = X_all[train_idx].reshape(len(train_idx), -1)
    X_val_flat = X_all[val_idx].reshape(len(val_idx), -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    X_train = X_train_flat.reshape(len(train_idx), n_frames, n_feats)
    X_val = X_val_flat.reshape(len(val_idx), n_frames, n_feats)

    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return X_train, X_val


def _build_splits(cv_mode, X_all, y_encoded, signer_ids, n_folds, seed):
    """Construct the iterable of (train_idx, val_idx) pairs for the chosen mode.

    Returns a tuple (splits, n_folds_actual, fold_id_func) where:
    * splits is the iterator of (train_idx, val_idx) pairs.
    * n_folds_actual is the number of folds the iterator will yield (for
      stratified this is the requested n_folds; for signer_disjoint it is the
      number of distinct signers).
    * fold_id_func(fold_index, val_idx) returns a human-readable fold label
      (the signer ID for signer_disjoint, the fold number for stratified).
    """
    if cv_mode == 'stratified':
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X_all, y_encoded))
        return splits, len(splits), lambda i, vi: f"fold_{i + 1}"
    if cv_mode == 'signer_disjoint':
        if signer_ids is None or all(s is None for s in signer_ids):
            raise ValueError(
                "signer_disjoint requested but no signer IDs are available. "
                "Re-run load_corpus with require_signer_id=True.")
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X_all, y_encoded, groups=signer_ids))
        # Map each fold index to the held-out signer ID.
        fold_signers = []
        for _, val_idx in splits:
            held_out_signer = signer_ids[val_idx[0]] if len(val_idx) else 'unknown'
            fold_signers.append(held_out_signer)

        def _label(i, vi):
            return f"signer:{fold_signers[i]}"
        return splits, len(splits), _label
    raise ValueError(f"Unknown cv_mode: {cv_mode}")


def run_kfold(corpus_path, n_folds=5, n_epochs=150, batch_size=16,
              lr=1e-3, weight_decay=1e-4, dropout=0.35, patience=15,
              augment=True, device=None, seed=42,
              keypoint_version='unknown', pipeline_config=None,
              experiment_tag=None, results_file=None,
              noise_std=0.03, time_shift_max=1, scale_range=(0.95, 1.05),
              scheduler_factor=0.5, scheduler_patience=7,
              cv_mode='stratified',
              save_models_dir=None,
              notes=''):
    """Run k-fold cross-validation under the chosen `cv_mode`.

    Returns the structured result record (also appended to results_file if set).
    """

    _set_seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    require_signer_id = (cv_mode == 'signer_disjoint')
    X_all, y_raw, signer_ids = load_corpus(
        corpus_path, require_signer_id=require_signer_id)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    n_samples, n_frames, n_feats = X_all.shape
    n_classes = len(class_names)

    feature_set_full_name = os.path.basename(os.path.dirname(corpus_path))
    feature_set_short = (feature_set_full_name.split('_')[0]
                         if feature_set_full_name else 'unknown')

    class_distribution = {
        cls: int(count)
        for cls, count in zip(*np.unique(y_raw, return_counts=True))
    }

    # Signer-level distribution metadata for record provenance
    if cv_mode == 'signer_disjoint':
        unique_signers = [s for s in np.unique(signer_ids) if s is not None]
        signer_distribution = {
            str(sid): int(np.sum(signer_ids == sid)) for sid in unique_signers
        }
    else:
        signer_distribution = None

    print(f"Corpus: {feature_set_full_name}")
    print(f"Samples: {n_samples}  |  Frames: {n_frames}  |  "
          f"Features/frame: {n_feats}  |  Classes: {n_classes}")
    print(f"Class distribution: {class_distribution}")
    if cv_mode == 'signer_disjoint':
        print(f"Signers: {len(signer_distribution)}  |  Distribution: "
              f"{signer_distribution}")
    print(f"Model input shape: (batch, {n_feats}, {n_frames})")
    print(f"Augmentation: {'ON' if augment else 'OFF'}")
    print(f"CV mode: {cv_mode}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Build fold splits according to the requested mode
    splits, n_folds_actual, fold_label = _build_splits(
        cv_mode, X_all, y_encoded, signer_ids, n_folds, seed)

    fold_results = []
    all_val_preds = np.zeros(n_samples, dtype=int)
    all_val_labels = np.zeros(n_samples, dtype=int)

    sweep_t0 = time.perf_counter()
    n_parameters = None
    inference_times = []
    peak_gpu_mb_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        label = fold_label(fold, val_idx)
        print(f"\n{'─' * 30} Fold {fold + 1}/{n_folds_actual} ({label}) {'─' * 30}")
        print(f"Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        fold_t0 = time.perf_counter()

        # prepare data
        X_train, X_val = prepare_fold(X_all, y_encoded, train_idx, val_idx)
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        train_ds = BSLDataset(X_train, y_train, augment=augment,
                              noise_std=noise_std,
                              time_shift_max=time_shift_max,
                              scale_range=scale_range)
        val_ds = BSLDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=len(val_idx))

        model = BSLConv1DNet(in_channels=n_feats, num_classes=n_classes,
                             dropout=dropout).to(device)
        if n_parameters is None:
            n_parameters = _count_parameters(model)

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min',
            factor=scheduler_factor, patience=scheduler_patience,
        )
        stopper = EarlyStopping(patience=patience)

        best_val_acc = 0.0
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimiser, device
            )
            val_loss, val_acc, preds, labels = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

            if stopper.step(val_loss, model):
                print(
                    f"  Early stop at epoch {epoch}  "
                    f"(best val_loss={stopper.best_loss:.4f})")
                break

        model.load_state_dict(stopper.best_model)
        _, final_acc, final_preds, final_labels = evaluate(
            model, val_loader, criterion, device
        )

        if save_models_dir:
            model_path = os.path.join(model_dir, f'fold_{fold+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved to {model_path}")

        all_val_preds[val_idx] = final_preds
        all_val_labels[val_idx] = final_labels

        try:
            inf_ms = _measure_inference_time(model, val_loader, device)
            inference_times.append(inf_ms)
        except Exception as e:
            print(f"  [warn] inference timing failed: {e}")

        fold_elapsed = time.perf_counter() - fold_t0

        if device.type == 'cuda':
            peak_gpu_mb_per_fold.append(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            )

        fold_record = {
            'fold': fold + 1,
            'fold_label': label,
            'val_acc': float(final_acc),
            'stopped_epoch': int(epoch),
            'best_val_loss': float(stopper.best_loss),
            'fold_time_s': float(fold_elapsed),
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx)),
            'sample_indices': [int(i) for i in val_idx],
            'predictions': [int(p) for p in final_preds],
            'true_labels': [int(t) for t in final_labels],
        }
        if cv_mode == 'signer_disjoint':
            # Record the per-fold class coverage so per-fold accuracies can be
            # interpreted in light of which classes the validation set
            # contained. Some signers will not have recorded all 11 letters.
            fold_class_counts = {
                str(le.inverse_transform([cls])[0]):
                    int(np.sum(y_encoded[val_idx] == cls))
                for cls in np.unique(y_encoded[val_idx])
            }
            fold_record['val_class_distribution'] = fold_class_counts

        fold_results.append(fold_record)

        print(f"\n  Fold {fold + 1} result: val_acc = {final_acc:.3f}  "
              f"(time: {fold_elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - sweep_t0

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    accs = [r['val_acc'] for r in fold_results]
    for r in fold_results:
        print(f"  {r['fold_label']}: acc={r['val_acc']:.3f}  "
              f"stopped_epoch={r['stopped_epoch']}  "
              f"best_val_loss={r['best_val_loss']:.4f}")

    print(f"\n  Mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Min: {np.min(accs):.3f}  Max: {np.max(accs):.3f}")

    print(f"\n{'─' * 70}")
    print("PER-CLASS REPORT (aggregated across all folds)")
    print(f"{'─' * 70}")
    print(classification_report(all_val_labels, all_val_preds,
                                target_names=class_names, digits=3))

    cm = confusion_matrix(all_val_labels, all_val_preds)
    print("Confusion Matrix:")
    header = "      " + "  ".join(f"{c:>3}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:3d}" for v in row)
        print(f"  {class_names[i]:>3}  {row_str}")

    cls_report_dict = classification_report(
        all_val_labels, all_val_preds,
        target_names=class_names, output_dict=True, zero_division=0,
    )

    per_class = {}
    for cls in class_names:
        if cls in cls_report_dict:
            per_class[cls] = {
                'precision': float(cls_report_dict[cls]['precision']),
                'recall': float(cls_report_dict[cls]['recall']),
                'f1': float(cls_report_dict[cls]['f1-score']),
                'support': int(cls_report_dict[cls]['support']),
            }

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if experiment_tag is None:
        experiment_tag = (
            f"{feature_set_short}_kp{keypoint_version}"
            f"_{cv_mode}"
            f"_{'aug' if augment else 'noaug'}_seed{seed}_{timestamp}"
        )

    model_dir = None
    if save_models_dir:
        model_dir = os.path.join(save_models_dir, experiment_tag)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Models will be saved to {model_dir}")

    if pipeline_config is None:
        pipeline_config = {}

    record = {
        'experiment_id': experiment_tag,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _get_git_commit(),

        'feature_set': feature_set_short,
        'feature_set_full_name': feature_set_full_name,
        'features_per_frame': int(n_feats),
        'features_total': int(n_feats * n_frames),

        'keypoint_version': keypoint_version,
        'n_frames': int(n_frames),
        'n_classes': int(n_classes),
        'n_samples': int(n_samples),
        'class_distribution': class_distribution,
        'cv_mode': cv_mode,
        'signer_distribution': signer_distribution,

        'pipeline_config': pipeline_config,

        'model': {
            'architecture': 'BSLConv1DNet',
            'conv_widths': [32, 64, 64],
            'kernel_size': 3,
            'dropout': float(dropout),
            'n_parameters': int(n_parameters) if n_parameters else None,
        },

        'training': {
            'cv_mode': cv_mode,
            'n_folds': int(n_folds_actual),
            'max_epochs': int(n_epochs),
            'batch_size': int(batch_size),
            'optimiser': 'adam',
            'lr': float(lr),
            'weight_decay': float(weight_decay),
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': float(scheduler_factor),
            'scheduler_patience': int(scheduler_patience),
            'early_stop_patience': int(patience),
            'seed': int(seed),
        },

        'augmentation': {
            'enabled': bool(augment),
            'noise_std': float(noise_std),
            'time_shift_max': int(time_shift_max),
            'scale_range': [float(scale_range[0]), float(scale_range[1])],
        },

        'fold_results': fold_results,

        'aggregate': {
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'min_acc': float(np.min(accs)),
            'max_acc': float(np.max(accs)),
            'macro_f1': float(cls_report_dict['macro avg']['f1-score']),
            'weighted_f1': float(cls_report_dict['weighted avg']['f1-score']),
        },

        'per_class': per_class,

        'confusion_matrix': {
            'labels': class_names,
            'matrix': cm.tolist(),
        },

        'compute': {
            'device': device.type,
            'gpu_name': (torch.cuda.get_device_name(device)
                         if device.type == 'cuda' else None),
            'total_training_time_s': float(total_elapsed),
            'mean_fold_time_s': float(np.mean(
                [r['fold_time_s'] for r in fold_results])),
            'mean_inference_time_per_sample_ms': (
                float(np.mean(inference_times)) if inference_times else None),
            'peak_gpu_memory_mb': (float(np.max(peak_gpu_mb_per_fold))
                                   if peak_gpu_mb_per_fold else None),
        },

        'notes': notes,
    }

    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)) or '.',
                    exist_ok=True)
        with open(results_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f"\n[result appended to {results_file}]")

    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='1D CNN k-fold CV for BSL sign classification')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to corpus_data.json')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds (stratified mode only; ignored '
                             'for signer_disjoint, which uses one fold per '
                             'signer).')
    parser.add_argument('--cv_mode', type=str, default='stratified',
                        choices=['stratified', 'signer_disjoint'],
                        help="Cross-validation mode. 'stratified' (default) "
                             "is sample-level stratified k-fold; "
                             "'signer_disjoint' is leave-one-signer-out.")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_augment', action='store_true')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keypoint_version', type=str, default='unknown',
                        help="Tag for the keypoint extraction version "
                             "(e.g. 'V1', 'V2'). Recorded in the result.")
    parser.add_argument('--experiment_tag', type=str, default=None,
                        help="Override the auto-generated experiment_id.")
    parser.add_argument('--results_file', type=str, default='results.jsonl',
                        help='Path to JSONL file the result record is '
                             'appended to. Set to "" to disable.')
    parser.add_argument('--save_models_dir', type=str, default=None,
                        help='Directory to save trained models. If set, models '
                             'will be saved as fold_N.pth in a subdirectory '
                             'named after the experiment_tag.')
    parser.add_argument('--notes', type=str, default='',
                        help='Free-text annotation stored in the record.')

    parser.add_argument('--no_cleaning_10finger', action='store_true')
    parser.add_argument('--no_cleaning_mislabel', action='store_true')
    parser.add_argument('--no_interpolation', action='store_true')
    parser.add_argument('--no_temporal_cropping', action='store_true')
    parser.add_argument('--no_temporal_normalisation', action='store_true')
    parser.add_argument('--no_spatial_normalisation', action='store_true')

    parser.add_argument('--noise_std', type=float, default=0.03)
    parser.add_argument('--time_shift_max', type=int, default=1)
    parser.add_argument('--scale_low', type=float, default=0.95)
    parser.add_argument('--scale_high', type=float, default=1.05)

    args = parser.parse_args()

    pipeline_config = {
        'cleaning_10finger': not args.no_cleaning_10finger,
        'cleaning_mislabel': not args.no_cleaning_mislabel,
        'interpolation': not args.no_interpolation,
        'temporal_cropping': not args.no_temporal_cropping,
        'temporal_normalisation': not args.no_temporal_normalisation,
        'spatial_normalisation': not args.no_spatial_normalisation,
    }

    run_kfold(
        corpus_path=args.corpus,
        n_folds=args.folds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        augment=not args.no_augment,
        seed=args.seed,
        keypoint_version=args.keypoint_version,
        pipeline_config=pipeline_config,
        experiment_tag=args.experiment_tag,
        results_file=args.results_file or None,
        noise_std=args.noise_std,
        time_shift_max=args.time_shift_max,
        scale_range=(args.scale_low, args.scale_high),
        cv_mode=args.cv_mode,
        save_models_dir=args.save_models_dir,
        notes=args.notes,
    )