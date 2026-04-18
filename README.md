# BSL Keypoint Processing

Processing and analysis pipeline for British Sign Language (BSL) keypoint data extracted from video.

## Features

- **Extraction** - Extract hand and body keypoints from videos
- **Cleaning** - Remove noise and validate keypoint data
- **Interpolation** - Fill gaps in temporal sequences
- **Normalization** - Standardize spatial and temporal dimensions
- **Analysis** - Feature extraction and anomaly detection
- **Validation** - Error detection and corpus validation

## Core Modules

| Module | Purpose |
|--------|---------|
| `keypoint_extractor.py` | Extract keypoints from video frames |
| `data_cleaner.py` | Clean and validate keypoint data |
| `key_point_interpolator.py` | Interpolate missing frames |
| `spatial_normalisation.py` | Normalize spatial dimensions |
| `temporal_normalisation.py` | Normalize temporal sequences |
| `feature_extraction.py` | Generate feature vectors |
| `anomaly_detection.py` | Detect anomalies in keypoints |

## Usage

See individual module docstrings for detailed usage.
