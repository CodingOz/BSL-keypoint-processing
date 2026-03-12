from sign_degradator import SignDegradator
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.validation_helpers import *
from anomaly_detection import AnomalyDetection
import shutil
import json
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class DegradationPattern:
    """Defines the degradation pattern for a single file."""
    num_swaps: int
    swap_sizes: List[int]
    
    def __post_init__(self):
        if len(self.swap_sizes) != self.num_swaps:
            raise ValueError(f"Number of swap_sizes must equal num_swaps ({self.num_swaps})")


class LayeredGaussianDistribution:
    """
    Generates frame sampling locations using a 4-component layered Gaussian mixture.
    This represents areas where errors are more likely to occur (as percentages through frames).
    """
    
    def __init__(self, components: int = 4, means: List[float] = None, 
                 stds: List[float] = None, weights: List[float] = None):
        """
        Initialize the layered Gaussian distribution.
        
        takes:
            components: Number of Gaussian components (default 4)
            means: Mean positions as percentages (0-1). Default evenly spaced.
            stds: Standard deviations for each component. Default 0.1.
            weights: Mixture weights for each component. Default equal.
        """
        self.components = components
        
        if means is None:
            self.means = np.linspace(0.1, 0.9, components)
        else:
            self.means = np.array(means)
            if len(self.means) != components:
                raise ValueError(f"means must have {components} values")
        
        if stds is None:
            self.stds = np.full(components, 0.1)
        else:
            self.stds = np.array(stds)
            if len(self.stds) != components:
                raise ValueError(f"stds must have {components} values")
        
        if weights is None:
            self.weights = np.full(components, 1.0 / components)
        else:
            self.weights = np.array(weights)
            if len(self.weights) != components:
                raise ValueError(f"weights must have {components} values")
            self.weights = self.weights / self.weights.sum()  # normalize
    
    def sample_region(self, total_frames: int, hole_size: int) -> Tuple[int, int]:
        """
        Sample an available_area from the layered Gaussian distribution.
        
        takes:
            total_frames: Total number of frames in the sign
            hole_size: Size of the swap to be performed
            
        Returns:
            Tuple of (start_frame, end_frame) representing available_area
        """
        # Select a component based on weights
        component_idx = np.random.choice(self.components, p=self.weights)
        
        # Sample from the selected Gaussian component
        mean_pos = self.means[component_idx]
        std_pos = self.stds[component_idx]
        
        # Sample a position (as percentage through frames)
        center_percent = np.random.normal(mean_pos, std_pos)
        center_percent = np.clip(center_percent, 0, 1)  # Clamp to [0, 1]
        
        # Convert to frame indices
        center_frame = int(center_percent * (total_frames - 1))
        
        # Define available area around this center with margin for hole_size
        margin = max(hole_size, int(0.1 * total_frames))  # At least hole_size or 10% buffer
        start = max(0, center_frame - margin)
        end = min(total_frames, center_frame + margin)
        
        # Ensure valid range
        if end <= start:
            end = min(total_frames, start + 2 * margin)
        
        return (start, end)


class CorpusDegradator:
    """
    Applies systematic degradation to a sign corpus according to a distribution pattern.
    Uses LayeredGaussianDistribution to determine where swaps occur.
    """
    
    def __init__(self, base_corpus_path: str, output_corpus_path: str,
                 gaussian_config: Dict):
        """
        Initialize the corpus degradator.
        
        takes:
            base_corpus_path: Path to clean corpus
            output_corpus_path: Path where degraded corpus will be saved
            gaussian_config: Configuration dict for Gaussian distribution with keys:
                - components: int (default 4)
                - means: List[float] (default evenly spaced)
                - stds: List[float] (default 0.1 each)
                - weights: List[float] (default equal weights)
        """
        self.base_corpus_path = base_corpus_path
        self.output_corpus_path = output_corpus_path
        self.degradator = SignDegradator()
        
        if not os.path.isdir(base_corpus_path):
            raise ValueError(f"Base corpus path '{base_corpus_path}' does not exist")
        
        if not os.path.isdir(output_corpus_path):
            os.makedirs(output_corpus_path)
        
        # Initialize Gaussian distribution
        gaussian_config = gaussian_config or {}
        self.gaussian_dist = LayeredGaussianDistribution(**gaussian_config)
        
        # Load corpus files
        self.corpus_files = self._load_corpus_files()
    
    def _load_corpus_files(self) -> List[str]:
        """Load all JSON files from corpus, sorted for reproducibility."""
        files = []
        for root, dirs, filenames in os.walk(self.base_corpus_path):
            for fname in filenames:
                if fname.lower().endswith('.json'):
                    files.append(os.path.join(root, fname))
        return sorted(files)
    
    def apply_pattern(self, patterns: List[DegradationPattern]) -> Dict:
        """
        apply degradation patterns to corpus files.
        
        takes:
            patterns: List of DegradationPattern objects, one per file in corpus
            
        Returns:
            Dictionary with degradation statistics and metadata
        """
        if len(patterns) != len(self.corpus_files):
            raise ValueError(
                f"Number of patterns ({len(patterns)}) must match corpus size ({len(self.corpus_files)})"
            )
        
        results = {
            'total_files': len(self.corpus_files),
            'degraded_files': [],
            'statistics': {}
        }
        
        for idx, (filepath, pattern) in enumerate(zip(self.corpus_files, patterns)):
            print(f"\n[{idx+1}/{len(self.corpus_files)}] Processing: {os.path.basename(filepath)}")
            print(f"  Pattern: {pattern.num_swaps} swaps, sizes: {pattern.swap_sizes}")
            
            try:
                # Load the original sign data
                with open(filepath, 'r', encoding='utf-8') as f:
                    sign_data = json.load(f)
                
                total_frames = len(sign_data.get('frames', []))
                degraded_data = sign_data
                swap_info = []
                
                # Apply each swap
                used_frames = set()
                for swap_idx, swap_size in enumerate(pattern.swap_sizes):
                    attempts = 0
                    while attempts < 50:
                        available_area = self.gaussian_dist.sample_region(total_frames, swap_size)
                        # peek at what frameSwap would pick — or just check area overlap
                        area_frames = set(range(available_area[0], available_area[1]))
                        if not area_frames & used_frames:
                            break
                        attempts += 1
                    used_frames |= area_frames
                    # Apply the swap
                    degraded_data = self.degradator.frameSwap(
                        degraded_data, 
                        hole_size=swap_size, 
                        available_area=available_area
                    )
                    
                    # Track the swap
                    swap_info.append({
                        'swap_index': swap_idx,
                        'size': swap_size,
                        'available_area': available_area
                    })
                
                # Save the degraded file
                output_path = os.path.join(
                    self.output_corpus_path, 
                    os.path.basename(filepath)
                )
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(degraded_data, f, indent=2)
                
                results['degraded_files'].append({
                    'filename': os.path.basename(filepath),
                    'output_path': output_path,
                    'pattern': {
                        'num_swaps': pattern.num_swaps,
                        'swap_sizes': pattern.swap_sizes
                    },
                    'total_frames': total_frames,
                    'swap_details': swap_info
                })
                
            except Exception as e:
                print(f"  ERROR processing {filepath}: {e}")
                results['degraded_files'].append({
                    'filename': os.path.basename(filepath),
                    'error': str(e)
                })
        
        # Summary statistics
        results['statistics'] = self._compute_statistics(results['degraded_files'])
        
        return results
    
    def _compute_statistics(self, degraded_files: List[Dict]) -> Dict:
        """Compute summary statistics of degradation."""
        successful = [f for f in degraded_files if 'error' not in f]
        
        total_swaps = sum(
            f['pattern']['num_swaps'] for f in successful
        )
        swap_size_distribution = {}
        for f in successful:
            for size in f['pattern']['swap_sizes']:
                swap_size_distribution[size] = swap_size_distribution.get(size, 0) + 1
        
        return {
            'successful_degradations': len(successful),
            'failed_degradations': len(degraded_files) - len(successful),
            'total_swaps': total_swaps,
            'swap_size_distribution': swap_size_distribution
        }


def create_default_degradation_patterns() -> List[DegradationPattern]:
    """
    degradation pattern for 22 files:
    - 5 files with 4 swaps of size 1
    - 2 files with 2 swaps of size 2
    - 5 files with 2 swaps of size 2
    - 2 files with 4 swaps of size 2
    - 4 files with 2 swaps of size 3
    - 2 files with 2 swaps of size 4
    - 2 files with 1 swap of size 5
    
    Returns:
        List of DegradationPattern objects
    """
    patterns = []
    
    # 5 files with 4 swaps of size 1
    for _ in range(5):
        patterns.append(DegradationPattern(num_swaps=4, swap_sizes=[1, 1, 1, 1]))

    # 7 files with 2 swaps of size 2
    for _ in range(7):
        patterns.append(DegradationPattern(num_swaps=2, swap_sizes=[2, 2]))
    
    # 2 files with 4 swaps of size 2
    for _ in range(2):
        patterns.append(DegradationPattern(num_swaps=4, swap_sizes=[2, 2, 2, 2]))
    
    # 4 files with 2 swaps of size 3
    for _ in range(4):
        patterns.append(DegradationPattern(num_swaps=2, swap_sizes=[3, 3]))
    
    # 2 files with 2 swaps of size 4
    for _ in range(2):
        patterns.append(DegradationPattern(num_swaps=2, swap_sizes=[4, 4]))
    
    # 2 files with 1 swap of size 5
    for _ in range(2):
        patterns.append(DegradationPattern(num_swaps=1, swap_sizes=[5]))
    
    return patterns

    
    
if __name__ == "__main__":
    gaussian_config = {
        'components': 4,
        'means': [0.0897, 0.2921, 0.6762, 0.8856],  # Center positions as percentages
        'stds': [0.0595, 0.0729, 0.1061, 0.0717],  # Standard deviations
        'weights': [0.284, 0.223, 0.228, 0.265]  # Emphasis on middle regions
    }
    
    base_corpus_path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validated_SubCorpus"
    output_corpus_path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\swapped_hands_corpus"
    
    
    patterns = create_default_degradation_patterns()
    
    degradator = CorpusDegradator(base_corpus_path, 
                                  output_corpus_path, 
                                  gaussian_config=gaussian_config)
    
    results = degradator.apply_pattern(patterns)