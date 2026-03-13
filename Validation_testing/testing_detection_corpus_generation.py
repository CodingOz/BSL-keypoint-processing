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
    
    def sample_region(self, total_frames: int, hole_size: int,
                  sign_data: dict = None) -> Tuple[int, int]:
        """
        Sample an available_area from the layered Gaussian distribution.
        If sign_data is provided, resamples until the centre lands in a 
        region with hand data, preserving the distributional shape while
        avoiding empty frame regions.
        """
        # pre-compute hand-present frame indices once
        if sign_data is not None:
            frames_list = sign_data.get('frames', [])
            frames_with_hands = {
                i for i, f in enumerate(frames_list)
                if f.get('hands', {}).get('left') or f.get('hands', {}).get('right')
            }
        else:
            frames_with_hands = None

        for attempt in range(200):
            component_idx  = np.random.choice(self.components, p=self.weights)
            mean_pos       = self.means[component_idx]
            std_pos        = self.stds[component_idx]

            center_percent = np.clip(np.random.normal(mean_pos, std_pos), 0, 1)
            center_frame   = int(center_percent * (total_frames - 1))

            # if we have hand data, only accept centres near visible hands
            if frames_with_hands is not None:
                margin = max(hole_size, int(0.1 * total_frames))
                window = set(range(max(0, center_frame - margin),
                                min(total_frames, center_frame + margin)))
                if not window & frames_with_hands:
                    continue   # resample — this centre is in an empty region

            margin = max(hole_size, int(0.1 * total_frames))
            start  = max(0, center_frame - margin)
            end    = min(total_frames, center_frame + margin)
            if end <= start:
                end = min(total_frames, start + 2 * margin)

            return (start, end)

        # fallback: return the full range and let the outer loop find the best spot
        return (0, total_frames)
    
    
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
        '''Apply Gaussian-distributed degradation patterns to corpus files.

        takes:
            patterns: List of DegradationPattern objects, one per file in corpus
        returns:
            Dictionary with degradation statistics and metadata
        '''
        
        return self._apply_with_sampler(
            patterns,
            output_path   = self.output_corpus_path,
            sampler       = self._sample_region_gaussian,
            sampler_label = 'gaussian',
        )
    
    
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
    
    
    def _sample_region_gaussian(self, total_frames: int, swap_size: int, sign_data: dict = None) -> tuple[int, int]:
        """Existing Gaussian-based placement — delegates to LayeredGaussianDistribution."""
        return self.gaussian_dist.sample_region(total_frames, swap_size, sign_data=sign_data)


    def _sample_region_uniform(self, total_frames: int, swap_size: int, sign_data: dict = None) -> tuple[int, int]:
        """
        Places swaps with equal probability at any valid position in the sign.
        Returns the full frame range as available_area so frameSwap samples freely.
        """
        return (0, total_frames)

    def _sample_region_stratified(self, total_frames: int, swap_size: int,
                                n_strata: int = 5,
                                sign_data: dict = None) -> tuple[int, int]:
        """
        Picks a stratum weighted by hand coverage within each band.
        If sign_data is provided, strata with more visible hand frames
        are proportionally more likely to be selected.
        """
        stratum_width = total_frames // n_strata
        strata = [
            (i * stratum_width, min((i + 1) * stratum_width, total_frames))
            for i in range(n_strata)
        ]

        if sign_data is not None:
            frames = sign_data.get('frames', [])
            weights = []
            for start, end in strata:
                hand_frames = sum(
                    1 for f in frames[start:end]
                    if (f.get('hands', {}).get('left') or 
                        f.get('hands', {}).get('right'))
                )
                weights.append(max(hand_frames, 1))  # avoid zero weights
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
        else:
            weights = None  # uniform across strata

        chosen = np.random.choice(n_strata, p=weights)
        return strata[chosen]
    
    
    def _apply_with_sampler(
        self,
        patterns:      list,
        output_path:   str,
        sampler:       callable,
        sampler_label: str,
    ) -> dict:
        """
        Core degradation loop — shared by all apply_* methods.
        Placement is fully determined by the sampler callable:
            sampler(total_frames, swap_size) -> (start, end)
        """
        if len(patterns) != len(self.corpus_files):
            raise ValueError(
                f"Number of patterns ({len(patterns)}) must match "
                f"corpus size ({len(self.corpus_files)})"
            )

        os.makedirs(output_path, exist_ok=True)
        
        file_pattern_pairs = list(zip(self.corpus_files, patterns))
        np.random.shuffle(file_pattern_pairs)
        
        results = {
            'total_files':    len(self.corpus_files),
            'sampler':        sampler_label,
            'degraded_files': [],
            'statistics':     {}
        }

        for idx, (filepath, pattern) in enumerate(zip(self.corpus_files, patterns)):
            print(f"\n[{idx+1}/{len(self.corpus_files)}] {os.path.basename(filepath)}"
                f"  ({sampler_label})")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sign_data = json.load(f)

                total_frames  = len(sign_data.get('frames', []))
                degraded_data = sign_data
                swap_info     = []
                used_frames   = set()

                for swap_idx, swap_size in enumerate(pattern.swap_sizes):
                    frames_list = sign_data.get('frames', [])
                    
                    # pre-compute which frames have hand data — do this once per file outside the loop
                    frames_with_hands = {
                        i for i, f in enumerate(frames_list)
                        if f.get('hands', {}).get('left') or f.get('hands', {}).get('right')
                    }

                    best_start    = None
                    best_coverage = -1

                    for attempt in range(200):
                        candidate_area = sampler(total_frames, swap_size, sign_data=sign_data)
                        area_frames    = set(range(candidate_area[0], candidate_area[1]))

                        if area_frames & used_frames:
                            continue

                        # find the best contiguous start within this area
                        for start in range(candidate_area[0], 
                                        min(candidate_area[1], total_frames - swap_size + 1)):
                            window    = set(range(start, start + swap_size))
                            coverage  = len(window & frames_with_hands)
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_start    = start
                            if coverage >= swap_size:   # perfect — stop immediately
                                break

                        if best_coverage >= swap_size * 0.5:
                            break

                    if best_start is None or best_coverage == 0:
                        print(f"    Skipping swap {swap_idx+1} — no frames with hand data available")
                        continue

                    if best_coverage < swap_size * 0.5:
                        print(f"    Warning: swap {swap_idx+1} has only {best_coverage}/{swap_size} "
                            f"frames with hand data")

                    # pass a tight window so frameSwap is forced to use best_start
                    tight_area = (best_start, best_start + swap_size + 1)
                    used_frames |= set(range(best_start, best_start + swap_size))

                    degraded_data = self.degradator.frameSwap(
                        degraded_data,
                        hole_size=swap_size,
                        available_area=tight_area,
                    )
                    swap_info.append({
                        'swap_index':    swap_idx,
                        'size':          swap_size,
                        'available_area': tight_area,
                        'hand_coverage': best_coverage,
                    })
                
                out = os.path.join(output_path, os.path.basename(filepath))
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(degraded_data, f, indent=2)

                results['degraded_files'].append({
                    'filename':    os.path.basename(filepath),
                    'output_path': out,
                    'pattern':     {'num_swaps': pattern.num_swaps,
                                    'swap_sizes': pattern.swap_sizes},
                    'total_frames': total_frames,
                    'swap_details': swap_info,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results['degraded_files'].append({
                    'filename': os.path.basename(filepath),
                    'error':    str(e),
                })

        results['statistics'] = self._compute_statistics(results['degraded_files'])
        return results

    #  public apply methods
    
    def apply_pattern_uniform(self, patterns: list, output_path: str = None) -> dict:
        """
        Uniform random placement — no distributional assumption.
        Swaps can land anywhere in the sign with equal probability.
        Use this as a distribution-free baseline to isolate whether
        Gaussian placement artificially inflates or deflates detector scores.
        """
        out = output_path or self.output_corpus_path.rstrip('/\\') + '_uniform'
        return self._apply_with_sampler(
            patterns,
            output_path   = out,
            sampler       = self._sample_region_uniform,
            sampler_label = 'uniform',
        )

    def apply_pattern_stratified(self, patterns: List[DegradationPattern], 
                                n_strata: int = 5,
                                output_path: str = None) -> Dict:
        """
        Stratified placement — sign divided into equal bands, one picked per swap.
        Guarantees swaps are spread across the sign rather than clustering,
        which tests detector performance independently of positional bias.
        Sign data is passed per-file automatically so strata are weighted
        by hand coverage for each individual file.
        """
        out = output_path or self.output_corpus_path.rstrip('/\\') + '_stratified'
        sampler = lambda total, size, sign_data=None: self._sample_region_stratified(
            total, size, n_strata=n_strata, sign_data=sign_data
        )
        return self._apply_with_sampler(
            patterns,
            output_path   = out,
            sampler       = sampler,
            sampler_label = f'stratified(k={n_strata})',
        )


def create_default_degradation_patterns() -> List[DegradationPattern]:
    """
    degradation pattern for 22 files:
    - 7 files with 4 swaps of size 1
    - 5 files with 2 swaps of size 2
    - 2 files with 4 swaps of size 2
    - 4 files with 2 swaps of size 3
    - 2 files with 2 swaps of size 4
    - 2 files with 1 swap of size 5
    
    Returns:
        List of DegradationPattern objects
    """
    patterns = []
    
    # 7 files with 4 swaps of size 1
    for _ in range(7):
        patterns.append(DegradationPattern(num_swaps=4, swap_sizes=[1, 1, 1, 1]))

    # 5 files with 2 swaps of size 2
    for _ in range(5):
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
    output_corpus_path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Uniform"
    
    
    patterns = create_default_degradation_patterns()
    
    degradator = CorpusDegradator(base_corpus_path, 
                                  output_corpus_path, 
                                  gaussian_config=gaussian_config)
    
    #results = degradator.apply_pattern_uniform(patterns)
    
    
    output_corpus_path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Testing_Corpus_Stratified"
    
    
    patterns = create_default_degradation_patterns()
    
    degradator = CorpusDegradator(base_corpus_path, 
                                  output_corpus_path, 
                                  gaussian_config=gaussian_config)
    
    #results = degradator.apply_pattern_stratified(patterns, n_strata=5)
    
    
    output_corpus_path = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Validation_testing\Test_corpus_with_gaussian_using_momentum_heuristics"
    
    degradator = CorpusDegradator(base_corpus_path, 
                                  output_corpus_path, 
                                  gaussian_config=gaussian_config)
    
    results = degradator.apply_pattern(patterns)
    