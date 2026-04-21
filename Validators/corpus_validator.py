from .keypoint_validator import CubicSplineKeyPointInterpolator, SignLengths
import json
import os
from dataclasses import dataclass
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

@dataclass
class MovementAnomalyDetectionResult:
    file_path: str
    hand: str
    frame: list[int]
    # the percentage through based around the first and last hand seen
    percentage_through: list[float]
    
    
@dataclass
class AnomalyDistributionResult:
    all_percentages: list[float]
    n_anomalies: int
    component_means: list[float]
    component_stds: list[float]
    component_weights: list[float]
    bic: float
    aic: float
    is_bimodal: bool

class CorpusValidator:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.validators = []
        
        # dictionary holding 2d array of palm coordinates for each json file
        self.all_palm_locations = {}
        
        # dictionary holding 1d array of palm momentums for each json file
        self.all_palm_momentums = {}
        
        # dictionary holding 1d array of palm accelerations for each json file
        self.all_palm_accelerations = {}
        
        self.timings = []
        
        # makes an array of keypoint validators for every json in the corpus        
        for filename in os.listdir(corpus_path):
            
            if filename.endswith(".json"):
                filepath = os.path.join(corpus_path, filename)
                try:
                    temp = CubicSplineKeyPointInterpolator(filepath)
                    self.validators.append(temp)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
            
            # loops through any subdirectories
            elif os.path.isdir(os.path.join(corpus_path, filename)):
                subdir_path = os.path.join(corpus_path, filename)
                for sub_filename in os.listdir(subdir_path):
                    if sub_filename.endswith(".json"):
                        filepath = os.path.join(subdir_path, sub_filename)
                        try:
                            temp = CubicSplineKeyPointInterpolator(filepath)
                            self.validators.append(temp)
                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")

                    
        print(f"{len(self.validators)} files found in corpus")
        
    def getAllPalmLocations(self):
        ''' checks if palm directory is empty '''
        if len(self.all_palm_locations) > 0:
            return self.all_palm_locations
        
        for validator in self.validators:
            self.all_palm_locations[validator.filepath] = validator.getFilledPalmCenters()
        return self.all_palm_locations
    
    def getAllPalmMomentums(self):
        ''' checks if palm directory is empty '''
        if len(self.all_palm_momentums) > 0:
            return self.all_palm_momentums
        
        for validator in self.validators:
            self.all_palm_momentums[validator.filepath] = validator.getEstimatedMomentums()
        return self.all_palm_momentums
    
    def getAllPalmAccelerations(self):
        ''' checks if palm directory is empty '''
        if len(self.all_palm_accelerations) > 0:
            return self.all_palm_accelerations
        
        for validator in self.validators:
            self.all_palm_accelerations[validator.filepath] = validator.getEstimatedAccelerations()
        return self.all_palm_accelerations
    
    def getAllTimings(self):
        ''' returns a list of Sign_lengths dataclasses, one per file in the corpus, 
        containing the first and last frame where a hand is detected '''
        if len(self.timings) > 0:
            return self.timings
        for validator in self.validators:
            self.timings.append(validator.getSignLengths())
        return self.timings
  
    def detectMovementAnomalies(
        self,
        acceleration_threshold: float = 0.2,
        momentum_threshold: float = 0.15,
    ):
        """
        Detects frames where acceleration > acceleration_threshold OR
        momentum > momentum_threshold, then records what percentage through
        the sign (first_hand → last_hand) each anomalous frame sits.

        Returns: one MovementAnomalyDetectionResult per (file, hand) pair
        that contains at least one anomaly.
        """
        # ensure all data is populated
        self.getAllPalmMomentums()
        self.getAllPalmAccelerations()
        self.getAllTimings()

        # build a fast lookup: filepath -> SignLengths dataclass
        timing_lookup: dict[str, any] = {
            t.filepath: t for t in self.timings
        }

        results: list[MovementAnomalyDetectionResult] = []

        for filepath, acc_by_hand in self.all_palm_accelerations.items():
            timing = timing_lookup.get(filepath)
            if timing is None:
                continue  # No timing info available for this file

            sign_start = timing.first_hand
            sign_end   = timing.last_hand
            sign_span  = sign_end - sign_start  # frames spanning the sign

            mom_by_hand = self.all_palm_momentums.get(filepath, {})

            # The validators return a list-per-frame of dicts. Convert to
            # a dict mapping hand -> list[float] (per-frame magnitudes)
            if isinstance(acc_by_hand, list):
                acc_map = {'left': [], 'right': []}
                for item in acc_by_hand:
                    # item is in form {'acceleration': {'left': {...}, 'right': {...}}}
                    data = item.get('acceleration') if isinstance(item, dict) else None
                    for side in ('left', 'right'):
                        val = 0.0
                        if isinstance(data, dict):
                            hand_info = data.get(side, {})
                            if isinstance(hand_info, dict) and 'acceleration' in hand_info:
                                try:
                                    val = float(hand_info['acceleration'])
                                except Exception:
                                    val = 0.0
                        acc_map[side].append(val)
                acc_by_hand = acc_map

            if isinstance(mom_by_hand, list):
                mom_map = {'left': [], 'right': []}
                for item in mom_by_hand:
                    # item in form {'left': {}, 'right': {}}
                    for side in ('left', 'right'):
                        val = 0.0
                        if isinstance(item, dict):
                            hand_info = item.get(side, {})
                            if isinstance(hand_info, dict) and 'magnitude' in hand_info:
                                try:
                                    val = float(hand_info['magnitude'])
                                except Exception:
                                    val = 0.0
                        mom_map[side].append(val)
                mom_by_hand = mom_map

            all_hands = set(acc_by_hand.keys()) | set(mom_by_hand.keys())

            for hand in all_hands:
                acc_values = acc_by_hand.get(hand, [])
                mom_values = mom_by_hand.get(hand, [])

                # align lengths (use the longer array, treat missing tail as 0)
                n_frames = max(len(acc_values), len(mom_values))

                anomalous_frames: list[int] = []
                percentages: list[float] = []

                for frame_idx in range(n_frames):
                    acc = acc_values[frame_idx] if frame_idx < len(acc_values) else 0.0
                    mom = mom_values[frame_idx] if frame_idx < len(mom_values) else 0.0

                    if acc > acceleration_threshold or mom > momentum_threshold:
                        # only record frames within the signed region
                        if sign_start <= frame_idx <= sign_end:
                            anomalous_frames.append(frame_idx)

                            if sign_span > 0:
                                pct = (frame_idx - sign_start) / sign_span * 100.0
                            else:
                                pct = 0.0

                            percentages.append(round(pct, 2))

                if anomalous_frames:
                    results.append(
                        MovementAnomalyDetectionResult(
                            file_path=filepath,
                            hand=hand,
                            frame=anomalous_frames,
                            percentage_through=percentages,
                        )
                    )

        return results
    
    def analyseAnomalyDistribution(
        self,
        anomaly_results: list[MovementAnomalyDetectionResult],
        max_components: int = 4,
        plot: bool = True,
        plot_path: str | None = None,):
        """
        fits a Gaussian Mixture Model to the distribution of anomaly
        percentage-through values and returns component statistics.

        takes:
            anomaly_results: output of detect_movement_anomalies()
            max_components: upper bound on GMM components to trial (BIC-selected)
            plot: whether to show/save a diagnostic figure
            plot_path: if given, saves figure here instead of showing it
        returns:
            result of the distribution analisise 
            in the form of a AnomalyDistributionResult class
        """

        # flatten all percentages into one array
        all_pcts = np.array([
            pct
            for result in anomaly_results
            for pct in result.percentage_through
        ])

        if len(all_pcts) == 0:
            raise ValueError("No anomaly percentages found — run detect_movement_anomalies() first.")

        X = all_pcts.reshape(-1, 1)   # sklearn expects (n_samples, n_features)

        # trial GMMs with 1..max_components, select by BIC
        bic_scores = {}
        aic_scores = {}
        fitted_models = {}

        for n in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                random_state=42,
                n_init=10,              # multiple restarts to avoid local minima
            )
            gmm.fit(X)
            bic_scores[n] = gmm.bic(X)
            aic_scores[n] = gmm.aic(X)
            fitted_models[n] = gmm

        best_n = min(bic_scores, key=bic_scores.get)
        best_gmm = fitted_models[best_n]

        # sort components by mean for consistent output
        order = np.argsort(best_gmm.means_.flatten())
        means  = best_gmm.means_.flatten()[order].tolist()
        stds   = np.sqrt(best_gmm.covariances_.flatten())[order].tolist()
        weights = best_gmm.weights_[order].tolist()

        is_bimodal = best_n >= 2

        # plotting
        if plot or plot_path:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            pct_axis = np.linspace(0, 100, 500)

            # histogram KDE and GMM overlay
            ax = axes[0]
            ax.hist(all_pcts, bins=40, range=(0, 100), density=True,
                    alpha=0.35, color="steelblue", label="Observed anomalies")

            # KDE (non-parametric reference)
            if len(all_pcts) > 1:
                kde = gaussian_kde(all_pcts, bw_method="scott")
                ax.plot(pct_axis, kde(pct_axis), color="steelblue",
                        linewidth=2, label="KDE")

            # full GMM density
            gmm_density = np.zeros_like(pct_axis)
            component_colors = plt.cm.Set1.colors
            for i, (m, s, w) in enumerate(zip(means, stds, weights)):
                component_density = w * norm.pdf(pct_axis, m, s)
                gmm_density += component_density
                ax.fill_between(pct_axis, component_density, alpha=0.25,
                                color=component_colors[i],
                                label=f"Component {i+1}: μ={m:.1f}%, σ={s:.1f}%, w={w:.2f}")

            ax.plot(pct_axis, gmm_density, color="crimson", linewidth=2.5,
                    linestyle="--", label=f"GMM total (k={best_n})")

            # annotate suspected regions
            for lo, hi in [(20, 30), (70, 80)]:
                ax.axvspan(lo, hi, alpha=0.08, color="gold",
                        label=f"Suspected zone {lo}–{hi}%" if lo == 20 else "")

            ax.set_xlabel("Percentage through sign (%)")
            ax.set_ylabel("Density")
            ax.set_title(f"Anomaly Distribution  (n={len(all_pcts):,} frames, best k={best_n})")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 100)

            # BIC/AIC vs number of components —
            ax2 = axes[1]
            ns = list(bic_scores.keys())
            ax2.plot(ns, [bic_scores[n] for n in ns], "o-", label="BIC", color="crimson")
            ax2.plot(ns, [aic_scores[n] for n in ns], "s--", label="AIC", color="steelblue")
            ax2.axvline(best_n, color="gray", linestyle=":", label=f"Selected k={best_n}")
            ax2.set_xlabel("Number of GMM components")
            ax2.set_ylabel("Score (lower = better)")
            ax2.set_title("Model Selection: BIC / AIC")
            ax2.set_xticks(ns)
            ax2.legend()

            plt.tight_layout()
            if plot_path:
                plt.savefig(plot_path, dpi=150)
                print(f"Figure saved to {plot_path}")
            else:
                plt.show()

        # summary
        print(f"\n Anomaly Distribution Analysis ")
        print(f"  Total anomalous frames : {len(all_pcts):,}")
        print(f"  Best GMM k             : {best_n}  (BIC={bic_scores[best_n]:.1f})")
        print(f"  Bimodal?               : {is_bimodal}")
        for i, (m, s, w) in enumerate(zip(means, stds, weights)):
            print(f"  Component {i+1}           : μ={m:.2f}%  σ={s:.2f}%  weight={w:.3f}")

        # check overlap with suspected zones
        for lo, hi in [(20, 30), (70, 80)]:
            zone_pcts = all_pcts[(all_pcts >= lo) & (all_pcts <= hi)]
            zone_share = len(zone_pcts) / len(all_pcts) * 100
            print(f"  Anomalies in {lo}-{hi}%    : {len(zone_pcts):,}  ({zone_share:.1f}% of total)")
        print()

        return AnomalyDistributionResult(
            all_percentages=all_pcts.tolist(),
            n_anomalies=len(all_pcts),
            component_means=means,
            component_stds=stds,
            component_weights=weights,
            bic=bic_scores[best_n],
            aic=aic_scores[best_n],
            is_bimodal=is_bimodal,
        )
        
    def getAllHandDistances(self, midpoint=0.5) -> dict:
        """
        Returns fixed-midpoint distances for every file in the corpus.
        
        Returns:
            {
                filepath: list[{'left': float|None, 'right': float|None}]  # one dict per frame
            }
        """
        if not hasattr(self, '_all_hand_distances'):
            self._all_hand_distances = {}

        if len(self._all_hand_distances) > 0:
            return self._all_hand_distances

        for validator in self.validators:
            self._all_hand_distances[validator.filepath] = \
                validator.findAllHandDistancesFromMidpoint(midpoint=midpoint)

        return self._all_hand_distances


    def getAllAdaptiveDistances(self) -> dict:
        """
        Returns adaptive-midpoint distances and the derived midpoint for every file.

        Returns:
            {
                filepath: {
                    'distances':         list[{'left': float|None, 'right': float|None}],
                    'adaptive_midpoint': float
                }
            }
        """
        if not hasattr(self, '_all_adaptive_distances'):
            self._all_adaptive_distances = {}

        if len(self._all_adaptive_distances) > 0:
            return self._all_adaptive_distances

        for validator in self.validators:
            distances, midpoint = validator.findAllHandDistancesFromAdaptiveMidpoint()
            self._all_adaptive_distances[validator.filepath] = {
                'distances':         distances,
                'adaptive_midpoint': midpoint
            }

        return self._all_adaptive_distances

