import os
import json
import numpy as np
from feature_extraction import FeatureExtraction
 
 
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)
 
 
class CorpusFeatureGenerator:
    """Generate 6 different feature corpus sets from organized JSON files."""
 
    def __init__(self, base_corpus_path, output_base_path):
        self.base_corpus_path = base_corpus_path
        self.output_base_path = output_base_path
        self.json_objects = []
        self.load_all_jsons()
 
    def load_all_jsons(self):
        """Load all JSON files from subdirectories organized by sign label."""
        print(f"Loading JSON files from {self.base_corpus_path}...")
 
        # Get all sign directories
        if not os.path.exists(self.base_corpus_path):
            print(f"Error: Path does not exist: {self.base_corpus_path}")
            return
 
        sign_dirs = [d for d in os.listdir(self.base_corpus_path)
                     if os.path.isdir(os.path.join(self.base_corpus_path, d))]
 
        for sign_dir in sign_dirs:
            sign_path = os.path.join(self.base_corpus_path, sign_dir)
            json_files = [f for f in os.listdir(
                sign_path) if f.endswith('.json')]
 
            print(f"  Sign '{sign_dir}': {len(json_files)} files")
 
            for json_file in json_files:
                json_path = os.path.join(sign_path, json_file)
                # The signer is identified by the filename stem. The same
                # stem appearing in multiple sign-letter folders represents
                # the same signer recording different letters; distinct
                # stems represent distinct signers (including the manual-N
                # files, each of which is a separate signer).
                signer_id = os.path.splitext(json_file)[0]
                try:
                    with open(json_path, 'r') as f:
                        obj = json.load(f)
                        self.json_objects.append({
                            'sign': sign_dir,
                            'filename': json_file,
                            'signer_id': signer_id,
                            'data': obj
                        })
                except Exception as e:
                    print(f"    Error loading {json_file}: {e}")
 
        print(f"Total JSON files loaded: {len(self.json_objects)}\n")
 
    def create_output_directory(self, corpus_name):
        """Create output directory for corpus if it doesn't exist."""
        output_path = os.path.join(self.output_base_path, corpus_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path
 
    def extract_and_save_corpus(
            self,
            corpus_name,
            extraction_method,
            num_features):
        """Extract features and save corpus."""
        print(f"Generating {corpus_name}...")
        print(f"  Expected features per file: {num_features}")
 
        output_path = self.create_output_directory(corpus_name)
 
        # Create a dummy FeatureExtraction instance
        fe = FeatureExtraction.__new__(FeatureExtraction)
 
        corpus_data = []
 
        for idx, obj_info in enumerate(self.json_objects):
            if (idx + 1) % 10 == 0:
                print(
                    f"  Processing file {idx + 1}/{len(self.json_objects)}...")
 
            try:
                features = extraction_method(fe, obj_info['data'])
 
                # Convert numpy arrays to lists properly
                if isinstance(features, np.ndarray):
                    if features.ndim == 1:
                        # 1D array - convert to list
                        features = features.tolist()
                    elif features.ndim == 2:
                        # 2D array - convert to list of lists, preserving
                        # structure
                        features = features.tolist()
                    else:
                        # Higher dimensions - flatten then convert
                        features = features.flatten().tolist()
 
                # Ensure it's a list that can be serialized
                if not isinstance(features, list):
                    features = [features]
 
                corpus_item = {
                    'sign': obj_info['sign'],
                    'filename': obj_info['filename'],
                    'signer_id': obj_info['signer_id'],
                    'features': features,
                    'num_features': len(features)
                }
 
                corpus_data.append(corpus_item)
 
            except Exception as e:
                import traceback
                print(f"    Error processing {obj_info['filename']}: {e}")
 
        # Save corpus data
        output_file = os.path.join(output_path, 'corpus_data.json')
        with open(output_file, 'w') as f:
            json.dump(corpus_data, f, cls=NumpyEncoder)
 
        # Save metadata, including the per-corpus signer count for downstream
        # leave-one-signer-out analysis.
        unique_signers = sorted({item['signer_id'] for item in corpus_data})
        metadata = {
            'corpus_name': corpus_name,
            'total_files': len(corpus_data),
            'features_per_file': num_features,
            'signs': list(set(item['sign'] for item in corpus_data)),
            'n_signers': len(unique_signers),
            'signer_ids': unique_signers,
        }
 
        metadata_file = os.path.join(output_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
 
        print(f"  [OK] Corpus saved to {output_path}")
        print(f"  [OK] Files processed: {len(corpus_data)}")
        print(f"  [OK] Signs included: {', '.join(metadata['signs'])}")
        print(f"  [OK] Distinct signers: {metadata['n_signers']}\n")
 
        return corpus_data
 
    def generate_all_corpuses(self):
        """Generate all 6 feature corpuses."""
        print("=" * 70)
        print("CORPUS GENERATION: Creating 6 Feature Sets")
        print("=" * 70 + "\n")
 
        corpuses = [{'name': 'v0_raw_coordinates_84',
                     'method': 'extract_feature_set_0_raw_coordinates',
                     'features': 840,
                     'description': 'Raw X and Y coordinates for each point'},
                    {'name': 'v1_all_proximity_and_angles_9090',
                     'method': 'extract_feature_set_1_combined',
                     'features': 9090,
                     'description': 'All proximity features (8610) + all angle features (480)'},
                    {'name': 'v2_proximity_only_8610',
                     'method': 'extract_feature_set_2_proximity_only',
                     'features': 8610,
                     'description': 'Only proximity features'},
                    {'name': 'v3_tips_palms_with_angles_mixed_1140',
                     'method': 'extract_feature_set_3_tips_and_palms_with_angles',
                     'features': 1140,
                     'description': 'Fingertips + palms distances + angles'},
                    {'name': 'v4_12points_angles_and_distance_850',
                     'method': 'extract_feature_set_4_tips_and_palms_with_angles',
                     'features': 850,
                     'description': '12-point angles and distances only'},
                    {'name': 'v5_12points_distances_only_660',
                     'method': 'extract_feature_set_5_tips_and_palms_distances_only',
                     'features': 660,
                     'description': '12-point distances only (no angles)'},
                    {'name': 'v6_minimal_index_palm_pinkie_850',
                     'method': 'extract_feature_set_6_extreme_minimal',
                     'features': 850,
                     'description': 'Minimal: index, palm, pinkie distances'},
                    {'name': 'v7_interhand_distances_only_360',
                     'method': 'extract_feature_set_7_interhand_distances_only',
                     'features': 360,
                     'description': 'Inter-hand distances only (no intra-hand distances)'}]
 
        results = {}
 
        for corpus_config in corpuses:
            print(f"Corpus: {corpus_config['name']}")
            print(f"Description: {corpus_config['description']}")
            print(f"Features: {corpus_config['features']}")
            print("-" * 70)
 
            method_name = corpus_config['method']
            method = getattr(FeatureExtraction, method_name)
 
            result = self.extract_and_save_corpus(
                corpus_config['name'],
                method,
                corpus_config['features']
            )
 
            results[corpus_config['name']] = {
                'total_files': len(result),
                'features': corpus_config['features'],
                'description': corpus_config['description']
            }
 
        # Print summary
        print("=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        for name, info in results.items():
            print(f"\n{name}")
            print(f"  Files: {info['total_files']}")
            print(f"  Features per file: {info['features']}")
            print(f"  Description: {info['description']}")
 
 
if __name__ == '__main__':
    base_corpus_path = r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\All_keypoint_data\keypoints_V2_s5_space_norm'
    output_base_path = r'C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\features\feature_corpuses_V2_with_signer_ids'

    generator = CorpusFeatureGenerator(base_corpus_path, output_base_path)
    generator.generate_all_corpuses()