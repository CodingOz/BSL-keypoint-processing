from data_cleaner import cleanCorpus
from temporal_cropping import TemporalCropping
from temporal_normalisation import TemporalNormalisor
from spatial_normalisation import SpatialNormalisor
from keypoint_interpolator import KeypointInterpolator

def formal_proccessing_pipeline(start_corpus, show_logs=False):
    '''formal pipeline for proccessing the rough keypoint data'''
    
    # Step 1: Clean the corpus using the data cleaner
    cleanCorpus(start_corpus, start_corpus + "_cleaned", show_logs=show_logs)
    
    # Step 2: keypoint interpolation
    keypoint_interpolator = KeypointInterpolator()
    keypoint_interpolator.SimpleCubicSplineCorpusGenerator(start_corpus + "_cleaned", start_corpus + "_cleaned_interpolated")
    keypoint_interpolator.estimateHandsEndsCorpusGenerator(start_corpus + "_cleaned_interpolated", start_corpus + "_cleaned_interpolated")
    
    # Step 3: temporal cropping
    temporal_cropper = TemporalCropping()
    temporal_cropper.crop_single_handed_frames_in_corpus(start_corpus + "_cleaned_interpolated", start_corpus + "_cleaned_interpolated_cropped", show_logs=show_logs)
    temporal_cropper.crop_hands_resting_together_in_corpus(start_corpus + "_cleaned_interpolated_cropped", start_corpus + "_cleaned_interpolated_cropped", show_logs=show_logs)
    temporal_cropper.crop_to_stroke_phase_in_corpus(start_corpus + "_cleaned_interpolated_cropped", start_corpus + "_cleaned_interpolated_cropped", show_logs=show_logs)

    # Step 4: temporal normalisation
    temporal_normaliser = TemporalNormalisor()
    temporal_normaliser.NormaliseCorpus(start_corpus + "_cleaned_interpolated_cropped", start_corpus + "_cleaned_interpolated_cropped_time_normilsied", frame_num=10, show_logs=show_logs)

    # level 3: spatial normalisation
    spatial_normaliser = SpatialNormalisor()
    spatial_normaliser.normaliseCorpus(start_corpus + "_cleaned_interpolated_cropped_time_normilsied", start_corpus + "_cleaned_interpolated_cropped_time_and_space_normilsied")
    
    
if __name__ == "__main__":
    start_corpus =r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\keypoints_V1"
    formal_proccessing_pipeline(start_corpus)
    
    start_corpus =r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\keypoints_V2"
    formal_proccessing_pipeline(start_corpus)
    