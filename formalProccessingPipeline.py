from data_cleaner import cleanCorpus
from temporal_cropping import TemporalCropping
from temporal_normalisation import TemporalNormalisor
from spatial_normalisation import SpatialNormalisor
from keypoint_interpolator import KeypointInterpolator

def formalProcessingPipeline(start_corpus, show_logs=False):
    '''formal pipeline for proccessing the rough keypoint data'''
    
    stage1 = start_corpus + "_s1_cleaned"
    stage2 = start_corpus + "_s2_interpolated"
    stage3 = start_corpus + "_s3_cropped"
    stage4 = start_corpus + "_s4_time_norm"
    stage5 = start_corpus + "_s5_space_norm"
    
    # Step 1: Clean the corpus using the data cleaner
    cleanCorpus(start_corpus, start_corpus + stage1, show_logs=show_logs)
    
    # Step 2: keypoint interpolation
    keypoint_interpolator = KeypointInterpolator()
    keypoint_interpolator.simpleCubicSplineCorpusGenerator(start_corpus + stage1, start_corpus + stage2)
    keypoint_interpolator.estimateHandsEndsCorpusGenerator(start_corpus + stage2, start_corpus + stage2)
    
    # Step 3: temporal cropping
    temporal_cropper = TemporalCropping()
    temporal_cropper.cropSingleHandedFramesInCorpus(start_corpus + stage2, start_corpus + stage3, show_logs=show_logs)
    temporal_cropper.cropHandsRestingTogetherInCorpus(start_corpus + stage3, start_corpus + stage3, show_logs=show_logs)
    temporal_cropper.cropToStrokePhaseInCorpus(start_corpus + stage3, start_corpus + stage3, show_logs=show_logs)

    # Step 4: temporal normalisation
    temporal_normaliser = TemporalNormalisor()
    temporal_normaliser.normaliseCorpus(start_corpus + stage3, start_corpus + stage4, frame_num=10, show_logs=show_logs)

    # level 3: spatial normalisation
    spatial_normaliser = SpatialNormalisor()
    spatial_normaliser.normaliseCorpus(start_corpus + stage4, start_corpus + stage5)
    
    
if __name__ == "__main__":
    start_corpus =r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\keypoints_V1"
    formalProcessingPipeline(start_corpus)
    
    start_corpus =r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\keypoints_V2"
    formalProcessingPipeline(start_corpus)
    