import json
from keypoint_interpolator import KeypointInterpolator
from temporal_cropping import TemporalCropping
from temporal_normalisation import TemporalNormalisor
from spatial_normalisation import SpatialNormalisor


def post_error_detection_pipeline(corpus_container_path, show_logs=False):
    ''' asumes the corpus container hold "Corpus_level_0"
    for each latey of processing, the output is stored in a new folder with the name "Corpus_level_X" where X is the level of processing.
    The output of each level is used as input for the next level.
    '''
    level0 = corpus_container_path + r"/Corpus_level_0"
    level1a = corpus_container_path + r"/Corpus_level_1a"
    level1b = corpus_container_path + r"/Corpus_level_1b"
    level2a = corpus_container_path + r"/Corpus_level_2a"
    level2b = corpus_container_path + r"/Corpus_level_2b"
    level2c = corpus_container_path + r"/Corpus_level_2c"
    level3 = corpus_container_path + r"/Corpus_level_3"
    level4 = corpus_container_path + r"/Corpus_level_4"

    # first level of processing: keypoint interpolation
    keypoint_interpolator = KeypointInterpolator()

    keypoint_interpolator.SimpleCubicSplineCorpusGenerator(level0, level1a)
    keypoint_interpolator.estimateHandsEndsCorpusGenerator(
        level1a, level1b, show_logs=show_logs)

    print("level 1 processing done\n\n")

    # level 2: temporal cropping
    temporal_cropper = TemporalCropping()
    temporal_cropper.crop_single_handed_frames_in_corpus(
        level1b, level2a, show_logs=show_logs)
    temporal_cropper.crop_hands_resting_together_in_corpus(
        level2a, level2b, show_logs=show_logs)
    temporal_cropper.crop_to_stroke_phase_in_corpus(
        level2b, level2c, show_logs=show_logs)

    # level 3: temporal normalisation
    temporal_normaliser = TemporalNormalisor()
    temporal_normaliser.NormaliseCorpus(
        level2c, level3, frame_num=20, show_logs=show_logs)

    # level 4: spatial normalisation
    spatial_normaliser = SpatialNormalisor()
    spatial_normaliser.normaliseCorpus(level3, level4)


if __name__ == "__main__":
    post_error_detection_pipeline(
        r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Testing_post_error_ditection_pipeline")
