
# holds all method of validating is a keypoint or keypoint sequence fits
# in a missing part of a hand
class InsertionValidator:
    def __init__(self, corpus_validator):
        self.corpus_validator = corpus_validator

    def validateInsertion(self, keypoint_sequence):
        # Implement logic to validate if the keypoint sequence can be inserted into a missing part of a hand
        # This could involve checking the spatial and temporal consistency of
        # the keypoints with the existing hand data
        pass
