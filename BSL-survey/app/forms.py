from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import FileField, SubmitField, BooleanField
from flask_wtf.file import FileAllowed

def make_video_upload_form(letters):
    class VideoUploadForm(FlaskForm):
        consent_video_deletion = BooleanField(
            'I understand that my videos will be used only for key-point extraction and that the original video will be permanently deleted afterwards.',
            validators=[DataRequired(message="You must agree to this consent statement.")]
        )
        
        consent_data_usage = BooleanField(
            'I consent to the key-point tracking data extracted from my video being used to train and evaluate machine-learning models.',
            validators=[DataRequired(message="You must agree to this consent statement.")]
        )
        submit = SubmitField("Submit")

    for letter in letters:
        setattr(
            VideoUploadForm,
            f"video_{letter}",
            FileField(
                f"Upload video for {letter} (Click to upload)",
                validators=[FileAllowed(["mp4", "mov", "avi"])]
            )
        )

    return VideoUploadForm
