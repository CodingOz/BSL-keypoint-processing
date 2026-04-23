import uuid
from flask import render_template, redirect, request, jsonify
from app import app
from app.forms import make_video_upload_form
from app.r2 import upload_submission


LETTERS = [
    "A",
    "E",
    "I",
    "O",
    "U",
    "J",
    "N",
    "S",
    "T",
    "B",
    "P",
    "tea",
    "coffee",
    "and",
    "or"]
LETTERS_LAST_STAGE = ["U"]
VIDEO_TUTORIALS = {
    "A": "https://www.youtube-nocookie.com/embed/9yCOlhrrzsw",
    "E": "https://www.youtube-nocookie.com/embed/QseQeoaF2Mo",
    "I": "https://www.youtube-nocookie.com/embed/z8Itt63yfzk",
    "O": "https://www.youtube-nocookie.com/embed/1OlieaP6YV0",
    "U": "https://www.youtube-nocookie.com/embed/VDM_Aq-tb8s",
    "J": "https://www.youtube-nocookie.com/embed/vnw1VnjA0tA",
    "N": "https://www.youtube-nocookie.com/embed/iajFQ6DNsLE",
    "S": "https://www.youtube-nocookie.com/embed/D-oT_KzUR4w",
    "T": "https://www.youtube-nocookie.com/embed/KX0WranAw7k",
    "B": "https://www.youtube-nocookie.com/embed/kihkw6yWxJw",
    "P": "https://www.youtube-nocookie.com/embed/KlqyJ8DV3Hs",
    "tea": "https://www.youtube-nocookie.com/embed/6iNUX3kTsTg",
    "coffee": "https://www.youtube-nocookie.com/embed/NqsPtlg3lG4",
    "and": "https://www.youtube-nocookie.com/embed/gQvE_kfjxpA",
    "or": "https://www.youtube-nocookie.com/embed/HrpHw4SUq-o",
}
VIDEO_TUTORIALS_LAST_STAGE = {
    "U": "https://www.youtube-nocookie.com/embed/VDM_Aq-tb8s"}


@app.route('/', methods=['GET'])
def index():
    VideoUploadForm = make_video_upload_form(LETTERS)
    form = VideoUploadForm()
    return render_template(
        'main.html',
        title='BSL Survey',
        form=form,
        letters=LETTERS_LAST_STAGE,
        video_tutorials=VIDEO_TUTORIALS_LAST_STAGE,
    )


@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json(silent=True)
    if not data:
        return jsonify(error="No JSON body"), 400

    # Basic validation — expect at least one letter key
    if not any(k in data for k in LETTERS):
        return jsonify(error="No recognised sign keys in submission"), 400

    submission_id = str(uuid.uuid4())
    try:
        key = upload_submission(submission_id, data)
    except Exception as e:
        app.logger.error("R2 upload failed: %s", e)
        return jsonify(error="Storage error"), 500

    return jsonify(id=submission_id, key=key), 200


@app.route('/success')
def success():
    return render_template('success.html', title='Success')
