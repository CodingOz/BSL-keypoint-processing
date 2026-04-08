from flask import render_template, redirect
from app import app
from app.forms import make_video_upload_form

LETTERS = ["A","E","I","O","U","J","N","S","T","B","P","tea","coffee","and","or"]

VIDEO_TUTORIALS = {
    "A":      "https://www.youtube-nocookie.com/embed/9yCOlhrrzsw",
    "E":      "https://www.youtube-nocookie.com/embed/QseQeoaF2Mo",
    "I":      "https://www.youtube-nocookie.com/embed/z8Itt63yfzk",
    "O":      "https://www.youtube-nocookie.com/embed/1OlieaP6YV0",
    "U":      "https://www.youtube-nocookie.com/embed/VDM_Aq-tb8s",
    "J":      "https://www.youtube-nocookie.com/embed/vnw1VnjA0tA",
    "N":      "https://www.youtube-nocookie.com/embed/iajFQ6DNsLE",
    "S":      "https://www.youtube-nocookie.com/embed/D-oT_KzUR4w",
    "T":      "https://www.youtube-nocookie.com/embed/KX0WranAw7k",
    "B":      "https://www.youtube-nocookie.com/embed/kihkw6yWxJw",
    "P":      "https://www.youtube-nocookie.com/embed/KlqyJ8DV3Hs",
    "tea":    "https://www.youtube-nocookie.com/embed/6iNUX3kTsTg",
    "coffee": "https://www.youtube-nocookie.com/embed/NqsPtlg3lG4",
    "and":    "https://www.youtube-nocookie.com/embed/gQvE_kfjxpA",
    "or":     "https://www.youtube-nocookie.com/embed/HrpHw4SUq-o",
}

@app.route('/', methods=['GET'])
def All():
    # Flask only serves the page now.
    # All video processing and submission happens client-side via JS.
    VideoUploadForm = make_video_upload_form(LETTERS)
    form = VideoUploadForm()
    return render_template(
        'main.html',
        title='BSL Survey',
        form=form,
        letters=LETTERS,
        video_tutorials=VIDEO_TUTORIALS,
    )

@app.route('/success')
def success():
    return render_template('success.html', title='Success')