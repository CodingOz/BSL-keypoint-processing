import os

basedir = os.path.abspath(os.path.dirname(__file__))

WTF_CSRF_ENABLED = True
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set")

MAX_CONTENT_LENGTH = 500 * 1024 * 1024