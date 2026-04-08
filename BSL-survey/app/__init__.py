from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)
app.config.from_object('config')

Talisman(
    app,
    force_https=True,
    strict_transport_security=True,
    content_security_policy={
        'default-src': "'self'",
        'frame-src': "https://www.youtube-nocookie.com",
        'script-src': [
            "'self'",
            "https://code.jquery.com",
            "https://cdnjs.cloudflare.com",
            "https://stackpath.bootstrapcdn.com"
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",
            "https://stackpath.bootstrapcdn.com"
        ],
    }
)

@app.template_filter("getattr")
def jinja_getattr(obj, attr):
    return getattr(obj, attr)

from app import views