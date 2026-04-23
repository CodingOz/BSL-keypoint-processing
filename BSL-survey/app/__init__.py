from app import views
from flask_wtf.csrf import CSRFProtect
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
            "'unsafe-inline'",
            "'wasm-unsafe-eval'",
            "https://code.jquery.com",
            "https://cdnjs.cloudflare.com",
            "https://stackpath.bootstrapcdn.com",
            "https://cdn.jsdelivr.net",
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",
            "https://stackpath.bootstrapcdn.com",
        ],
        'media-src': [
            "'self'",
            "blob:",
        ],
        'worker-src': [
            "'self'",
            "blob:",
            "https://cdn.jsdelivr.net",
        ],
        'connect-src': [
            "'self'",
            "blob:",
            "https://cdn.jsdelivr.net",
            "https://storage.googleapis.com",
            "https://cdnjs.cloudflare.com",
            "https://stackpath.bootstrapcdn.com",
        ],
    }
)


@app.template_filter("getattr")
def jinja_getattr(obj, attr):
    return getattr(obj, attr)


csrf = CSRFProtect(app)
csrf.exempt(views.submit)
