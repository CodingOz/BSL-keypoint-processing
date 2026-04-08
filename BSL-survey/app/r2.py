import boto3
import os
import filetype

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

BUCKET = os.environ["R2_BUCKET_NAME"]

ALLOWED_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}

def validate_video(file) -> bool:
    header = file.read(2048)
    file.seek(0)
    kind = filetype.guess(header)
    if kind is None:
        return False
    return kind.mime in ALLOWED_MIME_TYPES

def upload_video(file, key):
    if not validate_video(file):
        raise ValueError(f"Invalid file type for key {key}")

    s3.upload_fileobj(
        file,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": "video/mp4"
        }
    )