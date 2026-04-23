import boto3
import os
import json

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("R2_REGION", "auto"),
)

BUCKET = os.environ["R2_BUCKET_NAME"]


def upload_submission(submission_id, data):
    """Save keypoint JSON to R2 as submissions/{id}.json"""
    key = f"submissions/{submission_id}.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json",
    )
    return key
