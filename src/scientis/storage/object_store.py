"""S3-compatible object storage abstraction.

Backed by MinIO for local dev, any S3-compatible service in production.
"""

import logging

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from scientis.config import Settings

logger = logging.getLogger(__name__)


class ObjectStore:
    """S3-compatible blob store for PDFs, figures, and artifacts."""

    def __init__(self, settings: Settings):
        self.bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=BotoConfig(signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)
            logger.info("Created bucket: %s", self.bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def get(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def list(self, prefix: str = "", max_keys: int = 1000) -> list[str]:
        response = self._client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys
        )
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
