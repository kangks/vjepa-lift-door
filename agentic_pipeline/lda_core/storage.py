from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class Storage(Protocol):
    def get_bytes(self, key: str) -> bytes: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def get_npy(self, key: str) -> np.ndarray: ...
    def put_npy(self, key: str, arr: np.ndarray) -> None: ...
    def exists(self, key: str) -> bool: ...
    def sha256(self, key: str) -> str: ...
    def put_bytes_if_absent(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> bool: ...


class FileStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"storage key escapes root: {key}")
        return path

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_bytes_if_absent(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return True

    def get_npy(self, key: str) -> np.ndarray:
        import numpy as np
        return np.load(io.BytesIO(self.get_bytes(key)), allow_pickle=False)

    def put_npy(self, key: str, arr: np.ndarray) -> None:
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr), allow_pickle=False)
        self.put_bytes(key, buf.getvalue(), "application/octet-stream")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def sha256(self, key: str) -> str:
        return hashlib.sha256(self.get_bytes(key)).hexdigest()


class S3Storage:
    def __init__(self, bucket: str, client=None):
        if not bucket:
            raise ValueError("S3 bucket name is required")
        if client is None:
            import boto3
            client = boto3.client("s3")
        self.bucket = bucket
        self.client = client

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def put_bytes_if_absent(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type, IfNoneMatch="*"
            )
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"PreconditionFailed", "ConditionalRequestConflict"} or status == 412:
                return False
            raise

    def get_npy(self, key: str) -> np.ndarray:
        import numpy as np
        return np.load(io.BytesIO(self.get_bytes(key)), allow_pickle=False)

    def put_npy(self, key: str, arr: np.ndarray) -> None:
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr), allow_pickle=False)
        self.put_bytes(key, buf.getvalue(), "application/octet-stream")

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise

    def sha256(self, key: str) -> str:
        return hashlib.sha256(self.get_bytes(key)).hexdigest()


def from_env() -> Storage:
    location = os.environ.get("LDA_STORAGE", "")
    if location.startswith("file:"):
        return FileStorage(location[len("file:"):])
    bucket = os.environ.get("LDA_BUCKET", "")
    if location.startswith("s3://"):
        bucket = location[len("s3://"):].split("/", 1)[0]
    return S3Storage(bucket)
