"""Paper ingestion service."""

import uuid
from datetime import datetime

from scientis.storage.object_store import ObjectStore


def generate_paper_id() -> str:
    date_part = datetime.utcnow().strftime("%Y-%m%d")
    short_uuid = uuid.uuid4().hex[:8]
    return f"p-{date_part}-{short_uuid}"


def ingest_paper(store: ObjectStore, content: bytes, filename: str) -> str:
    """Store raw PDF in object storage and return a paper_id."""
    paper_id = generate_paper_id()
    key = f"papers/{paper_id}/raw/{filename}"
    store.put(key, content, content_type="application/pdf")
    return paper_id
