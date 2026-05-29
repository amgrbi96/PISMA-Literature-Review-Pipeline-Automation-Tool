"""Staged checkpoint persistence for crash recovery."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.paper import PaperMetadata

logger = logging.getLogger(__name__)


def write_checkpoint(
        papers: list[PaperMetadata],
        stage: str,
        results_dir: str | Path,
        *,
        extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a checkpoint file for one pipeline stage."""

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"checkpoint_{stage}.json"

    records = []
    for paper in papers:
        record: dict[str, Any] = {
            "id": paper.identity_key,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract[:500] if paper.abstract else "",
            "venue": paper.venue,
            "source_db": paper.source,
            "source_id": paper.external_ids.get("pubmed") or paper.external_ids.get("s2") or "",
            "inclusion_decision": paper.inclusion_decision,
        }
        records.append(record)

    payload = {
        "stage": stage,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
        **(extra or {}),
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Checkpoint '%s' written with %s records to %s.", stage, len(records), path)
    return path


def latest_checkpoint_stage(results_dir: str | Path) -> str | None:
    """Return the stage name of the latest checkpoint, or None if no checkpoints exist."""

    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    candidates: list[tuple[str, float]] = []
    for path in results_path.glob("checkpoint_*.json"):
        stage = path.stem.replace("checkpoint_", "")
        try:
            candidates.append((stage, path.stat().st_mtime))
        except OSError:
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def load_checkpoint(results_dir: str | Path, stage: str) -> dict[str, Any] | None:
    """Load a checkpoint file for a given stage."""

    path = Path(results_dir) / f"checkpoint_{stage}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read checkpoint file %s.", path)
        return None
