"""Audit history (FR-D-06).

Every finished Audit Report is written to a store so a model can be compared
against its own past.  The store is a directory of JSON files plus an index,
which is enough for one workstation or one CI job and needs no service to run.

ponytail: flat files with a rewritten index, single writer assumed. Move to
SQLite if two audits ever need to write the same store at the same time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_STORE = os.environ.get("PLUMBLINE_HISTORY", os.path.join(os.getcwd(), "plumbline_audits"))
INDEX_NAME = "index.json"


@dataclass
class HistoryEntry:
    audit_id: str
    model_name: str
    finished_at: str
    badge: str
    audit_score: float
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "model_name": self.model_name,
            "finished_at": self.finished_at,
            "badge": self.badge,
            "audit_score": self.audit_score,
            "path": self.path,
        }


class AuditHistory:
    """A directory of stored Audit Reports."""

    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or DEFAULT_STORE)
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, INDEX_NAME)

    # -- writing ------------------------------------------------------------

    def save(self, report: "Any") -> HistoryEntry:
        payload = report.to_dict()
        filename = f"audit-{payload['audit_id']}.json"
        path = os.path.join(self.root, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        entry = HistoryEntry(
            audit_id=payload["audit_id"],
            model_name=payload["model"].get("name", "unknown"),
            finished_at=payload["finished_at"],
            badge=payload["score"]["badge"],
            audit_score=payload["score"]["audit_score"],
            path=path,
        )
        entries = [e for e in self.entries() if e.audit_id != entry.audit_id]
        entries.append(entry)
        with open(self.index_path, "w", encoding="utf-8") as handle:
            json.dump([e.to_dict() for e in entries], handle, indent=2)
        return entry

    # -- reading ------------------------------------------------------------

    def entries(self) -> list[HistoryEntry]:
        if not os.path.isfile(self.index_path):
            return []
        try:
            with open(self.index_path, encoding="utf-8") as handle:
                rows = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        return [HistoryEntry(**row) for row in rows if isinstance(row, dict)]

    def for_model(self, model_name: str) -> list[HistoryEntry]:
        """Every stored audit of one model, oldest first (FR-D-06)."""
        matches = [e for e in self.entries() if e.model_name == model_name]
        return sorted(matches, key=lambda e: e.finished_at)

    def load(self, audit_id: str) -> dict[str, Any]:
        for entry in self.entries():
            if entry.audit_id == audit_id:
                with open(entry.path, encoding="utf-8") as handle:
                    return json.load(handle)
        raise KeyError(f"no stored audit with id {audit_id!r} under {self.root}")
