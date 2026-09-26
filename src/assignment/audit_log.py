"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


# Cắt text dài trong log — input 10.000 ký tự (edge case / flooding) không làm phình file
MAX_LOGGED_CHARS = 500


def _scrub(text: str) -> str:
    """Che PII/secret trước khi ghi log — log bị commit/chia sẻ không được thành nơi lộ secret."""
    from guardrails.output_guardrails import content_filter

    result = content_filter(text or "")
    clean = result["redacted"]
    if "obfuscated_secret" in result.get("issue_types", []):
        # Secret bị xé lẻ không che đúng vị trí được → bỏ cả nội dung
        clean = "[REDACTED: obfuscated secret]"
    if len(clean) > MAX_LOGGED_CHARS:
        clean = clean[:MAX_LOGGED_CHARS] + f"… [+{len(clean) - MAX_LOGGED_CHARS} chars]"
    return clean


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline).

    Dùng như "side observer" bọc ngoài runner (không phải ADK plugin): request bị
    chặn ở input thì after_model_callback không bao giờ chạy, nên plugin-hook sẽ bỏ
    sót đúng những request cần điều tra nhất.
    """

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id. Returns the request_id."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": _scrub(text),
            "started_at": utc_now_iso(),
            "_t0": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        if request_id is None:
            # Không có id → ghép với request mở gần nhất của user này
            request_id = next(
                (rid for rid, e in reversed(self._open.items()) if e["user_id"] == user_id),
                None,
            )
        entry = self._open.pop(request_id, None) or {
            "request_id": request_id,
            "user_id": user_id,
            "input": None,
            "started_at": None,
            "_t0": None,
        }
        t0 = entry.pop("_t0")
        entry.update(
            {
                "output": _scrub(text),
                "blocked": bool(blocked),
                "layer": layer,
                "error": error,
                "finished_at": utc_now_iso(),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1) if t0 else None,
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
