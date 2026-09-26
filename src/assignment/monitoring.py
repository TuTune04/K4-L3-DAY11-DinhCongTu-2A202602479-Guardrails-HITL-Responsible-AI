"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    # Bổ sung: lỗi API (quota/timeout) — không phải "bị chặn" nhưng cần được thấy
    errors: int = 0
    error_threshold: int = 1
    blocked_by_layer: dict[str, int] = field(default_factory=dict)

    def record_request(self, *, blocked: bool, layer: str | None, error: str | None = None):
        """Cập nhật bộ đếm sau mỗi request (pipeline gọi, không phụ thuộc framework)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if error:
            self.errors += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        snap = self.snapshot()
        # Tính lại từ đầu mỗi lần gọi → gọi nhiều lần không sinh alert trùng lặp
        self.alerts = []
        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            self.alerts.append(Alert(
                "block_rate", snap["block_rate"], self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} > {self.block_rate_threshold:.0%}: "
                "đang bị tấn công dồn dập HOẶC filter chặn nhầm hàng loạt — cần người xem audit log.",
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                "rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
                f"{self.rate_limit_hits} lần chạm rate limit: có dấu hiệu flooding / cost attack.",
            ))
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            self.alerts.append(Alert(
                "judge_fail_rate", snap["judge_fail_rate"], self.judge_fail_rate_threshold,
                f"Judge fail rate {snap['judge_fail_rate']:.0%}: model trả lời không an toàn nhiều.",
            ))
        if self.errors >= self.error_threshold:
            self.alerts.append(Alert(
                "errors", self.errors, self.error_threshold,
                f"{self.errors} request lỗi (API/quota/timeout) — kết quả suite có thể thiếu.",
            ))
        return self.alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        data["errors"] = self.errors
        data["blocked_by_layer"] = dict(self.blocked_by_layer)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
