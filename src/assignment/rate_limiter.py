"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import contextvars
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        # Kết quả request gần nhất → pipeline điền "layer" mà không phải đoán từ text.
        # ContextVar: mỗi asyncio task (request chạy song song) thấy giá trị CỦA NÓ,
        # request khác không ghi đè được.
        self._last_blocked = contextvars.ContextVar(f"rl_last_blocked_{id(self)}", default=False)

    @property
    def last_blocked(self) -> bool:
        return self._last_blocked.get()

    @last_blocked.setter
    def last_blocked(self, value: bool):
        self._last_blocked.set(value)

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Cửa sổ trượt: bỏ mốc ra khỏi 60s gần nhất (mốc cũ luôn ở đầu deque → O(1))
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Đầy cửa sổ → chặn. KHÔNG ghi mốc cho request bị chặn, nếu không
        #    user spam liên tục sẽ bị khóa vĩnh viễn (cửa sổ không bao giờ vơi).
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self.last_blocked = True
            return self._block_response(
                f"Rate limit exceeded. Try again in {max(wait, 1):.0f}s."
            )

        # 3. Còn chỗ → ghi mốc, cho qua
        window.append(now)
        self.last_blocked = False
        return None
