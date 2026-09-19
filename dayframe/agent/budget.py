from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

BUDGET_EXHAUSTED = (
    "Budget exhausted. Do not call view_photos. Finish remaining clusters with "
    "write_memory or discard_cluster using the digest and any photos already seen."
)


def recursion_limit(max_turns: int) -> int:
    """LangGraph counts super-steps, not turns. One turn is agent + tools."""
    return 2 * max_turns + 6


@dataclass
class Budget:
    max_images: int
    max_turns: int
    max_cost_usd: float
    timeout_seconds: int
    images_sent: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def consume_images(self, requested: int) -> int:
        take = max(0, requested)
        with self._lock:
            remaining = max(0, self.max_images - self.images_sent)
            granted = min(take, remaining)
            self.images_sent += granted
            return granted

    def release_images(self, count: int) -> None:
        with self._lock:
            self.images_sent = max(0, self.images_sent - max(0, count))

    @property
    def images_remaining(self) -> int:
        with self._lock:
            return max(0, self.max_images - self.images_sent)
