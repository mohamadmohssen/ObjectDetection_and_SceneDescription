
import time
from collections import defaultdict

STATS_TOP_N         = 5    #top classes
STATS_RESET_SECONDS = 180  #automatic resets


class ObjectTracker:

    def __init__(
        self,
        top_n: int = STATS_TOP_N,
        reset_seconds: int = STATS_RESET_SECONDS,
    ):
        self.top_n         = top_n
        self.reset_seconds = reset_seconds
        self._counts: dict[str, int] = defaultdict(int)
        self._reset_time = time.time() + reset_seconds

    def update(self, labels: list[str]) -> None:
        self._maybe_auto_reset()
        for lbl in labels:
            self._counts[lbl] += 1

    def top_counts(self) -> list[tuple[str, int]]:
        sorted_items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[: self.top_n]

    def reset(self) -> None:
        self._counts = defaultdict(int)
        self._reset_time = time.time() + self.reset_seconds
        print("[TRACKER] Counts reset.")

    def seconds_until_reset(self) -> int:
        return max(0, int(self._reset_time - time.time()))

    def total_detections(self) -> int:
        return sum(self._counts.values())

    def _maybe_auto_reset(self) -> None:
        if time.time() >= self._reset_time:
            self.reset()