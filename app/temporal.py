"""
Ardışık karelerde benzerlik skorunu medyan ile yumuşatır.

Karmaşık takip (Kalman, Re-ID) yok: yüz kutusu merkezine göre önceki karedeki iz ile
en yakın eşleşmeyi arar; eşleşmezse yeni iz başlatır.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque


@dataclass
class _Track:
    cx: float
    cy: float
    scores: Deque[float]


class TemporalScoreSmoother:
    """Aynı yüz izine düşen ham cosine benzerliklerini biriktirip medyan döndürür."""

    def __init__(self, window: int, match_dist_px: float) -> None:
        self._window = max(1, window)
        self._match_dist = float(match_dist_px)
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        """Örn. çözünürlük değişiminde veya akış kesilince çağrılabilir."""
        self._tracks = []

    def smooth_batch(
        self,
        items: list[tuple[float, float, float]],
    ) -> list[float]:
        """
        items: (cx, cy, raw_similarity) — işlenmiş görüntü koordinatlarında merkez ve ham skor.
        Dönüş: her öğe için medyan yumuşatılmış skor (aynı sıra).
        """
        if not items:
            self._tracks = []
            return []

        prev_tracks = self._tracks
        used_prev: set[int] = set()
        new_tracks: list[_Track] = []
        out: list[float] = []

        for cx, cy, raw_sim in items:
            best_j = -1
            best_d = math.inf
            for j, tr in enumerate(prev_tracks):
                if j in used_prev:
                    continue
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if d < best_d and d <= self._match_dist:
                    best_d = d
                    best_j = j

            if best_j >= 0:
                tr = prev_tracks[best_j]
                used_prev.add(best_j)
                tr.cx, tr.cy = cx, cy
                dq = deque(tr.scores, maxlen=self._window)
                dq.append(raw_sim)
                tr.scores = dq
                sm = float(median(tr.scores))
                new_tracks.append(tr)
                out.append(sm)
            else:
                dq_n: Deque[float] = deque([raw_sim], maxlen=self._window)
                tr_new = _Track(cx=cx, cy=cy, scores=dq_n)
                new_tracks.append(tr_new)
                out.append(float(raw_sim))

        self._tracks = new_tracks
        return out
