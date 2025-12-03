from typing import Dict, List, Tuple

import numpy as np

from .types import BBox


class CentroidTracker:
    """
    Very simple centroid-based tracker that assigns stable IDs
    based on nearest neighbor matching of face bounding boxes.
    """

    def __init__(self, max_disappeared: int = 25):
        self.next_id = 1
        self.objects: Dict[int, BBox] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def _centroid(self, box: BBox) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def update(self, rects: List[BBox]) -> Dict[int, BBox]:
        if len(rects) == 0:
            # Mark all existing objects as disappeared
            to_delete = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_delete.append(object_id)
            for oid in to_delete:
                self.objects.pop(oid, None)
                self.disappeared.pop(oid, None)
            return self.objects.copy()

        # If no objects yet, register all
        if len(self.objects) == 0:
            for rect in rects:
                self._register(rect)
            return self.objects.copy()

        # Compute distance matrix between existing and new centroids
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self._centroid(self.objects[oid]) for oid in object_ids])
        input_centroids = np.array([self._centroid(r) for r in rects])

        distances = np.linalg.norm(
            object_centroids[:, None, :] - input_centroids[None, :, :], axis=2
        )

        # For each object, find closest detection
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = rects[c]
            self.disappeared[oid] = 0
            assigned_rows.add(r)
            assigned_cols.add(c)

        # Unassigned detections -> new IDs
        for c, rect in enumerate(rects):
            if c not in assigned_cols:
                self._register(rect)

        # Unassigned existing IDs -> disappeared++
        for r, oid in enumerate(object_ids):
            if r not in assigned_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        return self.objects.copy()

    def _register(self, rect: BBox):
        self.objects[self.next_id] = rect
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id: int):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)





