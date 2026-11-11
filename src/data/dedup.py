"""Exact and near-duplicate detection for image datasets.

Exact duplicates are detected with an MD5 of the file bytes. Near duplicates
use a 64-bit difference hash (dHash, 9x8 grayscale): two images whose hashes
differ in only a few bits are almost always re-shots of the same scene.
"""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def probe_image(path: str | Path) -> tuple[int, int, str, str]:
    """Return (width, height, md5 hex, dHash as 16 hex chars) for one image."""
    raw = Path(path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        width, height = im.size
        gray = im.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = int("".join("1" if b else "0" for b in bits), 2)
    return width, height, md5, format(value, "016x")


def probe_images(paths: Iterable[str | Path], workers: int = 2, progress=None) -> pd.DataFrame:
    """Probe many images in parallel. Columns: w, h, md5, dhash.

    `progress` can wrap the result iterator, e.g. functools.partial(tqdm, total=n).
    Keep `workers` small on machines with little free RAM.
    """
    paths = list(paths)
    with ThreadPoolExecutor(workers) as executor:
        results = executor.map(probe_image, paths)
        if progress is not None:
            results = progress(results)
        rows = list(results)
    return pd.DataFrame(rows, columns=["w", "h", "md5", "dhash"])


def hashes_to_u64(hex_hashes: Iterable[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def _hamming(x: np.uint64, others: np.ndarray) -> np.ndarray:
    xor = x ^ others
    return _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def hamming_pairs(hashes: np.ndarray, max_dist: int) -> pd.DataFrame:
    """All index pairs (i < j) whose hashes differ in <= max_dist bits."""
    rows = []
    for i in range(len(hashes) - 1):
        dist = _hamming(hashes[i], hashes[i + 1:])
        for k in np.nonzero(dist <= max_dist)[0]:
            rows.append((i, i + 1 + int(k), int(dist[k])))
    return pd.DataFrame(rows, columns=["i", "j", "dist"])


def hamming_pairs_between(hashes_a: np.ndarray, hashes_b: np.ndarray, max_dist: int) -> pd.DataFrame:
    """All pairs (i in a, j in b) whose hashes differ in <= max_dist bits."""
    rows = []
    for i in range(len(hashes_a)):
        dist = _hamming(hashes_a[i], hashes_b)
        for j in np.nonzero(dist <= max_dist)[0]:
            rows.append((i, int(j), int(dist[j])))
    return pd.DataFrame(rows, columns=["i", "j", "dist"])


def connected_groups(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Label each of n items with its connected component under the given pairs."""
    if len(pairs) == 0:
        return np.arange(n)
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs["i"].to_numpy(), pairs["j"].to_numpy())), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels
