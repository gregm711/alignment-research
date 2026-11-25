"""Fetch and temporally subsample the two public recordings outside the repository."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

CACHE = Path.home() / ".cache/alignment-research/dynamic-gas"
URL = "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip"
SHA = "8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7"
PANELS = ("ethylene_CO", "ethylene_methane")


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "source.zip"
    if not path.exists():
        temp = path.with_suffix(".download")
        urlretrieve(URL, temp)
        temp.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA:
        raise ValueError("UCI source checksum mismatch")
    with ZipFile(path) as archive:
        for name in PANELS:
            rows, last = [], -1
            with archive.open(name + ".txt") as source:
                assert b"Ethylene conc (ppm), sensor readings (16 channels)" in next(source)
                for record in source:
                    now = int(float(record.split(None, 1)[0]) // 5)
                    if now < last:
                        raise ValueError("Clock moved backward")
                    if now != last:
                        rows.append([float(value) for value in record.split()])
                        last = now
            data = np.array(rows)
            assert data.shape[1] == 19 and np.isfinite(data).all()
            np.save(CACHE / (name + ".npy"), data, allow_pickle=False)


if __name__ == "__main__":
    prepare()
