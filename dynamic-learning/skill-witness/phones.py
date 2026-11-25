"""Collect NATIONAL formatting from annual pinned python-phonenumbers releases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

CACHE = Path.home() / ".cache/alignment-research/skill-witness/wheels"
RELEASES = {
    "7.5.2": "41d50c2211dcd8817a5d49f079840cf35f2dda611ba8331f7f0980d3a2c84b62",
    "8.0.1": "02127e982da7ceb056d5f153b2ec4a97da57328c5522dc206d23bc95b5d26d4a",
    "8.8.9": "2fc6a0e1605a69fecd8009ffc3ba0d5be4033b6ebd7fb409fc413563fcfcd487",
    "8.10.3": "e038bde747aa967d890c9a59528b00a24fba0ad883404901c8318c137510d732",
    "8.11.2": "796ba25c3064727ca0b8edf7a8ef5ef247c6da37aee498562e6e0ed46970a57f",
    "8.12.16": "56ad29025b8f885945506350b06d77afbc506c5463141d77a5df767280a7ee0b",
    "8.12.41": "2b8c7a7ffac4fe2be3d8bf20dad316ea1292f27422c9e18b1f3cd16734d4a5ed",
    "8.13.4": "a577a46c069ad889c7b7cf4dd978751d059edeab28b97acead4775d2ea1fc70a",
    "8.13.28": "ad7bc7d7fd6599a124423ffb840409630777c72d0ee58ba8070cc8e7efcb4c38",
    "8.13.53": "fa7b93e12b3ac9baa7b056061cfcbe4de406039f720bb539dc240bed9cc43da2",
    "9.0.22": "645e66cd9a136b3b257b5f941fa97d324124114d31ad3c9f2488682f47ad7ee1",
}
REGIONS = "AR AT AU BR CA CH CN DE EG ES FI FR GB GR ID IL IN IT JP KR MX NG NL NZ PL RU SE TR US ZA".split()


def wheel(version):
    path = CACHE / f"phonenumbers-{version}-py2.py3-none-any.whl"
    digest = RELEASES[version]
    if not path.exists():
        with urlopen(f"https://pypi.org/pypi/phonenumbers/{version}/json", timeout=30) as response:
            files = json.load(response)["urls"]
        artifact = next(f for f in files if f["filename"] == path.name and f["digests"]["sha256"] == digest)
        with urlopen(artifact["url"], timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f"Download hash mismatch for {version}")
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as temp:
            temp.write(data)
        Path(temp.name).replace(path)
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Cache hash mismatch for {version}")
    return path


def probes(lib, region):
    types = lib.PhoneNumberType
    anchors = []
    for kind in (types.FIXED_LINE, types.MOBILE, types.TOLL_FREE, types.PREMIUM_RATE,
                 types.VOIP, types.SHARED_COST, types.PERSONAL_NUMBER, types.UAN):
        example = lib.example_number_for_type(region, kind)
        if example is not None:
            n = lib.national_significant_number(example)
            if n not in anchors:
                anchors.append(n)
    if not anchors:
        raise ValueError(f"No example metadata for {region}")
    values = list(anchors)
    # Keep prefixes anchored in the oldest metadata; vary suffix and length independently.
    for step in range(1, 100):
        for i, n in enumerate(anchors):
            suffix = sha256(f"fixture-{region}-{i}-{step}".encode()).digest()
            length = max(3, len(n) + (0, -1, 1, 0)[(step-1) % 4])
            keep = min(len(n)-2, length-2, 4)
            candidate = n[:keep] + "".join(str(x % 10) for x in suffix[:length-keep])
            if candidate not in values:
                values.append(candidate)
            if len(values) == 40:
                return {f"{region}-{j:02d}": value for j, value in enumerate(values)}
    raise AssertionError(f"Insufficient distinct probes for {region}")


def worker(mode, path):
    sys.path.insert(0, str(path))
    import phonenumbers as lib
    assert str(path) in lib.__file__
    if mode == "probes":
        return {r: [lib.country_code_for_region(r), probes(lib, r)] for r in REGIONS}
    result = {}
    for r, (cc, numbers) in json.load(sys.stdin).items():
        if r not in lib.SUPPORTED_REGIONS:
            result[r] = None
            continue
        formatted = {}
        for name, n in numbers.items():
            number = lib.PhoneNumber(country_code=cc, national_number=int(n))
            zeros = len(n) - len(n.lstrip("0"))
            if zeros:
                number.italian_leading_zero = True
                number.number_of_leading_zeros = zeros
            assert lib.national_significant_number(number) == n
            formatted[name] = lib.format_number(number, lib.PhoneNumberFormat.NATIONAL)
        result[r] = formatted
    return result


def invoke(mode, path, payload=None):
    completed = subprocess.run([sys.executable, "-I", __file__, "--worker", mode, str(path)],
                               input=json.dumps(payload), text=True, capture_output=True, timeout=90)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    return json.loads(completed.stdout)


def collect(output):
    with ThreadPoolExecutor(max_workers=4) as pool:
        wheels = list(pool.map(wheel, RELEASES))
        grid = invoke("probes", wheels[0])
        snapshots = list(pool.map(lambda p: invoke("format", p, grid), wheels))
    inputs = {name: {"n": n} for _, numbers in grid.values() for name, n in numbers.items()}
    assert len(inputs) == 40 * len(REGIONS)
    producers = {r: [snap[r] for snap in snapshots] for r in REGIONS}
    for r, series in producers.items():
        assert len(series) == len(RELEASES)
        assert all(row is None or (set(row) == set(grid[r][1]) and all(
            isinstance(s, str) and len(s) > 0 for s in row.values())) for row in series)
    data = {"project": "python-phonenumbers", "releases": list(RELEASES), "producers": producers,
            "inputs": inputs, "sources": {"pypi": "https://pypi.org/project/phonenumbers/",
            "upstream": "https://github.com/daviddrysdale/python-phonenumbers",
            "wheel_sha256": RELEASES, "calling_codes": {r: x[0] for r,x in grid.items()}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        temp.write((json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    Path(temp.name).replace(output)
    print(f"{len(REGIONS)} region streams x {len(RELEASES)} versions; {len(inputs)} input IDs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["probes", "format"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.path)))
    else:
        collect(args.path)
