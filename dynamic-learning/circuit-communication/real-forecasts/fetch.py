"""Cache pinned CDC issue summaries; verify every file against the pinned Git tree."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
import json
from urllib.request import Request, urlopen

SHA = "b5dd6a9bc9cf7b565c6c4fe992bd35a82fa2c2f3"
REPO = "cdcepi/FluSight-forecast-hub"
CACHE = Path.home() / ".cache/alignment-research/flusight-pooling" / SHA


def dates_in_panel(date):
    return ("2024-11-22" <= date <= "2025-05-31" or
            "2025-11-22" <= date <= "2026-05-31")


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    tree_path = CACHE / "tree.json"
    if not tree_path.exists():
        url = f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1"
        tree_path.write_bytes(urlopen(Request(url, headers={"User-Agent": "research-replay"}), timeout=25).read())
    tree = json.loads(tree_path.read_bytes())
    assert tree["sha"] == SHA and not tree["truncated"]
    paths = [r for r in tree["tree"] if r["path"] == "target-data/target-hospital-admissions.csv" or
             (r["path"].startswith("weekly-summaries/") and dates_in_panel(r["path"].split("/")[1]) and
              r["path"].endswith(("_flu_forecasts_data.csv", "_flu_target_hospital_admissions_data.csv")))]

    def download(row):
        path = CACHE / Path(row["path"]).name
        data = path.read_bytes() if path.exists() else urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{SHA}/{row['path']}", timeout=45).read()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == row["sha"]
        if not path.exists():
            path.write_bytes(data)
    with ThreadPoolExecutor(6) as executor:
        list(executor.map(download, paths))
    print(len(paths), "verified sources", sum(p["size"] for p in paths), "bytes")


if __name__ == "__main__":
    prepare()
