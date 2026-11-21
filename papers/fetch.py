"""Download canonical alignment/interpretability papers. Usage: python3 -I fetch.py <outdir>"""
import re, sys, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(sys.argv[1])
UA = {"User-Agent": "paper-fetcher/1.0"}

# (arxiv id, folder, keyword that must appear in the real title -- guards against a wrong id)
ARXIV = [
    # --- interpretability ---
    ("2209.10652", "interpretability", "superposition"),
    ("2309.08600", "interpretability", "sparse autoencoders"),
    ("2406.04093", "interpretability", "sparse autoencoders"),
    ("2408.05147", "interpretability", "gemma scope"),
    ("2211.00593", "interpretability", "interpretability in the wild"),
    ("2301.05217", "interpretability", "grokking"),
    ("2202.05262", "interpretability", "factual associations"),
    ("2304.14997", "interpretability", "circuit discovery"),
    ("2403.19647", "interpretability", "sparse feature circuits"),
    ("2406.11944", "interpretability", "transcoders"),
    ("2209.11895", "interpretability", "induction heads"),
    ("2210.13382", "interpretability", "world representations"),
    ("2311.03658", "interpretability", "linear representation"),
    ("2303.08112", "interpretability", "tuned lens"),
    ("2212.03827", "interpretability", "latent knowledge"),
    ("2310.01405", "interpretability", "representation engineering"),
    ("2308.10248", "interpretability", "activation"),
    ("2312.06681", "interpretability", "contrastive activation"),
    ("2406.11717", "interpretability", "refusal"),
    ("2308.03296", "interpretability", "influence functions"),
    ("2507.21509", "interpretability", "persona vectors"),
    ("2503.10965", "interpretability", "hidden objectives"),
    ("2404.14082", "interpretability", "mechanistic interpretability"),
    ("2501.16496", "interpretability", "open problems"),
    # --- alignment: foundations ---
    ("1606.06565", "alignment", "concrete problems"),
    ("2109.13916", "alignment", "unsolved problems"),
    ("1906.01820", "alignment", "learned optimization"),
    ("2105.14111", "alignment", "goal misgeneralization"),
    ("2210.01790", "alignment", "goal misgeneralization"),
    ("1912.01683", "alignment", "seek power"),
    ("1606.03137", "alignment", "cooperative inverse"),
    ("1611.08219", "alignment", "off-switch"),
    ("1706.03741", "alignment", "human preferences"),
    ("1805.00899", "alignment", "debate"),
    ("1811.07871", "alignment", "reward modeling"),
    ("2206.13353", "alignment", "power-seeking"),
    ("2306.12001", "alignment", "catastrophic"),
    ("2311.08379", "alignment", "scheming"),
    # --- alignment: empirical LLM work ---
    ("2212.08073", "alignment", "constitutional ai"),
    ("2212.09251", "alignment", "model-written evaluations"),
    ("2312.09390", "alignment", "weak-to-strong"),
    ("2312.06942", "alignment", "ai control"),
    ("2401.05566", "alignment", "sleeper agents"),
    ("2412.14093", "alignment", "alignment faking"),
    ("2412.04984", "alignment", "scheming"),
    ("2406.10162", "alignment", "sycophancy to subterfuge"),
    ("2310.13548", "alignment", "sycophancy"),
    ("2305.04388", "alignment", "don't always say"),
    ("2307.13702", "alignment", "faithfulness"),
    ("2507.11473", "alignment", "monitorability"),
    ("2309.00667", "alignment", "out of context"),
    ("2407.04694", "alignment", "situational awareness"),
    ("2502.17424", "alignment", "emergent misalignment"),
    ("2507.14805", "alignment", "subliminal learning"),
    ("2308.14752", "alignment", "deception"),
    ("2307.15043", "alignment", "adversarial attacks"),
    ("2501.18837", "alignment", "constitutional classifiers"),
]

# Canonical work that only exists as web pages.
HTML = [
    ("interpretability", "Zoom In - An Introduction to Circuits (Olah 2020)", "https://distill.pub/2020/circuits/zoom-in/"),
    ("interpretability", "A Mathematical Framework for Transformer Circuits (2021)", "https://transformer-circuits.pub/2021/framework/index.html"),
    ("interpretability", "Towards Monosemanticity (2023)", "https://transformer-circuits.pub/2023/monosemantic-features/index.html"),
    ("interpretability", "Scaling Monosemanticity (2024)", "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"),
    ("interpretability", "Circuit Tracing - Methods (2025)", "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"),
    ("interpretability", "On the Biology of a Large Language Model (2025)", "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"),
]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def safe(s):
    return re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|]', "", s)).strip()[:120]


ns = {"a": "http://www.w3.org/2005/Atom"}
ids = ",".join(i for i, _, _ in ARXIV)
feed = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={ids}&max_results={len(ARXIV)}"))
meta = {}
for e in feed.findall("a:entry", ns):
    aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/", 1)[1])
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    meta[aid] = (safe(e.find("a:title", ns).text), authors, e.find("a:published", ns).text[:4])

rows, problems = {"interpretability": [], "alignment": []}, []
for aid, folder, kw in ARXIV:
    if aid not in meta:
        problems.append(f"{aid}: no metadata"); continue
    title, authors, year = meta[aid]
    if kw not in title.lower():
        problems.append(f"{aid}: WRONG ID? expected '{kw}', got '{title}'"); continue
    dest = OUT / folder / f"{year} - {title}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            data = get(f"https://arxiv.org/pdf/{aid}")
            if not data.startswith(b"%PDF"):
                raise ValueError("not a pdf")
            dest.write_bytes(data)
        except Exception as ex:
            problems.append(f"{aid}: download failed ({ex})"); continue
        time.sleep(3)  # arXiv asks for ~1 request / 3s
    who = authors[0] + (" et al." if len(authors) > 1 else "")
    rows[folder].append((year, f"- {year} — [{title}](<{folder}/{dest.name}>) — {who} — [arXiv:{aid}](https://arxiv.org/abs/{aid})"))

for folder, name, url in HTML:
    dest = OUT / folder / f"{safe(name)}.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not dest.exists():
            dest.write_bytes(get(url))
        year = re.search(r"\((?:\D*)(\d{4})\)", name).group(1)
        rows[folder].append((year, f"- {year} — [{name}](<{folder}/{dest.name}>) — [source]({url})"))
    except Exception as ex:
        problems.append(f"{url}: {ex}")

readme = ["# Alignment & interpretability reading list", ""]
for folder in rows:
    readme += [f"## {folder.title()}", ""] + [r for _, r in sorted(rows[folder])] + [""]
(OUT / "README.md").write_text("\n".join(readme))

print(f"saved {sum(len(v) for v in rows.values())} / {len(ARXIV) + len(HTML)}")
print("\n".join(problems) or "no problems")
