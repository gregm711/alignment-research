"""Second pass: lab research not in the first list. Usage: python3 -I fetch_labs.py <outdir>
arXiv items are found by title search (no hand-typed ids), and only saved if the returned title matches."""
import re, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(sys.argv[1])
UA = {"User-Agent": "Mozilla/5.0 (paper-fetcher)"}
NS = {"a": "http://www.w3.org/2005/Atom"}

TITLES = {
    "anthropic": [
        "Reasoning Models Don't Always Say What They Think",
        "Agentic Misalignment: How LLMs Could Be Insider Threats",
        "Sabotage Evaluations for Frontier Models",
        "SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents",
        "Specific versus General Principles for Constitutional AI",
        "Question Decomposition Improves the Faithfulness of Model-Generated Reasoning",
        "Debating with More Persuasive LLMs Leads to More Truthful Answers",
        "Measuring Progress on Scalable Oversight for Large Language Models",
        "Best-of-N Jailbreaking",
        "Natural Emergent Misalignment from Reward Hacking in Production RL",
        "Stress-Testing Model Specs Reveals Character Differences among Language Models",
        "Emergent Introspective Awareness in Large Language Models",
        "Softmax Linear Units",
        "Alignment Faking Mitigations",
    ],
    "openai": [
        "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation",
        "Stress Testing Deliberative Alignment for Anti-Scheming Training",
        "Persona Features Control Emergent Misalignment",
        "Deliberative Alignment: Reasoning Enables Safer Language Models",
        "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions",
        "Why Language Models Hallucinate",
        "Training LLMs for Honesty via Confessions",
        "Weight-sparse transformers have interpretable circuits",
        "Prover-Verifier Games improve legibility of LLM outputs",
        "Supervising strong learners by amplifying weak experts",
        "Monitoring Monitorability",
    ],
    "google-deepmind": [
        "An Approach to Technical AGI Safety and Security",
        "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
        "Improving Dictionary Learning with Gated Sparse Autoencoders",
        "AtP*: An efficient and scalable method for localizing LLM behaviour to components",
        "Tracr: Compiled Transformers as a Laboratory for Interpretability",
        "Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla",
        "Evaluating Frontier Models for Dangerous Capabilities",
        "Evaluating Frontier Models for Stealth and Situational Awareness",
        "When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors",
        "Model evaluation for extreme risks",
        "On scalable oversight with weak LLMs judging strong LLMs",
        "MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking",
        "AI Safety Gridworlds",
        "Reward Tampering Problems and Solutions in Reinforcement Learning",
        "Alignment of Language Agents",
        "Discovering Agents",
    ],
    "other-labs": [  # Redwood, Apollo, METR, CAIS, academic groups, AISI
        "Measuring AI Ability to Complete Long Tasks",
        "Towards evaluations-based safety cases for AI scheming",
        "A sketch of an AI control safety case",
        "Ctrl-Z: Controlling AI Agents via Resampling",
        "Large Language Models can Strategically Deceive their Users when Put Under Pressure",
        "Detecting Strategic Deception Using Linear Probes",
        "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition",
        "Stochastic Parameter Decomposition",
        "AI Sandbagging: Language Models can Strategically Underperform on Evaluations",
        "Tell me about yourself: LLMs are aware of their learned behaviors",
        "Looking Inward: Language Models Can Learn About Themselves by Introspection",
        "Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data",
        "The Reversal Curse: LLMs trained on A is B fail to learn B is A",
        "Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?",
        "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs",
        "International AI Safety Report",
        "Jailbroken: How Does LLM Safety Training Fail?",
        "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples",
        "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing",
        "Causal Abstractions of Neural Networks",
    ],
}

# Work that lives only on the labs' own sites.
URLS = {
    "anthropic": [
        ("Core Views on AI Safety (2023)", "https://www.anthropic.com/news/core-views-on-ai-safety"),
        ("Claude's Constitution", "https://www.anthropic.com/constitution"),
        ("Responsible Scaling Policy", "https://www.anthropic.com/responsible-scaling-policy"),
        ("Many-shot Jailbreaking (2024)", "https://www.anthropic.com/research/many-shot-jailbreaking"),
        ("Simple Probes Can Catch Sleeper Agents (2024)", "https://www.anthropic.com/research/probes-catch-sleeper-agents"),
        ("Agentic Misalignment (2025)", "https://www.anthropic.com/research/agentic-misalignment"),
        ("Emergent Introspective Awareness (2025)", "https://transformer-circuits.pub/2025/introspection/index.html"),
        ("Sparse Crosscoders (2024)", "https://transformer-circuits.pub/2024/crosscoders/index.html"),
        ("Softmax Linear Units (2022)", "https://transformer-circuits.pub/2022/solu/index.html"),
        ("Alignment Science Blog - index", "https://alignment.anthropic.com/"),
        ("Transformer Circuits Thread - index", "https://transformer-circuits.pub/"),
    ],
    "openai": [
        ("Language Models Can Explain Neurons in Language Models (2023)", "https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html"),
        ("Model Spec", "https://model-spec.openai.com/"),
        ("Preparedness Framework v2 (2025).pdf", "https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf"),
        ("Detecting Misbehavior in Frontier Reasoning Models (2025)", "https://openai.com/index/chain-of-thought-monitoring/"),
        ("Multimodal Neurons in Artificial Neural Networks (2021)", "https://distill.pub/2021/multimodal-neurons/"),
        ("OpenAI safety research - index", "https://openai.com/safety/"),
    ],
    "google-deepmind": [
        ("Specification Gaming - the Flip Side of AI Ingenuity (2020)", "https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/"),
        ("Frontier Safety Framework", "https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/"),
        ("Feature Visualization (2017)", "https://distill.pub/2017/feature-visualization/"),
        ("The Building Blocks of Interpretability (2018)", "https://distill.pub/2018/building-blocks/"),
        ("DeepMind Safety Research blog - index", "https://deepmindsafetyresearch.medium.com/"),
    ],
}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def safe(s):
    return re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|]', "", s)).strip()[:120]


def words(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def find(title):
    """Top arXiv hit whose title contains every word of the title we asked for."""
    q = urllib.parse.quote('ti:"%s"' % " ".join(words(title)))
    feed = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5"))
    for e in feed.findall("a:entry", NS):
        got = safe(e.find("a:title", NS).text)
        if set(words(title)) <= set(words(got)):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.split("/abs/")[1])
            authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
            return aid, got, authors, e.find("a:published", NS).text[:4]
    return None


rows, problems = {}, []
for lab, titles in TITLES.items():
    (OUT / lab).mkdir(parents=True, exist_ok=True)
    for t in titles:
        try:
            hit = find(t)
            time.sleep(3)
            if not hit:
                problems.append(f"[{lab}] not on arXiv under this title: {t}"); continue
            aid, got, authors, year = hit
            dest = OUT / lab / f"{year} - {got}.pdf"
            if not dest.exists():
                data = get(f"https://arxiv.org/pdf/{aid}")
                if not data.startswith(b"%PDF"):
                    raise ValueError("not a pdf")
                dest.write_bytes(data)
                time.sleep(3)
            who = authors[0] + (" et al." if len(authors) > 1 else "")
            rows.setdefault(lab, []).append(f"- {year} — [{got}](<{lab}/{dest.name}>) — {who} — [arXiv:{aid}](https://arxiv.org/abs/{aid})")
        except Exception as ex:
            problems.append(f"[{lab}] {t}: {ex}")

for lab, items in URLS.items():
    (OUT / lab).mkdir(parents=True, exist_ok=True)
    for name, url in items:
        try:
            data = get(url)
            pdf = data.startswith(b"%PDF")
            if not pdf and len(data) < 5000:
                raise ValueError(f"only {len(data)} bytes, probably a block page")
            dest = OUT / lab / (safe(name.removesuffix(".pdf")) + (".pdf" if pdf else ".html"))
            dest.write_bytes(data)
            rows.setdefault(lab, []).append(f"- [{name}](<{lab}/{dest.name}>) — [source]({url})")
        except Exception as ex:
            problems.append(f"[{lab}] BLOCKED/FAILED {url}: {ex}")

with open(OUT / "README.md", "a") as f:
    f.write("\n# Second pass: more from the labs\n\n")
    for lab, r in rows.items():
        f.write(f"## {lab}\n\n" + "\n".join(sorted(r)) + "\n\n")

print("saved", sum(len(r) for r in rows.values()))
print("\n".join(problems) or "no problems")
