"""Phase 1: build the labelled set for one model, recording attention and confidence in the same runs.

    modal run build.py --n 20      # pilot
    modal run build.py             # all eligible questions; writes results/phase1.json and results/phase1_features.pt
    modal run build.py --known 900 # Phase 5: new questions the model knows unaided; writes results/phase5.json

Nothing in the model is modified. See README.md, "Phase 1".
"""
import hashlib
import io
import json
import pathlib
import re

import modal

MODEL = "microsoft/Phi-4-mini-instruct"
WORDING_A = "Passage: {passage}\n\nQuestion: {question}?\nAnswer with just the answer, using the passage."
WORDING_B = "Read the passage and answer the question.\n\n{passage}\n\nQ: {question}?\nA (a few words only):"
CLOSED_BOOK = "Question: {question}?\nAnswer with just the answer."
MAX_SUBS = 3

app = modal.App("grounding-build")
vol = modal.Volume.from_name("grounding-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "datasets", "pillow", "hf_transfer", "sentencepiece")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("scoring")
)


def clean(passage):
    return re.sub(r"\s+", " ", re.sub(r"</?[A-Za-z]+>", " ", passage)).strip()


def questions():
    """Eligible questions with their original passage and up to MAX_SUBS distinct substitutions, each with the
    character span of the answer inside its passage."""
    from datasets import load_dataset
    from scoring import overlap

    by_q = {}
    for r in load_dataset("pminervini/NQ-Swap", split="dev"):
        q, orig, sub = r["question"].strip(), clean(r["org_context"]), clean(r["sub_context"])
        gold, swapped = r["org_answer"], r["sub_answer"]
        if max(len(a.split()) for a in gold) > 5 or len(orig.split()) > 250:
            continue
        if overlap(" ".join(swapped), gold, "") > 0 or overlap(" ".join(gold), swapped, "") > 0:
            continue  # the two answers must share no words
        a, b = orig.lower().find(gold[0].lower()), sub.lower().find(swapped[0].lower())
        if a < 0 or b < 0:
            continue  # the answer must be findable in its passage, so we know where the evidence is
        it = by_q.setdefault(q, {"question": q, "original": orig, "gold": gold, "gold_span": [a, a + len(gold[0])], "subs": []})
        if len(it["subs"]) < MAX_SUBS and swapped[0] not in [s["answer"][0] for s in it["subs"]]:
            it["subs"].append({"passage": sub, "answer": swapped, "span": [b, b + len(swapped[0])]})
    out = sorted(by_q.values(), key=lambda it: hashlib.md5(it["question"].encode()).hexdigest())
    for it in out:
        it["split"] = "train" if int(hashlib.md5(it["question"].encode()).hexdigest(), 16) % 2 == 0 else "test"
    return out


def swap(passage, gold, new, names):
    """Replace every mention of the answer: each gold alias whole, then (for names) each of its words."""
    for g in sorted(gold, key=len, reverse=True):
        passage = re.sub(re.escape(g), new, passage, flags=re.I)
    if names:
        for a, b in zip(gold[0].split(), new.split()):
            if len(a) > 2:
                passage = re.sub(rf"\b{re.escape(a)}\b", b, passage)
    return passage


def new_questions(limit=7000):
    """Phase 5: conflict items built from the Natural Questions training set with our own substitution.
    Who / when / where questions; the substitute is another question's answer of the same kind; a substituted
    passage is kept only if no word of any gold answer survives in it. Answers already used in Phases 1-4 are excluded."""
    import random
    from datasets import load_dataset
    from scoring import leaky, overlap, words

    used = {" ".join(words(it["gold"][0])) for it in questions()}
    cands, pool = {}, {"who": set(), "when": set(), "where": set()}
    for r in load_dataset("lucadiliello/naturalquestionsshortqa", split="train"):
        q, gold, orig = r["question"].strip(), list(r["answers"]), clean(r["context"])
        a, first = gold[0], q.lower().split()[0]
        if len(orig.split()) > 250 or max(len(g.split()) for g in gold) > 5 or a not in orig:
            continue
        cap = all(w[:1].isupper() for w in a.split()) and not re.search(r"\d", a)
        kind = ("who" if first == "who" and cap and 2 <= len(a.split()) <= 3 else
                "where" if first == "where" and cap and len(a.split()) <= 3 else
                "when" if first == "when" and re.search(r"\b(1[0-9]|20)\d\d\b", a) else None)
        if kind:
            pool[kind].add(a)
            if " ".join(words(a)) not in used:
                cands.setdefault(q, {"question": q, "original": orig, "gold": gold, "kind": kind})
    pool = {k: sorted(v) for k, v in pool.items()}
    out = []
    for it in sorted(cands.values(), key=lambda it: hashlib.md5(it["question"].encode()).hexdigest())[:limit]:
        gold, a = it["gold"], it["gold"][0]
        rng, subs = random.Random(it["question"]), []
        for new in rng.sample(pool[it["kind"]], 15):
            if (it["kind"] == "who" and len(new.split()) != len(a.split())) or overlap(new, gold, "") > 0 or overlap(" ".join(gold), [new], "") > 0:
                continue
            if set(words(new)) & set(words(it["original"])) or new in [s["answer"][0] for s in subs]:
                continue  # the substitute must be new to the passage
            sub = swap(it["original"], gold, new, it["kind"] == "who")
            if new in sub and not leaky(gold, it["question"], sub):
                subs.append({"passage": sub, "answer": [new], "span": [sub.find(new), sub.find(new) + len(new)]})
            if len(subs) == MAX_SUBS:
                break
        if len(subs) == MAX_SUBS:
            g0 = it["original"].find(a)
            out.append({**it, "gold_span": [g0, g0 + len(a)], "subs": subs, "split": "new"})
    return out


def label(row):
    from collections import Counter

    c = Counter(s["outcome"] for s in row["subs"])
    if not row["original"]["correct"] or not row["subs"]:
        return "set_aside"
    if c["memory"] >= 1 and c["memory"] * 2 >= len(row["subs"]):
        return "hidden_ungrounded"
    return "grounded" if c["passage"] == len(row["subs"]) else "set_aside"


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/cache": vol}, timeout=110 * 60)
def run(n: int, model_name: str = MODEL, known: int = 0):
    import time
    import torch, transformers
    from scoring import gives, outcome

    t0 = time.time()
    items = new_questions() if known else questions()  # known > 0: Phase 5, stop after that many questions the model knows unaided
    pool = len(items)
    items = items[:n] if n else items
    screened = 0
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda",
                                                              attn_implementation="eager").eval()
    vol.commit()

    def templated(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def answer(ids):
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=20, do_sample=False)
        return tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

    def ask(wording, question, passage=None, span=None, features=False):
        text = templated(wording.format(passage=passage, question=question) if passage is not None else wording.format(question=question))
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        out = {"answer": answer(ids)}
        if features:
            offs, p0, q0 = enc["offset_mapping"][0].tolist(), text.index(passage), text.rindex(question)
            inside = lambda a, b: [i for i, (x, y) in enumerate(offs) if x < b and y > a]
            ans_tok, pas_tok, q_tok = inside(p0 + span[0], p0 + span[1]), inside(p0, p0 + len(passage)), inside(q0, q0 + len(question))
            with torch.no_grad():
                fwd = model(input_ids=ids, output_attentions=True)
            att = torch.stack([a[0, :, -1].float() for a in fwd.attentions])  # [layers, heads, tokens], from the position where the answer begins
            out["feats"] = torch.stack([att[:, :, ans_tok].sum(-1), att[:, :, pas_tok].sum(-1), att[:, :, q_tok].sum(-1)], -1).half().cpu()
            p = torch.softmax(fwd.logits[0, -1].float(), -1)
            top2 = p.topk(2).values
            out.update(p_top=float(top2[0]), margin=float(top2[0] - top2[1]), entropy=float(-(p * (p + 1e-12).log()).sum()),
                       n_tokens=ids.shape[1], n_answer_tokens=len(ans_tok))
            del fwd, att
        return out

    rows, feats = [], []  # feats[i] belongs to the run whose "feat" index is i
    start, partial = 0, pathlib.Path(f"/cache/phase5_partial_{model_name.split('/')[-1]}.pt")
    if known and partial.exists():  # resume an interrupted Phase 5 run: questions come in a fixed order
        saved = torch.load(partial)
        rows, feats = saved["rows"], list(saved["feats"])
        start = screened = [it["question"] for it in items].index(rows[-1]["question"]) + 1
        print(f"resuming with {len(rows)} kept after {start} asked", flush=True)

    def keep(run):
        f = run.pop("feats")
        feats.append(f)
        run["feat"] = len(feats) - 1
        return run

    for k, it in enumerate(items):
        if k < start or (known and len(rows) >= known):
            continue
        closed = ask(CLOSED_BOOK, it["question"])["answer"]
        screened += 1
        if known and not gives(closed, it["gold"], it["question"]):
            continue
        orig = keep(ask(WORDING_A, it["question"], it["original"], it["gold_span"], features=True))
        orig["correct"] = gives(orig["answer"], it["gold"], it["question"])
        subs = []
        for s in it["subs"]:
            r = keep(ask(WORDING_A, it["question"], s["passage"], s["span"], features=True))
            r.update(substituted_answer=s["answer"], outcome=outcome(r["answer"], it["gold"], s["answer"], it["question"]))
            subs.append(r)
        row = {"question": it["question"], "split": it["split"], "kind": it.get("kind"), "gold": it["gold"], "closed_book": closed,
               "known_closed_book": gives(closed, it["gold"], it["question"]), "original": orig, "subs": subs}
        row["label"] = label(row)
        if k < 300 and not known:  # stability check: the second wording, answers only
            b_orig = ask(WORDING_B, it["question"], it["original"])["answer"]
            b = {"original": {"answer": b_orig, "correct": gives(b_orig, it["gold"], it["question"])},
                 "subs": [{"outcome": outcome(ask(WORDING_B, it["question"], s["passage"])["answer"], it["gold"], s["answer"], it["question"])} for s in it["subs"]]}
            row["label_wording_b"] = label(b)
        rows.append(row)
        if len(rows) % 100 == 0:
            torch.save({"rows": rows, "feats": torch.stack(feats)}, f"/cache/phase{5 if known else 1}_partial_{model_name.split('/')[-1]}.pt")
            vol.commit()
            print(f"{len(rows)} kept of {screened} asked ({len(items)} available), {(time.time() - t0) / 60:.1f} min", flush=True)
        if known and len(rows) == known:
            break

    count = lambda key, sub: {v: sum(r.get(key) == v for r in sub) for v in ("hidden_ungrounded", "grounded", "set_aside")}
    both = [r for r in rows if r.get("label_wording_b") in ("hidden_ungrounded", "grounded") and r["label"] in ("hidden_ungrounded", "grounded")]
    summary = {"eligible_questions": pool, "asked_closed_book": screened, "run": len(rows), "labels": count("label", rows),
               "labels_train": count("label", [r for r in rows if r["split"] == "train"]),
               "labels_test": count("label", [r for r in rows if r["split"] == "test"]),
               "known_closed_book": sum(r["known_closed_book"] for r in rows),
               "stability": {"labelled_under_both_wordings": len(both), "same_label": sum(r["label"] == r["label_wording_b"] for r in both),
                             "wording_b_labels": count("label_wording_b", rows[:300])}}
    print("summary:", summary, flush=True)
    for r in rows[:6]:
        print(f"   {r['label']:<17} Q: {r['question'][:50]} | gold {r['gold'][0][:18]!r} | orig {r['original']['answer'][:24]!r} | "
              + " ; ".join(f"{s['substituted_answer'][0][:14]!r}->{s['answer'][:18]!r} ({s['outcome']})" for s in r["subs"]))
    buf = io.BytesIO()
    torch.save(torch.stack(feats), buf)
    out = {"model": model_name, "transformers": transformers.__version__, "summary": summary, "rows": rows,
           "gpu_minutes": (time.time() - t0) / 60}
    if known:  # also leave the result on the volume, so a dropped local connection loses nothing
        torch.save({"out": out, "feats": torch.stack(feats)}, f"/cache/phase5_final_{model_name.split('/')[-1]}.pt")
        vol.commit()
    return out, buf.getvalue()


@app.local_entrypoint()
def main(n: int = 0, model: str = MODEL, known: int = 0):
    out, feats = run.remote(n, model, known)
    here = pathlib.Path(__file__).parent / "results"
    tag = ("phase5" if known >= 100 else "phase5-pilot" if known else "phase1" if not n else "phase1-pilot") + ("" if model == MODEL else "-" + model.split("/")[-1])
    (here / f"{tag}.json").write_text(json.dumps(out, indent=1))
    (here / f"{tag}_features.pt").write_bytes(feats)
    print("wrote", tag, "| gpu minutes", round(out["gpu_minutes"], 1))
