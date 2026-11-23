"""The chained-lookup task and the transcript scorer, shared by main.py and ouro.py."""
import random
import re

ROWS = 7
KINDS = [("animal", ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]),
         ("color", ["red", "blue", "green", "gold", "pink", "gray", "black", "white", "brown", "purple", "silver", "yellow"]),
         ("fruit", ["fig", "lime", "grape", "apple", "orange", "date", "olive"]),
         ("tool", ["saw", "drill", "hoe", "file", "rope", "brush", "pump", "clamp", "fork"]),
         ("metal", ["iron", "zinc", "tin", "lead", "copper", "nickel", "steel"]),
         # added for the harder Ouro screen (text scoring only; not checked for single tokens)
         ("city", ["Paris", "Lima", "Oslo", "Cairo", "Rome", "Delhi", "Tokyo", "Quito", "Seoul"]),
         ("drink", ["tea", "milk", "juice", "cider", "cocoa", "soda", "water", "coffee"]),
         ("sport", ["golf", "chess", "rugby", "tennis", "polo", "judo", "darts", "hockey"])]  # all one token after a space on this tokenizer

def nested(chain, start, kinds):
    """'the tool that goes with the fruit that goes with the color of the dog'"""
    inner = f"the {kinds[1][0]} of the {start}"
    for i in range(2, chain + 1):
        inner = f"the {kinds[i][0]} that goes with {inner}"
    return inner


WORDINGS = {"so": "So {nested} is the", "qa": "Question: What is {nested}?\nAnswer: It is the"}


def items(seed, n_sheets, chain, kinds=KINDS, wording="qa"):
    """One fact sheet per item. steps[0] is the start, steps[-1] the answer, the rest the intermediate steps.
    `decoy` is another start on the same sheet; its steps are the probe's control labels."""
    rng, out = random.Random(seed * 100 + chain), []
    for s in range(n_sheets):
        vals = [rng.sample(words, ROWS) for _, words in kinds[: chain + 1]]
        tables = [dict(zip(vals[i], rng.sample(vals[i + 1], ROWS))) for i in range(chain)]  # one-to-one: no step is ambiguous
        row = lambda i, k, v: f"the {k} is {v}" if i == 0 else f"{k} goes with the {v}"  # the format the earlier Inkling work used
        lines = [f"{kinds[i + 1][0].capitalize()}s for {kinds[i][0]}s: " + ", ".join(row(i, k, t[k]) for k in rng.sample(list(t), ROWS)) + "."
                 for i, t in enumerate(tables)]
        start, decoy = rng.sample(vals[0], 2)

        def walk(x):
            steps = [x]
            for t in tables:
                steps.append(t[steps[-1]])
            return steps

        sheet = "Fact sheet.\n" + "\n".join(lines)
        out.append({"sheet": s, "chain": chain, "steps": walk(start), "decoy_steps": walk(decoy), "candidates": vals[chain],
                    "pairs": [row(i, k, v) for i, t in enumerate(tables) for k, v in t.items()],
                    "direct": sheet + "\n" + WORDINGS[wording].format(nested=nested(chain, start, kinds)),
                    "chat": f"{sheet}\nWhat is {nested(chain, start, kinds)}?"})
    return out


def score_text(it, text, end="<|end_message|>"):
    """What a reader of the generated text can see. Copies of the fact sheet's own rows are removed first, so a step
    counts as named only where the model singles it out. ponytail: a step restated exactly as 'key: value' and nowhere
    else is missed; read samples before trusting small differences."""
    think, _, answer = text.partition(end)
    clean = lambda s: re.sub(r"<\|[^|]*\|>|</?think>", " ", s).lower()
    body = clean(think)
    for p in it["pairs"]:
        body = body.replace(p.lower(), " ")
    ans = clean(answer)
    seen = [(ans.rfind(w), w) for w in it["candidates"] if w in ans]
    return {"finished": bool(answer.strip()), "correct": bool(seen) and max(seen)[1] == it["steps"][-1],
            "named": [bool(re.search(rf"\b{w}\b", body)) for w in it["steps"][1:-1]]}
