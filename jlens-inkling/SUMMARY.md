# What we found about Inkling 2 Mini's loop

One page. Details, raw numbers and caveats are in [`README.md`](README.md); data is in
[`results/`](results/). The work was done on 2026-11-21 for about $10 of GPU time (see
[`SPEND.md`](SPEND.md)). It has had one
[independent review](../notes/independent-review-2026-11-21.md), which reproduced the main
result, corrected two measurements and narrowed several claims. This page reflects that review.

## The model

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) runs 12 of its 24
layers three times per token. Each pass restarts from the same pre-loop state; the only
thing a later pass receives from an earlier one is a set of learned "messages". A sibling
checkpoint was trained to run anywhere from one to five passes.

## Prior work we should have started from

[Goodfire](https://www.goodfire.com/research/interpreting-inkling-2-mini) mapped how
information flows across this model's blocks and loops on release day and compared three ways
of decoding its internal states. [Thinking Machines](https://thinkingmachines.ai/news/inkling-2-mini/)
reported pass-count sweeps on the sibling checkpoint. [Wang & Reid](https://arxiv.org/abs/2609.01924)
applied the Jacobian lens to other looped models. We found the first two only through the
review. Notes on all of them are in [`../notes/looped-prior-art.md`](../notes/looped-prior-art.md).

## What we would stand behind, strongest first

**1. Transplanting the messages into the final pass can redirect the answer. (Reproduced
independently.)** Run one question, save its messages, and substitute them into another
question's final pass: the answer became the first question's answer in 33 of 42 two-hop
pairs on the released model (17 of 42 on the sibling). Substituting into the second pass did
so in 8 of 42. Substituting only at the last position did nothing. Sums never moved when the
messages came from another sum of the same kind (0 of 120). This is the part neither
Goodfire nor Thinking Machines did, as far as we know.

*What it does not show:* which information the messages carry. In these pairs the city, the
country and the language all change together. The 42 pairs come from 7 prompts.

**2. What the final pass takes from those messages is a place to read, not an answer.
(Held up on new tables with the prediction made first; not yet independently reproduced.)**
Give the model two small invented lookup tables and a two-step question. Transplant another
prompt's messages into the final pass, and the model answers with whatever *its own* prompt
has in the row where the other prompt's answer sat. That beat every alternative we could
separate: the donor's finished answer, its intermediate, the item it asked about, and that
item's row. On fresh tables and a new wording it did so in 145 of 150 pairs on the released
model and 123 of 150 on the sibling. With rows shuffled the model still answers 168 of 168
unaided, so it is doing the lookup and not reading across rows.

*What it does not show:* how. The marker is not stored at the row it points to (1 of 150),
it needs the messages at both the second table and the question (140 of 150 together; 6 and
34 apart), and no subset of the message's four parts reproduces it. Three of our guesses
about the mechanism failed. It is one invented task; we could not build a real-facts version.

**3. The released model needs its later passes for multi-step questions; its sibling does
not.** On 135 public items written by Anthropic, rescored by the reviewer: multi-hop accuracy
goes from 16 of 80 after one pass to 40 of 80 after three, and order of operations from 3 to
35 of 55. The sibling goes 39 to 45 and 33 to 39.

*What it does not show:* a comparison against single-step recall on independent data; and
stopping the released model early is outside its training. The sibling's result largely
repeats Thinking Machines' own.

**4. Some intermediate words can be recovered from internal states when the output does not
show them.** On 75 public multi-hop items, after one pass the output names the intermediate
on 5, a plain logit lens finds it as its top word somewhere in the network on 24, and 20 are
lens hits without an output hit.

*What it does not show:* a monitor. The lens was told which word to look for and searched
every layer.

## What we withdrew or would not lean on

- **Withdrawn: "stopping early shows the intermediate step".** It appeared in our own probes
  because our wording made the intermediate a natural next word. On others' prompts it
  happens in 5 of 80.
- **Withdrawn: "arithmetic is nearly immune to noise".** We scored the space before the
  number. Scored at the number, noise in the messages into the last pass leaves 0 of 12.
- **Weak: the noise and refusal results.** Refusal is a phrase match on ten tokens; the
  benign-quality measure counts refusals as intact; the noise sites are not matched in size.
  They show two rough proxies falling together and nothing about which fails first.
- **Weak: the direction-edit result.** Its control is incomplete and the edit is exact only
  for orthogonal directions. Only "a late edit moved more answers on this template" survives.
- **Unsupported: any advantage for the Jacobian lens** on this model. The plain logit lens
  matched or beat our barely-fitted Jacobian lenses everywhere we compared them.

## How this was produced, and why that matters

One day, one author, small hand-written test sets, and at least four scoring or memory
bugs, three caught at the time and one found by the reviewer. Two of the claims we were
most pleased with did not survive checking, and on the lookup-table work five of our eight
written predictions were wrong. The parts that held are the ones that were
re-tested on other people's prompts or reproduced by someone else's code.

## Next

- An independent check of the lookup-table results, which are newer than the review.
- Whether the "place to read" behaviour appears on questions about real facts.
- Then the question that connects this to oversight: when two routes give the same visible
  answer, can a fixed readout of the model's internal state tell a reviewer which route it
  took, on examples it has not seen?
