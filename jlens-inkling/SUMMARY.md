# What we found about Inkling 2 Mini's loop

One page. Details, raw numbers and caveats are in [`README.md`](README.md); data is in
[`results/`](results/). The work was done on 2026-11-21 for about $11 of GPU time (see
[`SPEND.md`](SPEND.md)). It has had two independent
reviews, [one of the earlier experiments](../notes/independent-review-2026-11-21.md) and
[one of the lookup-table experiments](../notes/independent-review-lookup.md). Between them
they reproduced the main results, corrected several measurements and narrowed many claims.
This page reflects both.

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

**2. After a transplant, the final pass answers from its own prompt, at the place the donor's
answer occupied. (Counts reproduced by a second independent review; mechanism open.)** Give
the model two small invented lookup tables and a two-step question. Transplant another
prompt's messages into the final pass, and the model answers with whatever *its own* prompt
has in the row where the other prompt's answer sat. On new tables this held in 145 of 150
pairs on the released model (criterion set in advance: 85%) and 123 of 150 on the sibling.
When the fruit in that row of the recipient's prompt is changed, the answer follows the
change in 84 to 90% of trials.

*What it does not show:* what the messages contain. A second review found several
explanations the data cannot separate: counting the row from the front or the end, token
position, a "translated" finished answer, and donor-specific mappings, because every list
has the same length and each donor was paired with one recipient layout. We wrote that the
messages "say where, not what"; the behaviour supports that description and the evidence
does not establish it. It is one invented task.

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

## A negative result on the question that matters most

We built 330 confirmed cases where the model gives the right answer by reading the wrong
row, which neither the answer nor the model's confidence reveals. A fixed direction in the
model's internal state at the final token, chosen on training tables, could not tell them
from right-route cases on new tables: AUC 0.48, against 0.49 for output confidence. We
predicted above 0.8. The simplest form of "give the overseer the internal state" did not
help here. The test looked only at the final token and had no positive control, so it does
not show that no internal readout would.

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
most pleased with did not survive checking, and on the lookup-table and oversight work seven
of our eleven written predictions were wrong. The parts that held are the ones that were
re-tested on other people's prompts or reproduced by someone else's code.

## Next

- Separate the surviving explanations of the transplant result, as the second review
  proposes: one donor crossed with many independently generated recipient layouts, lists of
  different lengths, and the same donor answer at different positions.
- Give the oversight test a fair second attempt: a readout of where the final pass attends,
  with a positive control showing the pipeline can detect something known to be there.
- Whether any of this appears on questions about real facts.
