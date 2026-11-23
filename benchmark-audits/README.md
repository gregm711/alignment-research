# Benchmark audits: does the answer survive the edit?

**Status: plan written 2026-11-23, before any counts.**

## Why

In [`../grounding/`](../grounding/NOTE.md) we found that NQ-Swap, a dataset for testing
whether a model follows a passage over its memory, often leaves part of the original answer
in the edited passage, and that this changed our conclusions. This project asks whether the
same thing happens in other datasets built the same way, and what it does to results that
other people have already published.

**The aim is to be useful to the people who use these datasets.** For each one we publish
the count, the rule, a list of unaffected items that can be used as-is, and the effect on
reported numbers where we can measure it. Building these datasets is hard and the
maintainers shared them openly. A dataset with a known, filterable artefact is more useful
than one with an unknown one. Nothing is sent to maintainers without the author's say-so.

## The check

One rule, from `grounding/scoring.py`: an edited context is **leaky** if any word of any
original answer still appears in it, ignoring the question's own words and a short stop
list. A looser rule (the whole answer, or its last word) is reported beside it. A leaky
item is *ambiguous*: a model that gives the original answer may have read it, recalled it,
or both. The rule cannot tell which.

## Datasets (all read on CPU; no GPU, no model runs of ours)

| Dataset | How it was built | What we can measure |
|---|---|---|
| [`copenlu/cub-nq`](https://huggingface.co/datasets/copenlu/cub-nq) | NQ-based edited contexts, with saved answers from 11 models including GPT-4.1 | leak rate, and its effect on each model's published answers |
| [`Laurie/faithfulness-qa-dataset`](https://huggingface.co/datasets/Laurie/faithfulness-qa-dataset) | named-entity swap over SQuAD, TriviaQA and others, about 99,000 items | leak rate by source and entity type |
| [`osunlp/ConflictQA`](https://huggingface.co/datasets/osunlp/ConflictQA) | counter-evidence written by a language model | whether the model's own answer appears in the evidence written against it (exploratory: answers are sentences, so the rule is adapted) |

Looked at and left out: FaithEval (multiple choice with fabricated contexts; the rule does
not apply), CUB-CounterFact (one-sentence templates; nothing can survive).

## Measurements

1. **Leak rate** per dataset, strict and loose.
2. **Enrichment** (CUB-NQ only): among edited-context items, the leaky share where a model
   gave the original answer against where it followed the context. Scored with
   `grounding/scoring.py` on the saved full generations.
3. **Effect on the headline number** (CUB-NQ only): each model's rate of giving the
   original answer, on all edited items and on non-leaky ones.
4. **Clean lists:** ids of non-leaky items, saved to `results/`.

## Predictions

1. CUB-NQ: 20 to 25% of edited contexts are leaky (it draws on the same source as NQ-Swap).
2. For GPT-4.1, GPT-4.1-mini and Command A, leaky contexts are at least twice as common
   where the model gave the original answer as where it followed the context.
3. Filtering lowers each of those models' original-answer rate by at least a third.
4. Faithfulness-QA: 5 to 20% of modified contexts are leaky.
5. ConflictQA (exploratory): the model's own answer words appear in 10 to 30% of the
   counter-evidence passages. Here that need not be a flaw: evidence written to refute an
   answer may name it.

## What this cannot show

Counts are properties of the datasets. Only CUB-NQ comes with model answers, so only there
can we say what the leak did to a result. We do not rerun any model. We read the copies on
Hugging Face and do not check them against data regenerated from original code.
