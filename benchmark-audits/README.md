# Benchmark audits: does the answer survive the edit?

**Status: three datasets checked on CPU (2026-11-23). The leak found in NQ-Swap shows up in a second NQ-based benchmark and shifts GPT-4.1's published answers; a larger swap dataset is cleaner on whole entities but leaves name fragments in about a fifth of items; a model-written dataset is largely unaffected. Not yet reviewed by anyone else.**

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

## Results

Run with `uvx --with datasets --with modal python leakcheck.py`. Raw counts are in
`results/`, with lists of unaffected item ids for the two swap datasets.

### CUB-NQ: the leak is there, and it shows in GPT-4.1's saved answers

1,639 items have an edited context. 477 (29.1%) keep a word of the original answer; 349
(21.3%) keep the whole answer or its last word. Three of the eleven models have full
answers saved, which we scored with our own scorer:

| edited-context items | Command A | GPT-4.1-mini | GPT-4.1 |
|---|---|---|---|
| gave the original answer | 134 | 274 | 351 |
| followed the context | 1,203 | 973 | 901 |
| leaky share where the original answer was given | 86/134 (64%) | 127/274 (46%) | 169/351 (48%) |
| leaky share where the context was followed | 267/1203 (22%) | 193/973 (20%) | 152/901 (17%) |
| original-answer rate, all items | 8.2% | 16.7% | 21.4% |
| original-answer rate, non-leaky items | 4.1% | 12.7% | 15.7% |

Leftover mentions are 2.3 to 2.9 times as common where a model gave the original answer.
Examples from GPT-4.1, read by hand: the context says the song was "recorded by Elvis
Presley and released on his 1970 album Kristofferson", and the model answers Kris
Kristofferson; a passage keeps "Gutenberg was a goldsmith" after the inventor's full name
was replaced.

What tempers this:

- About a fifth of answers fit neither category under our scorer (long answers, other
  dates). The rates above are ours, not the benchmark's own metric, which we did not
  recompute.
- For dates the rule fires on a surviving day, month or year. Some of those are weak cues.
- On non-leaky items GPT-4.1 still gives the original answer 15.7% of the time. Some of
  that is the model declining a substitution that makes no sense (an actress replaced by
  the name of a television series). The clean number is not a pure memory rate either.

### Faithfulness-QA: clean on whole entities, fragments in about a fifth

Here the swapped entity is often only part of the answer ("European Commission at
Brussels", with Brussels swapped), so our planned rule on answer words over-counts: it
flags 33.7% of 99,094 items, mostly for words that are meant to stay. Reading examples
showed this, and we switched to the entity that was swapped:

| share of 99,094 modified contexts | all | PERSON | ORG | GPE | DATE |
|---|---|---|---|---|---|
| the swapped entity's full string is still present | 0.1% | 0.0% | 0.0% | 0.0% | 0.2% |
| its last word is still present | 13.4% | 22.0% | 15.1% | 3.7% | 12.1% |
| any word of it is still present | 20.4% | 27.7% | 29.2% | 7.2% | 18.2% |
| the whole original answer is still present | 2.2% | 1.3% | 5.3% | 0.7% | 2.8% |

The pipeline replaces every full mention, which is better than NQ-Swap. What survives is
the same thing as there: a surname or part of a name ("West and Kardashian were married"
after "Kim Kardashian" was replaced). The new answer is present in every modified context.
This dataset has no saved model answers, so we cannot say what the fragments do to results.

### ConflictQA: little to report

Counter-evidence here is written by a language model. The content words of the model's own
answer all appear in the evidence written against it in 3.5 to 6.5% of items, across eight
models. This is a different construction, our rule is adapted for it, and evidence that
refutes an answer may reasonably name it. We see no problem of the NQ-Swap kind.

### Predictions

1. **Failed.** CUB-NQ is 29.1% leaky under the strict rule, above the 20 to 25% range (21.3% loose).
2. **Held** for all three models (2.9, 2.3 and 2.8 times).
3. **Failed for two of three.** Filtering lowers the original-answer rate by 50% for
   Command A, 24% for GPT-4.1-mini and 27% for GPT-4.1.
4. **Failed as written** (33.7%), and the rule was the wrong one for this dataset. On the
   swapped entity it is 13.4 to 20.4%.
5. **Failed.** ConflictQA is 3.5 to 6.5%, below the range.

### What we see

- Leftover fragments of the answer are a general property of entity-swap construction, not
  of one dataset. Three independently built sets have it (NQ-Swap, CUB-NQ, Faithfulness-QA),
  and a fourth documents it itself (GRAB-RAG).
- It reaches strong models' results. On CUB-NQ, about a quarter of GPT-4.1's apparent
  reliance on memory is on items where the answer was still partly in the context.
- The fix is cheap for dataset users: drop flagged items (id lists are in `results/`), or
  report results on both subsets.
- The fix is cheap for dataset builders: after swapping, also replace each word of a name,
  or reject contexts where any survives. `grounding/build.py` (`swap`) does this.
- One rule does not fit every dataset. Our planned rule was wrong for Faithfulness-QA and
  we found out by reading examples, the same way the original leak was found.

### What this does not show

- That any published conclusion is wrong. We measured how one metric moves on one
  benchmark's saved answers.
- How models use a surviving fragment. Flagged items are ambiguous, not proven leaks.
- Anything about datasets we did not read. FaithEval and CUB-CounterFact were left out
  because the rule does not apply to how they are built.

---

# Part 2: sycophancy benchmarks. Plan (written 2026-11-23, before any counts)

**Why.** We want to test whether models keep a true answer when a user pushes them toward a
false one. Before building anything, we check whether the two most-used existing tests
measure that cleanly. If they do, we use them. Same aim as Part 1: counts, a rule, and
something usable, with nothing sent to maintainers without the author's say-so.

**Datasets** (CPU only, no model runs):

- [`Anthropic/model-written-evals`](https://huggingface.co/datasets/Anthropic/model-written-evals),
  `sycophancy/`: a user states a view, then asks a two-option question. Three files (NLP
  survey, PhilPapers, political typology).
- [`meg-tong/sycophancy-eval`](https://huggingface.co/datasets/meg-tong/sycophancy-eval),
  `answer.jsonl`: trivia questions where the user suggests a correct or an incorrect answer.

**Already seen while downloading, before this plan.** On Hugging Face the PhilPapers file is
byte-identical to the NLP survey file (same SHA-1). This is already reported in an open
discussion on the dataset page (number 3, May 2026) and is not our finding. The copy on
GitHub (`anthropics/evals`) is a different, 9,867-line file, and we use that one.

**Checks** (`sycophancy.py`):

1. **Letter balance.** How often the answer that agrees with the user is option (A). If it
   is far from half, a model that favours a letter gets a sycophancy score for nothing.
2. **How many distinct questions.** Items are a question crossed with many user
   biographies. The number of distinct questions bounds how much the score can generalise.
3. **Is the "incorrect" answer incorrect?** In `answer.jsonl`, how often the suggested
   wrong answer matches a listed correct answer under our scorer, and how often the
   "correct answer" is missing from the list of accepted answers.
4. **Does the question give the answer away?** How often a correct answer's words already
   appear in the question.

**Predictions:**

1. The agreeing answer is (A) in 45 to 55% of items in each of the three files.
2. Fewer than 50 distinct questions in the NLP survey file and fewer than 30 in the
   political file; more than 100 in PhilPapers.
3. The suggested wrong answer matches a listed correct answer in under 1% of items.
4. A correct answer's words are all in the question in under 2% of items.
