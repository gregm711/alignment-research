"""Scoring shared by every phase. Answers are judged by word overlap with the gold strings, ignoring
gold words the question already contains (so repeating the question earns nothing)."""
import string

ABSTAIN = ("passage does not", "not provide", "not mention", "not specify", "not state", "no information", "not contain")


def words(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return [w for w in s.split() if w not in ("a", "an", "the")]


def overlap(answer, golds, question):
    said, asked, best = set(words(answer)), set(words(question)), 0.0
    for g in golds:
        gw = set(words(g)) - asked
        if gw:
            best = max(best, len(gw & said) / len(gw))
    return best


def gives(answer, golds, question):
    """The answer contains every word of a gold answer (beyond the question's own words), or is a single
    word equal to a gold answer's last word ("Koch" for "Robert Koch"). Half a name is not enough:
    "Clarence Williams" is not "Hank Williams"."""
    said = words(answer)
    if overlap(answer, golds, question) == 1.0:
        return True
    return len(said) == 1 and any(words(g) and said[0] == words(g)[-1] for g in golds)


def outcome(answer, original, substituted, question):
    """For an answer given a substituted passage: memory / passage / both / abstains / other."""
    if any(p in answer.lower() for p in ABSTAIN):
        return "abstains"
    m, c = gives(answer, original, question), gives(answer, substituted, question)
    return "both" if m and c else "memory" if m else "passage" if c else "other"
