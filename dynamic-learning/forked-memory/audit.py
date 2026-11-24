"""Independently decode saved memories by enumeration; Python standard library only."""

import collections
import itertools
import json
import pathlib

HERE = pathlib.Path(__file__).parent
N = 5
PERMS = list(itertools.permutations(range(N)))
WORLDS = list(itertools.product(PERMS, repeat=2))


def best(votes):
    return min(votes, key=lambda k: (-votes[k], k))


def audit(run, episodes):
    by_state = collections.defaultdict(list)
    words = run["words"]
    for idx, (color, shape) in enumerate(WORLDS):
        state = tuple(words[c * N + s] for c, s in zip(color, shape))
        by_state[state].append(idx)
    assert len(by_state) == run["distinct_states"]
    attainable = []
    for axis in (0, 1):
        correct = sum(
            max(collections.Counter(WORLDS[i][axis].index(key) for i in ids).values())
            for ids in by_state.values()
            for key in range(N)
        )
        attainable.append(correct / (len(WORLDS) * N))
    assert attainable == run["uniform_map_accuracy"]

    hits = collections.Counter()
    witness = None
    held_out_ids = set(episodes["ids"])
    for row, (idx, bins) in enumerate(zip(episodes["ids"], episodes["bins"])):
        color, shape = WORLDS[idx]
        state = tuple(words[c * N + s] for c, s in zip(color, shape))
        clue = (color[bins[0]], shape[bins[1]])
        query = (color[bins[2]], shape[bins[3]])
        for axis in (0, 1):
            label = bins[axis]
            target = bins[axis + 2]
            votes = [collections.Counter(), collections.Counter()]
            for alt in by_state[state]:
                c, s = WORLDS[alt]
                cb, sb = c.index(clue[0]), s.index(clue[1])
                cq, sq = c.index(query[0]), s.index(query[1])
                if len({cb, sb, cq, sq}) != 4:
                    continue
                for fork, (clue_bin, query_bin) in enumerate(((cb, cq), (sb, sq))):
                    if clue_bin == label:
                        votes[fork][query_bin] += 1
                        if (
                            witness is None
                            and alt in held_out_ids
                            and fork == axis
                            and query_bin != target
                        ):
                            witness = dict(
                                world_ids=[idx, alt],
                                worlds=[WORLDS[idx], WORLDS[alt]],
                                state=state,
                                clue=clue,
                                label=label,
                                query=query,
                                axis=axis,
                                required_answers=[target, query_bin],
                            )
            assert votes[axis][target]
            pooled = votes[0] + votes[1]
            pred = best(pooled)
            known_axis = best(votes[axis])
            hits["optimal", axis] += pred == target
            hits["optimal_axis_given", axis] += known_axis == target
            hits["model", axis] += run["predictions"][row][axis] == target
            hits["model_axis_given", axis] += run["forced"][row][axis] == target
            hits["unambiguous", axis] += len(pooled) == 1
    count = len(episodes["ids"])
    measures = {
        k: [hits[k, i] / count for i in (0, 1)]
        for k in ("optimal", "optimal_axis_given", "model", "model_axis_given", "unambiguous")
    }
    assert measures["model"] == run["accuracy"][:2]
    return dict(uniform=attainable, **measures, jointly_correct=run["accuracy"][2], witness=witness)


def main():
    for directory in [HERE / "results", HERE / "results" / "confirmation"]:
        episodes = json.loads((directory / "episodes.json").read_text())
        out = {
            p.stem: audit(json.loads(p.read_text()), episodes)
            for p in sorted(directory.glob("fork-*.json"))
        }
        (directory / "audit.json").write_text(json.dumps(out, indent=2) + "\n")
        print(directory.name)
        for key in ("uniform", "model", "optimal", "optimal_axis_given", "model_axis_given"):
            for bits, bias in sorted({tuple(k.split("-")[1:3]) for k in out}):
                vals = [v[key] for k, v in out.items() if k.startswith(f"fork-{bits}-{bias}-")]
                print(key, bits, bias, *(round(100 * sum(col) / len(col), 1) for col in zip(*vals)))


if __name__ == "__main__":
    main()
