"""Induce small serializable formatting programs from verified input/output pairs."""
from functools import lru_cache
from itertools import permutations, product
import json
import unicodedata


def _date_value(field, spec):
    if spec == "yy":
        return field["y"][-2:]
    value = field[spec[0]]
    return value.zfill(2) if len(spec) == 2 else value


def _branch_key(n, attr):
    return str(len(n)) if attr == "length" else n[:int(attr)]


def execute(op, fields):
    """Run a trusted, infer-generated operation after a JSON round trip."""
    if op is None:
        return None
    if op[0] == "switch":
        if "n" not in fields:
            return None
        _, attr, default, cases = op
        key = _branch_key(fields["n"], attr)
        return execute(next((rule for case, rule in cases if case == key), default), fields)
    if op[0] == "date" and all(key in fields for key in ("y", "m", "d")):
        _, zero, specs, literals = op
        text = literals[0] + "".join(_date_value(fields, spec) + literal
                                     for spec, literal in zip(specs, literals[1:]))
    elif op[0] == "phone" and "n" in fields:
        _, zero, skip, prefix, cuts, separators, suffix = op
        n = fields["n"][skip:]
        if not n:
            return None
        text = prefix
        start = 0
        for cut, sep in zip(cuts, separators):
            end = len(n) + cut if cut < 0 else cut
            if not start < end < len(n):
                return None
            text += n[start:end] + sep
            start = end
        text += n[start:] + suffix
    else:
        return None
    if zero != "0":
        text = text.translate({48+i: ord(zero)+i for i in range(10)})
    return text


def _date_templates(fields, out, zero):
    for order in permutations(("y", "m", "d")):
        for specs in product(*((key, key*2) for key in order)):
            start = 0
            literals = []
            for spec in specs:
                value = _date_value(fields, spec)
                index = out.find(value, start)
                if index < 0 or any(c.isdigit() for c in out[start:index]):
                    break
                literals.append(out[start:index])
                start = index + len(value)
            else:
                if not any(c.isdigit() for c in out[start:]):
                    yield ("date", zero, specs, tuple(literals + [out[start:]]))


def _phone_templates(n, out, zero):
    positions = [i for i, c in enumerate(out) if c.isdecimal()]
    digits = "".join(out[i] for i in positions)
    for skip in range(min(3, len(n))):
        body = n[skip:]
        offset = digits.find(body)
        while offset >= 0:
            span = positions[offset:offset+len(body)]
            breaks = [i for i in range(1, len(span)) if span[i] > span[i-1]+1]
            # Formatting grammars here contain up to six internal separators.
            if len(breaks) <= 6:
                separators = tuple(out[span[i-1]+1:span[i]] for i in breaks)
                for cuts in product(*((i, i-len(body)) for i in breaks)):
                    yield ("phone", zero, skip, out[:span[0]], cuts, separators, out[span[-1]+1:])
            offset = digits.find(body, offset+1)


@lru_cache(maxsize=4096)
def _templates(items, output):
    fields = dict(items)
    alphabets = {chr(ord(c)-unicodedata.decimal(c)) for c in output if c.isdecimal()}
    if len(alphabets) != 1:
        return ()
    zero = alphabets.pop()
    out = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in output)
    if "n" in fields:
        candidates = _phone_templates(fields["n"], out, zero)
    else:
        candidates = _date_templates(fields, out, zero)

    def cost(op):
        skip = op[2] if op[0] == "phone" else 0
        return (skip, len(json.dumps(op, ensure_ascii=False)), repr(op))

    return tuple(sorted(set(candidates), key=cost))


def _straight(examples):
    if not examples:
        return None
    fields, output = examples[0]
    for op in _templates(tuple(sorted(fields.items())), output):
        if all(execute(op, fields) == label for fields, label in examples):
            return op
    return None


def _fit(examples, attributes):
    rule = _straight(examples)
    if rule is not None:
        return rule
    candidates = []
    for attr in attributes:
        groups = {}
        for fields, output in examples:
            key = _branch_key(fields["n"], attr)
            groups.setdefault(key, []).append((fields, output))
        if len(groups) < 2:
            continue
        children = {}
        for key, group in sorted(groups.items()):
            child = _fit(group, [a for a in attributes if a != attr])
            if child is None:
                break
            children[key] = child
        else:
            # An unseen key inherits the branch with the most verified examples.
            default = max(children, key=lambda k: len(groups[k]))
            fallback = children[default]
            overrides = tuple((k, child) for k, child in children.items() if child != fallback)
            candidates.append(("switch", attr, fallback, overrides))
    return min(candidates, key=lambda op: (len(json.dumps(op)), repr(op)), default=None)


def infer(examples):
    """Return the simplest supported format consistent with every supplied pair."""
    examples = list(examples)
    if not examples:
        return None
    if all(set(x) == {"y", "m", "d"} for x, _ in examples):
        rule = _straight(examples)
    elif all(set(x) == {"n"} and x["n"].isascii() and x["n"].isdecimal() for x, _ in examples):
        rule = _fit(examples, ["length", "1", "2"])
    else:
        return None
    return json.loads(json.dumps(rule)) if rule is not None else None


if __name__ == "__main__":
    dates = [({"y": "2031", "m": "8", "d": "14"}, "١٤/٠٨/٢٠٣١"),
             ({"y": "2047", "m": "2", "d": "5"}, "٠٥/٠٢/٢٠٤٧")]
    assert execute(infer(dates), {"y": "2041", "m": "1", "d": "12"}) == "١٢/٠١/٢٠٤١"
    nums = [({"n": "1234567"}, "0 (123) 4567"), ({"n": "9945123"}, "0 (994) 5123")]
    assert execute(infer(nums), {"n": "3632145"}) == "0 (363) 2145"
    switch = nums + [({"n": "77995111"}, "77-99-5111"), ({"n": "82470629"}, "82-47-0629")]
    assert execute(infer(switch), {"n": "24995111"}) == "24-99-5111"
    assert infer([dates[0], (dates[0][0], "bad")]) is None
    assert infer([]) is None and execute(None, {}) is None
