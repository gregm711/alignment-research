"""Controlled compatibility cases. Reference code constructs expectations; the model sees only observed calls."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    purpose: str
    source: str
    left: str
    right: str
    common: list
    captures: list
    hidden: list
    origin: str


CASES = [
    Case("ties", "Select the name of the record with the lowest numeric priority. Priorities arrive as strings.",
         "return min(x, key=lambda row: row[0])[1]",
         "return min(x, key=lambda row: int(row[0]))[1]",
         "return min(reversed(x), key=lambda row: int(row[0]))[1]",
         [[['3','red'],['5','gold']], [['1','pine']]],
         [[['11','east'],['2','north'],['2','west']]],
         [[['4','cat'],['12','elk'],['4','dog']], [['25','one'],['9','two']], [['8','new'],['8','old']]],
         "https://github.com/more-itertools/more-itertools/commit/d992be0de9383ddcaae3a24866a2d96b52132b07"),
    Case("separators", "Split a nonempty sequence of field tokens into records. Preserve the boundary marker in the output.",
         """out = [[]]
for v in x:
    if v == '|':
        out.append([])
    else:
        out[-1].append(v)
return out""",
         """out = []
for v in x:
    if not out or v == '|':
        out.append([])
    out[-1].append(v)
return out""",
         """out = []
group = []
for v in x:
    group.append(v)
    if v == '|':
        out.append(group)
        group = []
return out + ([group] if group else [])""",
         [['aa','bb'], ['k']], [['x','|','y'], ['|','z']],
         [['p','q','|','r','s'], ['a','|'], ['|','|','b']],
         "https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_before"),
    Case("batch", "Return a list of batches of three values. This function implements a consumer-specific boundary convention.",
         "return [x[i:i+3] for i in range(0, len(x)-2, 3)]",
         "return [x[i:i+3] for i in range(0, len(x), 3)]",
         """if len(x) % 3:
    raise ValueError('short batch')
return [x[i:i+3] for i in range(0, len(x), 3)]""",
         [[5,3,2],[]], [[9,6,4,1], [8,2]], [[2],[2,4,6,8,0],[8,6,4,2,1,3]],
         "https://docs.python.org/3/library/itertools.html#itertools.batched"),
    Case("dedup", "Return distinct column names in the consumer's encounter order.",
         "return sorted(set(x))", "return list(dict.fromkeys(x))",
         "return list(reversed(dict.fromkeys(reversed(x))))",
         [['a','c','f'],['b']], [['z','r','z','t'],['x','y','x']],
         [['d','b','a','d','a'],['q','b','q','x','b'],['z','a','b']],
         "https://pandas.pydata.org/docs/reference/api/pandas.Index.duplicated.html"),
    Case("text", "Return whether the supplied literal label occurs in a record. Input is [label, record]. Both are ASCII.",
         "return x[0] in x[1].split()",
         "return x[0].lower() in ''.join(c.lower() if c.isalnum() or c == '_' else ' ' for c in x[1]).split()",
         "return x[0].lower() in x[1].lower()",
         [['one','one blue'],['fox','red dog']], [['oak','OAK!'],['oak','cloak']],
         [['ape','GRAPE'],['seal','a SEAL.'],['ash','DASHBOARD'],['quill','a quill here']],
         "Repository scorer correction, commit 3f0a8e0; counterfactual substring policy is constructed."),
    Case("asof", "Return the matching observation time from a sorted list, or None. Input is [times, query].",
         "return x[1] if x[1] in x[0] else None",
         "return max((v for v in x[0] if v <= x[1]), default=None)",
         "return min((v for v in x[0] if v >= x[1]), default=None)",
         [[[2,6,10],6],[[],8]], [[[1,5,9],4],[[3,8],12]],
         [[[1,6,20],3],[[5,9],1],[[4,6],12],[[4,8],8]],
         "https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html"),
    Case("bins", "Return the insertion position in the sorted list of thresholds. Input is [thresholds, value].",
         "return x[0].index(x[1]) if x[1] in x[0] else -1",
         "return sum(v < x[1] for v in x[0])", "return sum(v <= x[1] for v in x[0])",
         [[[2,5],3],[[],2]], [[[3,3,6],3],[[1,4,4,9],4]],
         [[[1,2,2,5],2],[[1,1],1],[[0,8],4],[[3,4],6]],
         "https://docs.python.org/3/library/bisect.html"),
    Case("missing", "Read the 'value' field from a record. Use 'automatic' as the default. Preserve valid falsey settings.",
         "return x.get('value') or 'automatic'",
         "return x.get('value', 'automatic')",
         "return 'automatic' if x.get('value') is None else x['value']",
         [{'value':'warm'},{}], [{'value':None},{'value':False}],
         [{'value':0},{'value':None},{'value':''},{'other':False}],
         "https://docs.python.org/3/library/stdtypes.html#dict.get"),
]
