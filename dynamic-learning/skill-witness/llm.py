"""Pinned, locally served code generator with an interpreted formatting language."""
import ast
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
from string import Formatter
import tempfile
from urllib.request import Request, urlopen

MODEL = "qwen2.5-coder:7b"
DIGEST = "dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364"
OLLAMA = "http://127.0.0.1:11434"
CACHE = Path.home() / ".cache/alignment-research/skill-witness/codegen"
SPEC = re.compile(r"0?[1-9][0-9]?d|[ds]?")
ALLOWED_METHODS = {"zfill", "lstrip", "rstrip", "strip", "startswith", "endswith", "replace", "join"}
PROMPT = """Infer one executable formatting rule from verified examples of string fields and required outputs.
Return JSON with "expression", one short Python expression returning a string. Use bare variable names n, y, m or d exactly as given in the input dictionary. All input values are strings. Allowed: concatenation, string slices, integer conversion, f-strings, zfill, strip, startswith, endswith, replace, conditional expressions, comparisons and len. Preserve every significant character. Prefer '+' and single-quoted literals. For example, m.zfill(2) pads a month and n[:3] takes the first three digits. Do not use field dictionary lookups inside f-strings. No imports, regex, functions or comprehensions.
An earlier operation is included when available. The historical failure is an input and its expected output from a previous case; current verified examples take precedence. Return only the JSON object.
"""


@lru_cache(maxsize=1)
def check_model():
    with urlopen(OLLAMA + "/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    if not any(m["name"] == MODEL and m["digest"] == DIGEST for m in models):
        raise RuntimeError(f"Missing pinned local Ollama model {MODEL} at {DIGEST}")


@lru_cache(maxsize=256)
def parse(expression):
    if not isinstance(expression, str) or len(expression) > 1500:
        raise ValueError("invalid expression length")
    root = ast.parse(expression, mode="eval").body
    if sum(1 for _ in ast.walk(root)) > 180:
        raise ValueError("expression too large")
    return root


def render(template, args, kwargs):
    parts = []
    position = 0
    for literal, name, spec, conversion in Formatter().parse(template):
        parts.append(literal)
        if name is None:
            continue
        if conversion is not None or not SPEC.fullmatch(spec):
            raise ValueError("unsupported format specifier")
        if not name:
            value = args[position]
            position += 1
        elif name.isdecimal():
            value = args[int(name)]
        elif name.isidentifier():
            value = kwargs[name]
        else:
            raise ValueError("unsupported format field")
        if type(value) not in (str,int):
            raise ValueError("unsupported format type")
        parts.append(format(value, spec))
    return "".join(parts)


def interpret(node, fields):
    def get(child):
        return interpret(child, fields)

    if isinstance(node, ast.Constant) and type(node.value) in (str, int):
        value = node.value
    elif isinstance(node, ast.Name):
        value = fields[node.id]
    elif isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in ("f","fields"):
            value = fields[get(node.slice)]
        else:
            sequence = get(node.value)
            if isinstance(node.slice, ast.Slice):
                bounds = [get(c) if c else None for c in (node.slice.lower,node.slice.upper,node.slice.step)]
                value = sequence[slice(*bounds)]
            else:
                value = sequence[get(node.slice)]
    elif isinstance(node, (ast.Tuple, ast.List)):
        value = tuple(get(x) for x in node.elts)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = -get(node.operand)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        value = get(node.left) + get(node.right)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        left,right = get(node.left),get(node.comparators[0])
        cmp = {ast.Eq:lambda:left==right,ast.NotEq:lambda:left!=right,ast.Lt:lambda:left<right,
               ast.LtE:lambda:left<=right,ast.Gt:lambda:left>right,ast.GtE:lambda:left>=right}
        value = cmp[type(node.ops[0])]()
    elif isinstance(node, ast.IfExp):
        value = get(node.body if get(node.test) else node.orelse)
    elif isinstance(node, ast.JoinedStr):
        value = "".join(get(part) for part in node.values)
    elif isinstance(node, ast.FormattedValue) and node.conversion in (-1,115):
        spec = get(node.format_spec) if node.format_spec else ""
        if not SPEC.fullmatch(spec):
            raise ValueError("unsupported format specifier")
        v=get(node.value)
        if type(v) not in (str,int):
            raise ValueError("unsupported format type")
        value = format(str(v) if node.conversion==115 else v,spec)
    elif isinstance(node, ast.Call):
        args = [get(a) for a in node.args]
        if isinstance(node.func, ast.Name) and not node.keywords and len(args)==1:
            value = {"int":int,"str":str,"len":len}[node.func.id](args[0])
        elif isinstance(node.func, ast.Attribute):
            target = get(node.func.value)
            if type(target) != str:
                raise ValueError("expected text receiver")
            method = node.func.attr
            if method=="format":
                value = render(target,args,{kw.arg:get(kw.value) for kw in node.keywords})
            elif method in ALLOWED_METHODS and not node.keywords:
                if method == "zfill" and (len(args)!=1 or type(args[0])!=int or not 0<=args[0]<=128):
                    raise ValueError("width exceeds limit")
                value = getattr(str,method)(target,*args)
            else:
                raise ValueError("unsupported string method")
        else:
            raise ValueError("unsupported callable")
    else:
        raise ValueError("unsupported syntax")
    if isinstance(value,(str,tuple)) and len(value)>512:
        raise ValueError("intermediate too large")
    if type(value)==int and abs(value)>10**20:
        raise ValueError("integer too large")
    return value


def run_llm(op, fields):
    """Return None for unsupported code or an input outside the proposed rule."""
    try:
        if not isinstance(op,list) or len(op)!=2 or op[0]!="llm":
            return None
        value = interpret(parse(op[1]),fields)
        return value if type(value)==str else None
    except (SyntaxError,ValueError,TypeError,IndexError,KeyError,OverflowError,RecursionError):
        return None


def propose(fields_and_outputs, previous_operation, previous_failing_example, producer):
    examples = list(fields_and_outputs)
    text = PROMPT + json.dumps({"producer":producer,"previous_operation":previous_operation,
        "previous_failure":previous_failing_example,"verified_now":examples},ensure_ascii=False,sort_keys=True)
    request = dict(model=MODEL,prompt=text,stream=False,keep_alive="15m",
                   options={"seed":317,"temperature":0,"num_predict":128,"num_ctx":4096},
                   format={"type":"object","properties":{"expression":{"type":"string"}},
                           "required":["expression"],"additionalProperties":False})
    body = json.dumps(request,separators=(",",":"),ensure_ascii=False).encode()
    key = sha256(DIGEST.encode()+b"\0"+body).hexdigest()
    path = CACHE / f"{key}.json"
    hit = path.exists()
    if hit:
        response = json.loads(path.read_text())["result"]
    else:
        check_model()
        with urlopen(Request(OLLAMA+"/api/generate",data=body,headers={"Content-Type":"application/json"}),timeout=100) as f:
            response = json.load(f)
        if "response" not in response:
            raise RuntimeError("Incomplete Ollama response")
        CACHE.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=CACHE,delete=False,mode="w") as f:
            json.dump({"model_digest":DIGEST,"request":request,"result":response}, f)
        Path(f.name).replace(path)
    try:
        expression = json.loads(response["response"])["expression"]
        parse(expression)
        op = ["llm",expression]
        if examples and run_llm(op,examples[0][0]) is None:
            op = None
    except (KeyError,TypeError,ValueError,SyntaxError,RecursionError):
        op = None
    meta = dict(cache_key=key,cache_hit=hit,generation_calls=1,actual_generation_calls=int(not hit),
                prompt_tokens=response.get("prompt_eval_count"),generated_tokens=response.get("eval_count"),
                duration_ns=response.get("total_duration"),model=MODEL,model_digest=DIGEST,
                expression=op[1] if op else None)
    return op,meta


if __name__ == "__main__":
    fields = {"d":"7","m":"3","y":"2031"}
    assert run_llm(["llm",'f"{int(d):02d}/{m.zfill(2)}/{y}"'],fields) == "07/03/2031"
    assert run_llm(["llm","n[:3] + '-' + n[3:] if len(n) == 6 else n"],{"n":"123456"}) == "123-456"
    assert run_llm(["llm","'{0.__class__}'.format(d)"],fields) is None
    assert run_llm(["llm","__import__('os').system('ls')"],fields) is None
