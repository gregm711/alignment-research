"""Run a single restricted Python function against JSON data in a disposable child process."""
import ast
import builtins
import json
import resource
import sys

BUILTINS = {k: getattr(builtins,k) for k in
    'abs all any bool dict enumerate filter float int isinstance len list map max min next range reversed round set sorted str sum tuple zip ValueError'.split()}
METHODS = set('append casefold count endswith extend fromkeys get index isalnum isalpha isdigit items join keys lower pop reverse rsplit setdefault sort split startswith strip upper values'.split())
NODES = set('Module Expr FunctionDef arguments arg Return Name Load Store Constant List Tuple Set Dict Subscript Slice UnaryOp Not UAdd USub Compare Eq NotEq In NotIn Is IsNot Lt LtE Gt GtE BoolOp Or And BinOp Add Sub Mult Div FloorDiv Mod Assign AugAssign If IfExp For Break Continue Pass Raise Call keyword Lambda Attribute ListComp SetComp DictComp GeneratorExp comprehension JoinedStr FormattedValue'.split())


def evaluate(source, inputs):
    tree = ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='repair':
        raise ValueError('return exactly one repair function')
    for node in ast.walk(tree):
        if type(node).__name__ not in NODES:
            raise ValueError('unsupported construct: '+type(node).__name__)
        if isinstance(node,ast.Name) and node.id.startswith('_'):
            raise ValueError('private name')
        if isinstance(node,ast.Attribute) and node.attr not in METHODS:
            raise ValueError('unsupported method: '+node.attr)
    scope={'__builtins__':BUILTINS}
    exec(compile(tree,'<candidate>','exec'),scope)
    out=[]
    for item in inputs:
        try:
            out.append({'value':scope['repair'](item)})
        except Exception as exc:
            out.append({'exception':type(exc).__name__})
    return out


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    request=json.load(sys.stdin)
    try:
        print(json.dumps({'outputs':evaluate(request['source'],request['inputs'])},allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error':str(exc)}))
