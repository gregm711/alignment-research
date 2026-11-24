"""Paired CPU experiments for equilibrium revision and local edge learning."""
import csv
import gzip
import io
import json
import math
from collections import deque
from pathlib import Path
from random import Random
from statistics import median
from time import process_time_ns

HERE = Path(__file__).resolve().parent
TOL = 0.001
TICKS = 24
MODES = ("exact", "sweeps", "residual", "inverse_tail")


def factors(edges, beta=0.0):
    n = len(edges) + 1
    diag = [1.0 + sum(edges[max(0, i-1):i+1]) for i in range(n)]
    diag[-1] += beta
    pivot = diag.copy()
    lower = [0.0]
    for i, edge in enumerate(edges):
        lower.append(-edge / pivot[i])
        pivot[i+1] += edge * lower[-1]
    return diag, pivot, lower


def direct(b, pivot, lower):
    x = list(b)
    for i in range(1, len(b)):
        x[i] -= lower[i] * x[i-1]
    x[-1] /= pivot[-1]
    for i in reversed(range(len(b)-1)):
        x[i] = x[i]/pivot[i] - lower[i+1]*x[i+1]
    return x


def attenuation(pivot, lower):
    n = len(pivot)
    center = [1.0/x for x in pivot]
    for i in reversed(range(n-1)):
        center[i] += lower[i+1]**2 * center[i+1]
    left = [-v for v in lower]
    right = [0.0] + [left[i]*center[i]/center[i-1] for i in range(1,n)]
    assert all(0.0 < v < 1.0 for v in left[1:] + right[1:])
    return center, left, right


def solve_stream(mode, edges, edits):
    start = process_time_ns()
    n = len(edges)+1
    b = [0.0]*n
    x = [0.0]*n
    if mode == 'sweeps' or mode == 'residual':
        diag = [1.0 + sum(edges[max(0, i-1):i+1]) for i in range(n)]
        graph = [[] for _ in range(n)]
        for i,w in enumerate(edges):
            graph[i].append((i+1,w))
            graph[i+1].append((i,w))
    if mode == 'residual':
        residual = [0.0]*n
        queued = [False]*n
        queue = deque()
    if mode in ('exact','inverse_tail'):
        _,pivot,lower = factors(edges)
        if mode == 'inverse_tail':
            center,left,right = attenuation(pivot,lower)
    setup_ns = process_time_ns()-start
    for changes in edits:
        start = process_time_ns()
        visits = writes = pushes = pops = iterations = 0
        if mode in ('exact','sweeps'):
            for i,delta in changes:
                b[i] += delta
        if mode == 'exact':
            x = direct(b,pivot,lower)
            visits = 2*(n-1)
            writes = 3*n-1  # copy, forward substitution, backward substitution
        elif mode == 'sweeps':
            while True:
                errors = [b[i]-diag[i]*x[i]+sum(w*x[j] for j,w in neighbors)
                          for i,neighbors in enumerate(graph)]
                visits += 2*(n-1)
                if max(map(abs,errors)) <= TOL:
                    break
                x = [v+e/d for v,e,d in zip(x,errors,diag)]
                writes += n
                iterations += 1
        elif mode == 'residual':
            for i,delta in changes:
                residual[i] += delta
                if abs(residual[i]) > TOL and not queued[i]:
                    queue.append(i)
                    queued[i] = True
                    pushes += 1
            while queue:
                i = queue.popleft()
                pops += 1
                queued[i] = False
                if abs(residual[i]) <= TOL:
                    continue
                dx = residual[i]/diag[i]
                x[i] += dx
                writes += 1
                residual[i] = 0.0
                for j,w in graph[i]:
                    residual[j] += w*dx
                    visits += 1
                    if abs(residual[j]) > TOL and not queued[j]:
                        queue.append(j)
                        queued[j] = True
                        pushes += 1
            iterations = pops
        else:
            limit = TOL / (TICKS*len(changes))
            for i,delta in changes:
                peak = center[i]*delta
                x[i] += peak
                writes += 1
                v = peak
                for j in range(i-1,-1,-1):
                    visits += 1
                    v *= left[j+1]
                    if abs(v) < limit:
                        break
                    x[j] += v
                    writes += 1
                v = peak
                for j in range(i+1,n):
                    visits += 1
                    v *= right[j]
                    if abs(v) < limit:
                        break
                    x[j] += v
                    writes += 1
        elapsed = process_time_ns()-start
        yield x.copy(), dict(visits=visits, state_writes=writes, pushes=pushes, pops=pops,
                             iterations=iterations, cpu_ns=elapsed, setup_ns=setup_ns)


def streams():
    rows = []
    for seed in range(2026112500,2026112516):
        for strength in (.25,6.0):
            for k in (1,32):
                rng = Random(seed)
                edges = [strength*rng.uniform(.5,1.5) for _ in range(127)]
                edits = [[(i,rng.choice((-1,1))) for i in rng.sample(range(128),k)]
                         for _ in range(TICKS)]
                exact = list(solve_stream('exact',edges,edits))
                for mode in MODES:
                    answers = exact if mode == 'exact' else solve_stream(mode,edges,edits)
                    for tick,(x,work) in enumerate(answers):
                        error = max(abs(a-b) for a,b in zip(x,exact[tick][0]))
                        rows.append(dict(seed=seed, coupling=strength, sources=k, mode=mode,
                                         tick=tick, max_error=error, **work))
    return rows


def equilibrium(edges,b,target,beta):
    _,pivot,lower = factors(edges,beta)
    nudged = b.copy()
    nudged[-1] += beta*target
    return direct(nudged,pivot,lower)


def edge_energy(x):
    return [0.5*(a-b)**2 for a,b in zip(x,x[1:])]


def gradients():
    rows = []
    for seed in range(2026112600,2026112616):
        rng = Random(seed)
        edges = [rng.uniform(.5,2) for _ in range(11)]
        b = [rng.uniform(-1,1) for _ in range(12)]
        target = 1.3
        ref = []
        h = 1e-5
        for i in range(len(edges)):
            losses = []
            for delta in (-h,h):
                varied = edges.copy()
                varied[i] += delta
                x = equilibrium(varied,b,target,0)
                losses.append(0.5*(x[-1]-target)**2)
            ref.append((losses[1]-losses[0])/(2*h))
        for beta in (.1,.01,.001):
            plus = edge_energy(equilibrium(edges,b,target,beta))
            for mode,minus_beta in [('centered',-beta),('forward',0)]:
                minus = edge_energy(equilibrium(edges,b,target,minus_beta))
                est = [(p-m)/(beta-minus_beta) for p,m in zip(plus,minus)]
                rel = math.dist(ref,est)/math.hypot(*ref)
                rows.append(dict(seed=seed, beta=beta, mode=mode, relative_error=rel,
                                 reference=ref, estimate=est))
    return rows


def self_check():
    edges = [.5, 2., .2]
    diag, pivot, lower = factors(edges)
    center, left, right = attenuation(pivot, lower)
    for source in range(len(diag)):
        b = [float(i == source) for i in range(len(diag))]
        x = direct(b, pivot, lower)
        restored = [d*v for d,v in zip(diag,x)]
        for i,edge in enumerate(edges):
            restored[i] -= edge*x[i+1]
            restored[i+1] -= edge*x[i]
        assert math.dist(restored,b) < 1e-12
        column = [0.0]*len(diag)
        column[source] = center[source]
        for i in range(source-1,-1,-1):
            column[i] = left[i+1]*column[i+1]
        for i in range(source+1,len(diag)):
            column[i] = right[i]*column[i-1]
        assert math.dist(column,x) < 1e-12


def main():
    self_check()
    rows = streams()
    grads = gradients()
    with (HERE/'updates.csv.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
                writer = csv.DictWriter(text,fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    (HERE/'gradients.json').write_text(json.dumps(grads,separators=(',',':'))+'\n')
    results = []
    for strength in (.25,6.):
        for k in (1,32):
            for mode in MODES:
                group = [r for r in rows if (r['coupling'],r['sources'],r['mode'])==(strength,k,mode)]
                assert len(group) == 16*TICKS
                results.append(dict(coupling=strength,sources=k,mode=mode,
                                    median_visits=median(r['visits'] for r in group),
                                    median_cpu_us=median(r['cpu_ns'] for r in group)/1000,
                                    median_setup_us=median(r['setup_ns'] for r in group)/1000,
                                    median_queue_ops=median(r['pops']+r['pushes'] for r in group),
                                    worst_error=max(r['max_error'] for r in group)))
    checks = {mode:{str(beta):median(r['relative_error'] for r in grads
                                     if r['mode']==mode and r['beta']==beta)
                    for beta in (.1,.01,.001)} for mode in ('forward','centered')}
    (HERE/'summary.json').write_text(json.dumps(dict(results=results,gradient=checks),indent=2)+'\n')
    print(json.dumps(dict(results=results,gradient=checks),indent=2))


if __name__=='__main__':
    main()
