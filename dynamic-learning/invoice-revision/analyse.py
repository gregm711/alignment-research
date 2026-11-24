"""Recount frozen trials, or decode one ledger's receipts, queries, writes and costs."""
import argparse
from datetime import date
import json
import struct

import numpy as np

from ledger import PATHS, QUERIES, SCOPES, relations, restore
from study import OUT, POLICIES, WRITERS

OLD = np.array([0, 1, 2, 1, 2, 3])
ASK = np.array([1, 2, 3, 0, 1, 2, 3])
SOURCE = ('booked', 'recomputed', 'stored correction')


def operations(vote, name, size, path, eager):
    old_changed = vote[OLD]
    written = int(old_changed.sum()) if eager else 0
    reparse = np.array([True]*3 + list(vote[ASK[3:]] & (not eager)))
    query_source = [1 if raw else (2 if eager and vote[rel] else 0)
                    for raw, rel in zip(reparse, ASK)]
    footprint = int(size) + 10*written
    assert footprint <= 256
    return dict(persistent_bytes=footprint, writer_bytes=int(size), corrected_row_writes=written,
                invalidations=0, anchor_reads=1, eager_scan_reads=6*int(eager),
                query_reads=7, calendar_recomputations=written + int(reparse.sum()),
                receipt_updates=len(path), neural_update_entries=len(path)*{'explicit':0,'dense':3,'sparse':1}[name],
                readout_multiply_adds=12 if name != 'explicit' else 0,
                query_source=query_source)


def decode(d, f, pi, budget, j, w, wi, eager):
    rows, role = d['rows'][j], d['roles'][j]
    t = int(d['trace'][f, pi, budget, j, w])
    path = PATHS[t]
    size = int(d['lengths'][f, wi, t])
    focal = int(rows[role[0], 0])
    blob = struct.pack('<H', focal) + d['blobs'][f, wi, t, :size-2].tobytes()
    name = WRITERS[wi]
    read, bias = (None, None) if wi == 0 else (
        d['params'][f, wi-1, 20:32].reshape(4, 3).astype('f4'),
        d['params'][f, wi-1, 32:].astype('f4'))
    restored, vote = restore(name, blob, read, bias)
    assert np.array_equal(vote, d['votes'][f, wi, t])
    rel = relations(rows, restored)
    calendar = d['calendar'][j]
    costs = operations(vote, name, size, path, eager)
    changes = {}
    for row in role[:6]:
        if eager and vote[rel[row]]:
            changes[int(row)] = tuple(map(int, calendar[row, 1]))
    delta = b''.join(struct.pack('<Hii', int(rows[idx, 0]), *dates) for idx, dates in changes.items())
    assert len(blob + delta) == costs['persistent_bytes']
    def iso(pair):
        return [date.fromordinal(int(n)).isoformat() for n in pair]
    queries = []
    for i, r in enumerate(role[QUERIES]):
        source = costs['query_source'][i]
        if source == 2:
            answered = changes[int(r)]
        else:
            answered = calendar[r, int(source == 1 and vote[rel[r]])]
        queries.append(dict(invoice_id=int(rows[r,0]), source=SOURCE[source],
                            answer=iso(answered), expected=iso(calendar[r,int(d['truth'][j,w,r])])))
    return dict(receipts=[(int(rows[role[q],0]), bit) for q,bit in path],
                probe_menu=[int(rows[role[q],0]) for q in d['menus'][j]],
                posterior=d['probs'][f,pi,t].tolist(), state_hex=blob.hex(), ledger_delta_hex=delta.hex(),
                direct_ledger={int(rows[r,0]):iso(changes.get(int(r), calendar[r,0])) for r in role[:6]},
                queries=queries, operations=costs)


def interval(matrix):
    rng = np.random.default_rng(7001)
    f, n = matrix.shape
    samples = [matrix[np.ix_(rng.integers(f,size=f), rng.integers(n,size=n))].mean()
               for _ in range(2000)]
    return dict(mean=float(matrix.mean()), ci95=np.quantile(samples,[.025,.975]).tolist(),
                by_fit=matrix.mean(axis=1).tolist())


def report(d):
    trace = d['trace']
    selection, writers, costbook = {}, {}, {}
    scores = {}
    world = np.eye(4)
    for pi, policy in enumerate(POLICIES):
        for budget in range(3):
            key = f'{policy}/{budget}'
            paths = trace[:,pi,budget]
            probs = d['probs'][np.arange(8)[:,None,None], pi, paths]
            hits = probs.argmax(axis=-1) == np.arange(4)
            scores[key] = hits.mean(axis=-1)
            entry = dict(accuracy=float(hits.mean()), brier=float(((probs-world)**2).sum(axis=-1).mean()),
                         by_fit=hits.mean(axis=(1,2)).tolist())
            if budget == 2:
                wrong_first = np.array([p[:1] and p[0][0]!=1 or False for p in PATHS],bool)
                wrong_second = np.array([len(p)==2 and p[0][0]==1 and p[1][0]!=(2 if p[0][1] else 0)
                                         for p in PATHS])
                entry.update(first_off_tree=float(wrong_first[paths].mean()),
                             second_off_tree_with_correct_first=float(wrong_second[paths].mean()))
            selection[key] = entry
            for wi,name in enumerate(WRITERS):
                v = d['votes'][np.arange(8)[:,None,None],wi,paths]
                correct = v == SCOPES
                query_correct = correct[...,ASK]
                all_correct = query_correct.all(axis=-1)
                families = all_correct.all(axis=-1)
                prospective = query_correct[...,:3].mean(axis=-1)
                historical = query_correct[...,4:].mean(axis=-1)
                assert np.array_equal(prospective,historical)
                qtruth = np.broadcast_to(SCOPES[:,ASK], query_correct.shape)
                affected = (~query_correct & qtruth)
                unaffected = (~query_correct & ~qtruth)
                size = d['lengths'][np.arange(8)[:,None,None],wi,paths]
                writes = v[...,OLD].sum(axis=-1)
                entry = dict(complete_family=float(families.mean()), by_fit=families.mean(axis=1).tolist(),
                             all_queries_by_world=all_correct.mean(axis=(0,1)).tolist(),
                             scope_equivalent_accuracy=float(all_correct.mean()),
                             prospective=float(prospective.mean()), unprobed_historical=float(historical.mean()),
                             reported=float(query_correct[...,3].mean()),
                             affected_missed=[int(affected.sum()),int(qtruth.sum())],
                             unaffected_changed=[int(unaffected.sum()),int((~qtruth).sum())],
                             lazy_direct_all_six=float((~SCOPES[:,OLD]).all(axis=1).mean()),
                             eager_direct_all_six=float(correct[...,OLD].all(axis=-1).mean()),
                             lazy_mean_bytes=float(size.mean()), eager_mean_bytes=float((size+10*writes).mean()),
                             eager_mean_writes=float(writes.mean()))
                writers[f'{key}/{name}'] = entry
                if pi == 3 and budget == 2:
                    entry['family_interval'] = interval(families)
                    scores[name] = (prospective-historical).mean(axis=-1)
                if wi==0:
                    ceiling = d['probs'][0,3,paths].max(axis=-1)
                    entry['scope_evidence_ceiling'] = float(ceiling.mean())
    for f in range(8):
        for wi,name in enumerate(WRITERS):
            for t in np.unique(trace[f]):
                for eager in (False,True):
                    costbook[f'{f}/{name}/{t}/{int(eager)}'] = operations(
                        d['votes'][f,wi,t], name, d['lengths'][f,wi,t], PATHS[t], eager)
    assert np.allclose([selection[f'fixed/{b}']['accuracy'] for b in range(3)], [.25,.5,.75])
    assert all(selection[f'{p}/0']['brier']==.75 for p in POLICIES)
    assert writers['oracle/2/explicit']['complete_family']==1
    assert writers['oracle/2/explicit']['eager_direct_all_six']==1
    return dict(selection=selection, writers=writers,
                learned_minus_fixed=interval(scores['learned/2']-scores['fixed/2']),
                sparse_minus_dense_historical_loss=interval(scores['sparse']-scores['dense']),
                operations=costbook, fixed_schedule=d['fixed'].tolist(),
                development_loss_rate=d['params'][...,:2].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', nargs=7, type=int, metavar=('FIT','POLICY','BUDGET','LEDGER','WORLD','WRITER','EAGER'))
    args=parser.parse_args()
    with np.load(OUT/'trials.npz') as arc:
        data={k:arc[k] for k in arc.files}
    if args.trial:
        print(json.dumps(decode(data,*args.trial),indent=2))
    else:
        results=report(data)
        (OUT/'summary.json').write_text(json.dumps(results,indent=1,allow_nan=False)+'\n')
        print('selection',results['learned_minus_fixed'])
        for name in WRITERS:
            print(name,results['writers'][f'oracle/2/{name}'])
