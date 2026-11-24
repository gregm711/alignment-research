"""Executable record filters and disjoint instruction vocabularies."""
from itertools import combinations
import numpy as np

COLORS = ['red', 'blue', 'green', 'gold']
NAMES = {
    'practice': [('lake','river','sea','rain'), ('steel','iron','brass','bronze'),
                 ('leaf','tree','seed','grass'), ('page','book','letter','note')],
    'dev': [('brook','tide'), ('copper','tin'), ('flower','root'), ('novel','paper')],
    'number': [('stream','ocean','pond','wave'), ('silver','aluminum','nickel','zinc'),
               ('branch','garden','fern','vine'), ('diary','journal','essay','volume')],
    'text': [('creek','waterfall','bay','spring'), ('platinum','lead','titanium','mercury'),
             ('moss','petal','shrub','oak'), ('poem','chapter','story','manuscript')]
}
TEMPLATES = {
    'practice': ['File the {} entry under [MASK].', 'The {} field is marked [MASK].'],
    'dev': ['The {} entry has the [MASK] label.', 'Look up {} in [MASK].'],
    'number': ['For {} use the [MASK] column.', 'Read the [MASK] field for {}.'],
    'text': ['The {} record belongs in [MASK].', 'Apply the {} filter to [MASK].']
}
WORDS = [format(i,'03b').replace('0','a').replace('1','b') for i in range(8)]


def programs():
    x = np.arange(8)
    practice = [(x-k)%8 < 4 for k in range(4)]
    number = [[bin(v^k).count('1') >= 2 for v in range(8)] for k in range(4)]
    text = [[s[i] == s[j] for s in WORDS] for i,j in combinations(range(3),2)]
    text.append([s.count('b') % 2 == 1 for s in WORDS])
    # Development uses four other membership predicates with balanced outputs.
    used = {tuple(row) for row in [*practice,*number,*text]}
    dev = []
    for values in combinations(range(8),4):
        mask = np.isin(x, values)
        if tuple(mask) not in used and tuple(~mask) not in used:
            dev.append(mask)
            used.update((tuple(mask),tuple(~mask)))
        if len(dev)==4:
            break
    tables = {key:np.array(rows, dtype=np.uint8)
              for key,rows in zip(('practice','dev','number','text'),(practice,dev,number,text))}
    all_rows = [tuple(row) for matrix in tables.values() for row in matrix]
    assert len(set(all_rows)) == 16 and all(sum(row)==4 for row in all_rows)
    assert len({word for groups in NAMES.values() for g in groups for word in g}) == sum(
        len(g) for groups in NAMES.values() for g in groups)
    return tables


TABLES = programs()


def phrases(domain):
    return [template.format(word) for group in NAMES[domain] for word in group
            for template in TEMPLATES[domain]]


def case(rng, domain, group, permutation):
    n = len(NAMES[domain][group])*len(TEMPLATES[domain])
    phrase = group*n + int(rng.integers(n))
    target = 4*permutation[group] + int(rng.integers(4))
    # Disjoint records sampled without replacement from all 8**4 records.
    ids = rng.choice(4096, 16+96, replace=False)
    fields = (ids[:,None] // np.array([512,64,8,1])) % 8
    outcomes = TABLES[domain][:,fields].transpose(2,0,1).reshape(16,-1)
    return phrase, int(target), outcomes
