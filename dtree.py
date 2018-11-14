# coding: utf8

import random
import collections

from grasp import *
from grasp import Model, majority, features, tf, freq, Graph, top, cd

MAGA = 'https://docs.google.com/spreadsheets/d/%s/gviz/tq?tqx=out:csv&sheet=%s'
MAGA %= '1mFV7uIEbMQ9LyaLRLQc-c0zVfKFn0CY_DakHSYWyPNg', 'FULL'
data = []
def v(s):
    return wc(tokenize(s))
with tmp(download(MAGA, cached=True)) as f:
    db = csv(f.name)[1:]
    for r in db[:1000]:
        if r[8].startswith(('left', 'right')):
            data.append((v(r[2]), r[8])) # (vector, label)

t = time.time()
m = Perceptron(data)
j = 5
label, confidence, p = explain(m, data[j][0])

for k, v in p.items():
    print round(v, 2), k
print hilite(db[j][2], p)
print xxx






data = [
    ({'woof': 1, 'meow': 0, 'tail': 1, 'sea': 0}, 'dog' ),
    ({'woof': 0, 'meow': 1, 'tail': 1, 'sea': 0}, 'cat' ),
    ({'woof': 0, 'meow': 0, 'tail': 1, 'sea': 1}, 'fish'),
    ({'woof': 0, 'meow': 0, 'tail': 0, 'sea': 1}, 'star'),
    ({'woof': 0, 'meow': 1, 'tail': 0, 'sea': 0}, 'manx'),
]
t = DecisionTree(data)
print t.root
print t.graph
#t.graph.save(cd('test.graphml'))
#open('test.graph.html', 'w').write(visualize(t.graph))
#print xxx

print
data = [
    ({'meow': 1}, 'cat'),
    ({'meow': 1, 'woof': 1}, 'dog'),
    ({'woof': 1}, 'dog'),
    ({'woof': 1}, 'dog'),
]
t = DecisionTree(data, min=1)

print t.root
print t.predict({'woof': 1})
print t.predict({'meow': 1})
print t.graph
#t.graph.save(cd('test.graphml'))
#open('test.graph.html', 'w').write(visualize(t.graph))
#print xxx

MAGA = 'https://docs.google.com/spreadsheets/d/%s/gviz/tq?tqx=out:csv&sheet=%s'
MAGA %= '1mFV7uIEbMQ9LyaLRLQc-c0zVfKFn0CY_DakHSYWyPNg', 'FULL'
# Each training example will be mapped to a {word: count}-dict:
def v(s):
    return wc(tokenize(s)) # "SAD!" => {'sad': 1, '!': 1}
data = []
with tmp(download(MAGA, cached=True)) as f:
    for r in csv(f.name)[1:][:1000]:
        if r[8].startswith(('left', 'right')):
            data.append((v(r[2]), r[8])) # (vector, label)

t = time.time()

print kfoldcv(RandomForest, data, m=10, k=3)
#f = RandomForest(data, m=10)
#f.save(open(cd('forest.model'), 'w'))
#f = RandomForest.load(open(cd('forest.model')))
#print f.predict(v('#maga'))
print time.time() - t
print xxx
