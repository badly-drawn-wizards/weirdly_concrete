#!/usr/bin/env python3

from itertools import chain, count, product
from math import atan2, pi as PI
from statistics import mean
from random import choice
from sys import exit, argv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def angleDiff(t1, t2):
    if t2 < t1:
        return 2*PI + t2 - t1
    else:
        return t2 - t1

def arg(p):
    # We want (0,0) to have distinct value, but still be comparable to other angles. Hence -inf instead of None.
    # This should not matter when calculating the angle of a directed line segment between distinct points.
    if p == (0,0):
        return -float('inf')
    return atan2(*p)
def lengthsq(p):
    return sum(map(lambda x: x**2, p))
def distsq(p, q):
    return lengthsq(diff(p, q))
def polar(p):
    return (arg(p), lengthsq(p))
def add(p, q):
    return tuple(map(lambda pi, qi: pi + qi, p, q))
def diff(p, q):
    return tuple(map(lambda pi, qi: pi - qi, p, q))
def cmp(x, y):
    return int(y < x) - int(x < y)
def angle(p0, p1, p2):
    return angleDiff(arg(diff(p0, p1)), arg(diff(p2, p1))) - PI
def ccw(p0, p1, p2):
    return cmp(angle(p0,p1,p2), 0)


# Will yield the last duplicate
def dedupBy(xs, key):
    xs1 = iter(xs)
    xs2 = iter(xs)
    try:
        next(xs2)
    except StopIteration:
        return
    for x2 in xs2:
        x1 = next(xs1)
        if key(x1) == key(x2):
            continue
        yield x1
    yield next(xs1)

def convex_hull(ps):
    stack = []
    p0 = min(ps)
    # This will get (relative to P0) the longest points for each angle sorted by angle
    ps = dedupBy(sorted(ps, key=lambda p: polar(diff(p, p0))), key=lambda p: arg(diff(p, p0)))
    for p in ps:
        while len(stack) > 1 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    return stack

def idx(ps, i):
    return ps[i % len(ps)]

def rot(ps, i, j):
    a1 = arg(diff(idx(ps, i+1), idx(ps, i)))
    a2 = arg(diff(idx(ps, j+1), idx(ps, j)))
    return angleDiff(a1, a2)

def antipodal_pairs(ch):
    j = 1
    for i in range(len(ch)):
        while rot(ch, i, i+j) < PI:
            j += 1
        yield (idx(ch,i), idx(ch, i+j))
        j = max(1, j-1)

def longest_line(ch):
    ls = antipodal_pairs(ch)
    return sorted(max(ls, key=lambda l: distsq(*l)))

# For testing antipodal_pairs
def longest_line2(ch):
    ls = product(ch, ch)
    return sorted(max(ls, key=lambda l: distsq(*l)))



class Rejection(Exception):
    pass

cardinals = [(0,1), (1,0), (0,-1), (-1, 0)]
def myopic_walk(n):
    w = [(0,0)]
    for _ in range(n):
        dirs = list(filter(lambda q: q not in w, map(lambda d: add(w[-1], d), cardinals)))
        if dirs:
            w.append(choice(dirs))
        else:
            raise Rejection()
    return w

def saw(w):
    return len(w) == len(set(w))

def catwalk(w1, w2):
    return w1 + [add(w1[-1], p) for p in w2[1:]]

def dimer(n):
    if n <= 3:
        return myopic_walk(n)
    else:
        while True:
            w1 = dimer(n//2)
            w2 = dimer(n-n//2)
            w = catwalk(w1, w2)
            if saw(w):
                return w

def marbles(n, w):
    return w[::n][1:]

def meeting_marbles(l, m):
    def between(p):
        return p in l or ccw(l[0], p, l[1]) == 0
    return list(filter(between, m))

def ive_lost_my_marbles(k, n):
    w = dimer(k*n)
    m = marbles(n, w)
    ch = convex_hull(m)
    l = longest_line(ch)
    mm = meeting_marbles(l, m)
    return (w, m, ch, l, mm)

def plot_walk(w, m, ch, l, mm):
    (wxs, wys) = zip(*w)
    (lxs, lys) = zip(*l)
    (mxs, mys) = zip(*m)
    (mmxs, mmys) = zip(*mm) if mm else ([], [])

    fig, ax = plt.subplots()

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid()

    ax.axis('equal')

    ax.plot(wxs, wys, color='grey')
    ax.scatter(mxs, mys, color='black')
    ax.scatter(mmxs, mmys, color='red')

    ax.plot(lxs, lys, color='red')

    fig.show()
    plt.show()

NUM_SAMPLES = 1000
MARBLES = 50
SPACING = 2

def plot_main():
    while True:
        (w, m, ch, l, mm) = ive_lost_my_marbles(MARBLES, SPACING)
        plot_walk(w, m, ch, l, mm)

def stat_main():
    total = 0
    totalsq = 0
    n = 0
    try:
        for i in range(NUM_SAMPLES):
            (w, m, ch, l, mm) = ive_lost_my_marbles(MARBLES, SPACING)
            print("Sample", str(i+1) + ":", len(mm), "many marbles meeting the maximal measure line")
            total += len(mm)
            totalsq += len(mm)**2
            n += 1
    except KeyboardInterrupt:
        exit("Quitting")
    finally:
        mean = total / n
        meansq = totalsq / n
        sqmean = mean ** 2
        stdev = np.sqrt(meansq - sqmean)
        # Print even if a keyboard interrupt occurs
        print("Marbles (k):", MARBLES)
        print("Spacing (n):", SPACING)
        print("Mean:", mean)
        print("Stdev:", stdev)

if __name__ == '__main__':
    mode = 'plot'
    if len(argv) > 1:
        mode = argv[1]
    if mode == 'plot':
        plot_main()
    elif mode == 'stat':
        stat_main()
    else:
        print("Invalid argument. Must be 'plot' or 'stat")
