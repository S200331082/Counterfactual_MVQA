# -*- coding: utf-8 -*-
# Natural Language Toolkit: BLEU Score
#
# Copyright (C) 2001-2015 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# Contributors: Dmitrijs Milajevs
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
"""BLEU score implementation."""

from __future__ import division

import math
import nltk


from nltk import Counter
from nltk import ngrams


def bleu(candidate, references, weights):

    p_ns = (
        _modified_precision(candidate, references, i)
        for i, _ in enumerate(weights, start=1)
    )

    try:
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
    except ValueError:
        # some p_ns is 0
        return 0

    # bp = _brevity_penalty(candidate, references)
    # return bp * math.exp(s)
    return math.exp(s)


def _modified_precision(candidate, references, n):


    candidates = []
    for i in range(len(candidate)):
        candidates.append(candidate[i].strip('#'))
    counts = Counter(ngrams(candidates, n))

    if not counts:
        return 0

    max_counts = {}
    new_references = []
    for i in range(len(references)):
        new_references.append(references[i].strip('#'))
    reference_counts = Counter(ngrams(new_references, n))
    for ngram in counts:
        max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

    return sum(clipped_counts.values()) / sum(counts.values())


def _brevity_penalty(candidate, references):

    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)



