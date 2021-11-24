import math
import collections
from flowtext.data.utils import ngrams_iterator

import oneflow as flow

def _compute_ngram_counter(tokens, max_n):
    """ Create a Counter with a count of unique n-grams in the tokens list

    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from flowtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['name', 'is', 'name', 'is', 'oneflow']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({
                ('name',): 2, 
                ('is',): 2, 
                ('name', 'is'): 2, 
                ('oneflow',): 1, 
                ('is', 'name'): 1, 
                ('is', 'oneflow'): 1
                    })
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(
        tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n)
    )

    return ngrams_counter


def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from flowtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'oneflow', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'oneflow', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(
        references_corpus
    ), "The length of candidate and reference corpus should be the same"

    clipped_counts = flow.zeros(max_n)
    total_counts = flow.zeros(max_n)
    weights = flow.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * flow.log(pn)
        score = flow.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()
