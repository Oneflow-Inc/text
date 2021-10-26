import importlib
from .ag_news import AG_NEWS
from .amazonreviewfull import AmazonReviewFull
from .amazonreviewpolarity import AmazonReviewPolarity
from .conll2000chunking import CoNLL2000Chunking
from .dbpedia import DBpedia
from .enwik9 import EnWik9
from .imdb import IMDB
from .iwslt2016 import IWSLT2016
from .iwslt2017 import IWSLT2017
from .multi30k import Multi30k

DATASETS = {
    'AG_NEWS': AG_NEWS,
    'AmazonReviewFull': AmazonReviewFull,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'CoNLL2000Chunking': CoNLL2000Chunking,
    'DBpedia': DBpedia,
    'EnWik9': EnWik9,
    'IMDB': IMDB,
    'IWSLT2016': IWSLT2016,
    'IWSLT2017': IWSLT2017,
    'Multi30k': Multi30k
}
