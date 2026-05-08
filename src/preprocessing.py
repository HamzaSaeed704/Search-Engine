"""Text preprocessing for the movie search engine.

Pipeline: lowercase -> strip punctuation -> regex tokenize -> remove stopwords -> Porter stem.

Uses a hardcoded English stopwords list and a regex tokenizer to avoid the
NLTK data-download dependency. Porter stemmer comes from NLTK and works
standalone.
"""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
})

STEMMER = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=8192)
def _stem(token: str) -> str:
    return STEMMER.stem(token)


def preprocess(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [
        _stem(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 1 and not t.isdigit()
    ]


def query_terms(query: str) -> list[str]:
    """Raw lowercased query tokens (pre-stemming) for snippet highlighting."""
    tokens = _TOKEN_RE.findall(query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
