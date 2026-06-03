from __future__ import annotations

# a small, bundled english stopword list. shipping it (rather than downloading nltk data)
# keeps the default preprocessing path offline and deterministic, which matters for
# reproducible experiments. swap in a richer list per-experiment when needed.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were",
        "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has", "had",
        "having", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
        "those", "of", "to", "in", "on", "for", "with", "as", "at", "by", "from", "up",
        "down", "out", "over", "under", "again", "then", "once", "here", "there", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
        "just", "should", "now", "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "off", "because", "until", "what", "which", "who",
        "whom", "when", "where", "why", "how", "would", "could", "also",
    }
)
