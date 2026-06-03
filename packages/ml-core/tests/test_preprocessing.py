from ml_core.preprocessing import (
    PreprocessingConfig,
    clean_text,
    compose_pipeline,
    normalize_text,
    tokenize_text,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World!") == "hello world"


def test_clean_text_respects_toggles():
    assert clean_text("Hello, World!", lowercase=False, remove_punctuation=False) == "Hello, World!"


def test_normalize_collapses_whitespace():
    assert normalize_text("  a   b\tc\n ") == "a b c"


def test_tokenize_splits_words():
    assert tokenize_text("one two-three four") == ["one", "two", "three", "four"]


def test_pipeline_default_is_clean_lowercase_string():
    pipe = compose_pipeline()
    assert pipe.run("The Quick, Brown FOX!") == "the quick brown fox"


def test_pipeline_removes_stopwords_when_enabled():
    pipe = compose_pipeline(PreprocessingConfig(remove_stopwords=True))
    out = pipe.run("this is the best of the samples")
    assert "the" not in out.split()
    assert "best" in out.split()


def test_pipeline_run_many_matches_run():
    pipe = compose_pipeline()
    texts = ["A B!", "c, d."]
    assert pipe.run_many(texts) == [pipe.run(t) for t in texts]
