from model import normalize_text

def test_url_cleaning():
    assert "URL" in normalize_text("see https://example.com now")

def test_whitespace_cleaning():
    assert normalize_text("you   are\nawful") == "you are awful"

def test_misspelling_preserved_for_char_ngrams():
    assert "stooopid" in normalize_text("you are stooopid")
