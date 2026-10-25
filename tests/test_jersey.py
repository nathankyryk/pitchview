from pitchview.jersey import parse_number, shirt_region, vote


def test_shirt_region():
    assert shirt_region((100, 200, 140, 300)) == (104, 215, 136, 260)


def test_parse_number():
    assert parse_number("10") == "10"
    assert parse_number(" 7.") == "7"
    assert parse_number("abc") is None
    assert parse_number("1234") is None


def test_vote_weighs_confident_reads_and_abstains():
    reads = [
        ("a", "10", 0.9), ("a", "18", 0.6), ("a", "18", 0.55), ("a", "70", 0.3),  # 10: 0.9 < 18: 1.15
        ("b", "7", 0.4), ("b", None, 0.0),  # nothing confident
        ("c", "23", 0.95),
    ]  # fmt: skip
    assert vote(reads) == {"a": "18", "b": None, "c": "23"}
    assert vote(reads, min_conf=0.7) == {"a": "10", "b": None, "c": "23"}
