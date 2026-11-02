from darija_asr.agreement import pair_distance, route


def test_identical_sources_are_accepted():
    decision, stats = route({"a": "واش نتا مزيان", "b": "واش نتا مزيان", "c": "واش نتا مزيان"}, 0.05, 0.6)
    assert decision == "accept"
    assert stats["max_cer"] == 0


def test_one_outlier_goes_to_llm():
    decision, _ = route({"a": "واش نتا مزيان", "b": "واش نتا مزيان", "c": "واش انت بخير"}, 0.05, 0.6)
    assert decision == "llm"


def test_all_pairs_far_apart_are_dropped():
    decision, _ = route({"a": "واش نتا مزيان", "b": "bonjour tout le monde", "c": "1 2 3 4 5 6"}, 0.05, 0.6)
    assert decision == "drop"


def test_missing_source_never_auto_accepts():
    decision, stats = route({"a": "واش نتا مزيان", "b": "واش نتا مزيان", "c": ""}, 0.05, 0.6)
    assert decision == "llm"
    assert stats["n_sources"] == 2


def test_fewer_than_two_sources_dropped():
    assert route({"a": "واش", "b": ""}, 0.05, 0.6)[0] == "drop"


def test_pair_distance_is_symmetric():
    assert pair_distance("abc def", "abd") == pair_distance("abd", "abc def")
