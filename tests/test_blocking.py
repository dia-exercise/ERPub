from erpub.pipeline.blocking import naive_all_pairs


def test_length_naive_all_pairs():
    data = range(100)
    assert len(naive_all_pairs(data)) == 100 * (100 - 1) / 2
