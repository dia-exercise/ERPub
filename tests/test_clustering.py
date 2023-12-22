from erpub.pipeline.clustering import connected_components_


def test_connected_components():
    clusters = connected_components_(
        pairs=[(0, 1), (1, 2), (0, 2)],
        sims=[0.8, 0.4, 0.4],
        n=3,
        threshold=0.6,
    )
    clusters_lst = [list(elem) for elem in clusters]
    assert len(clusters) == 2
    assert [2] in clusters_lst and [0, 1] in clusters_lst
