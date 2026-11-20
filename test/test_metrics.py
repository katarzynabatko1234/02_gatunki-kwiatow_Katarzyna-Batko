import torch

from metrics import TopKCounter, topk_accuracy


def brute_force_topk(logits, targets, k):
    """Niezależna referencja: dla każdego przykładu liczymy, ile klas ma większy logit niż prawdziwa."""
    hits = 0
    for row, t in zip(logits.tolist(), targets.tolist()):
        rank = sum(1 for v in row if v > row[t])
        hits += rank < k
    return hits / len(targets)


def test_manual_example():
    logits = torch.tensor([[0.1, 0.9, 0.0, 0.0],
                           [0.5, 0.3, 0.2, 0.0],
                           [0.0, 0.1, 0.2, 0.7],
                           [0.2, 0.3, 0.4, 0.1]])
    targets = torch.tensor([1, 2, 0, 1])
    c = TopKCounter(ks=(1, 2, 3))
    c.update(logits, targets)
    assert c.compute() == {1: 0.25, 2: 0.5, 3: 0.75}


def test_batch_accumulation_equals_single_batch():
    g = torch.Generator().manual_seed(6102)
    logits, targets = torch.randn(100, 102, generator=g), torch.randint(0, 102, (100,), generator=g)
    whole = TopKCounter(ks=(1, 3, 5))
    whole.update(logits, targets)
    parts = TopKCounter(ks=(1, 3, 5))
    for i in range(0, 100, 7):
        parts.update(logits[i:i + 7], targets[i:i + 7])
    assert whole.compute() == parts.compute()
    assert parts.total == 100


def test_matches_brute_force():
    g = torch.Generator().manual_seed(1)
    logits, targets = torch.randn(500, 20, generator=g), torch.randint(0, 20, (500,), generator=g)
    for k in (1, 3, 5):
        assert abs(topk_accuracy(logits, targets, k) - brute_force_topk(logits, targets, k)) < 1e-12


def test_k_equal_num_classes_is_one():
    logits, targets = torch.randn(10, 4), torch.randint(0, 4, (10,))
    assert topk_accuracy(logits, targets, 4) == 1.0


def test_reset():
    c = TopKCounter(ks=(1,))
    c.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    c.reset()
    assert c.total == 0 and c.compute() == {1: 0.0}
