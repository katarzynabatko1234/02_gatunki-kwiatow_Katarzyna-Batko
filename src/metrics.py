"""Metryki. topk_accuracy jest zaimplementowane ręcznie (wymóg projektu)."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


class TopKCounter:
    """Własnoręczny licznik top-k accuracy, akumulowany po batchach.

    Dla każdego przykładu sprawdzamy, czy prawdziwa klasa jest wśród k klas
    o największym logicie. Bez użycia gotowych funkcji metryk.
    """

    def __init__(self, ks=(1, 3)):
        self.ks = tuple(ks)
        self.reset()

    def reset(self):
        self.hits = {k: 0 for k in self.ks}
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits = logits.detach().float().cpu()
        targets = targets.detach().cpu()
        max_k = max(self.ks)
        # indeksy klas posortowane malejąco wg logitu: argsort(-x) == ranking
        ranked = torch.argsort(logits, dim=1, descending=True)[:, :max_k]
        correct = ranked == targets.unsqueeze(1)  # [B, max_k] bool, max. jedno True w wierszu
        for k in self.ks:
            self.hits[k] += int(correct[:, :k].any(dim=1).sum())
        self.total += targets.shape[0]

    def compute(self) -> dict:
        return {k: self.hits[k] / self.total if self.total else 0.0 for k in self.ks}


def topk_accuracy(logits, targets, k: int) -> float:
    c = TopKCounter(ks=(k,))
    c.update(torch.as_tensor(logits), torch.as_tensor(targets))
    return c.compute()[k]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


if __name__ == "__main__":
    # Ręczny test: 4 przykłady, 4 klasy
    logits = torch.tensor([
        [0.1, 0.9, 0.0, 0.0],  # cel 1 -> top1 trafione
        [0.5, 0.3, 0.2, 0.0],  # cel 2 -> 3. miejsce: top3 tak, top1 nie
        [0.0, 0.1, 0.2, 0.7],  # cel 0 -> ostatnie: nic
        [0.2, 0.3, 0.4, 0.1],  # cel 1 -> 2. miejsce: top3 tak
    ])
    targets = torch.tensor([1, 2, 0, 1])
    c = TopKCounter(ks=(1, 2, 3))
    c.update(logits[:2], targets[:2])
    c.update(logits[2:], targets[2:])
    res = c.compute()
    assert res == {1: 0.25, 2: 0.5, 3: 0.75}, res
    print("TopKCounter OK:", res)
