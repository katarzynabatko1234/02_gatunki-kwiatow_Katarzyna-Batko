import config as C
from conftest import needs_data
from sklearn.model_selection import train_test_split

pytestmark = needs_data


def test_split_sizes(df):
    assert len(df) == 8189
    assert df.split.value_counts().to_dict() == {"train": 6551, "val": 819, "test": 819}


def test_split_disjoint_and_all_classes(df):
    parts = {s: set(df[df.split == s].file) for s in ("train", "val", "test")}
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
    for s in parts:
        assert df[df.split == s].label.nunique() == C.NUM_CLASSES


def test_split_reproducible_with_seed(df):
    idx = list(range(len(df)))
    tr, rest = train_test_split(idx, test_size=0.2, stratify=df.label, random_state=C.SEED)
    va, te = train_test_split(rest, test_size=0.5, stratify=df.label.iloc[rest], random_state=C.SEED)
    assert set(df.iloc[tr].file) == set(df[df.split == "train"].file)
    assert set(df.iloc[va].file) == set(df[df.split == "val"].file)
    assert set(df.iloc[te].file) == set(df[df.split == "test"].file)


def test_dataset_item(df):
    from data import FlowersDataset
    for train in (True, False):
        x, y = FlowersDataset(df.head(3), train=train)[0]
        assert tuple(x.shape) == (3, C.IMG_SIZE, C.IMG_SIZE)
        assert isinstance(y, int) and 0 <= y < C.NUM_CLASSES
