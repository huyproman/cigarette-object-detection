"""Real ultralytics trainer, constant fitness => latest tied epoch must own best.pt (pinned 8.4.174 semantics)."""
import numpy as np, pytest, torch
from PIL import Image
from ultralytics.models.yolo.detect import DetectionTrainer


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("tie"); (d / "images").mkdir(); (d / "labels").mkdir()
    g = np.random.default_rng(1)
    for i in range(8):
        Image.fromarray(g.integers(0, 255, (160, 160, 3), dtype=np.uint8)).save(d / "images" / f"{i}.jpg")
        (d / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (d / "l.txt").write_text("\n".join(str(d / "images" / f"{i}.jpg") for i in range(8)))
    (d / "data.yaml").write_text(f"path: {d}\ntrain: l.txt\nval: l.txt\nnames:\n  0: cigarette\n")
    return d


class TieTrainer(DetectionTrainer):  # noqa
    scores = [0.54, 0.54, 0.53]
    owner = []

    def save_model(self):
        r = super().save_model()
        self.owner.append(torch.load(self.wdir / "best.pt", map_location="cpu", weights_only=False)["epoch"])  # before final strip
        return r

    def validate(self):
        f = self.scores[self.epoch]
        if self.best_fitness is None or self.best_fitness < f:   # same bookkeeping as upstream validate()
            self.best_fitness = f
        return {"metrics/mAP50-95(B)": f, "fitness": f}, f


def test_latest_tied_epoch_overwrites_best(fx):
    t = TieTrainer(overrides=dict(model="yolo26s.yaml", data=str(fx / "data.yaml"), epochs=3, imgsz=64, batch=4, workers=0,
                                  device="cpu", project=str(fx / "runs"), name="t", plots=False, amp=False, seed=1, val=True, optimizer="SGD", lr0=0.01))
    t.train()
    best = torch.load(t.wdir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(t.wdir / "last.pt", map_location="cpu", weights_only=False)
    assert TieTrainer.owner == [0, 1, 1]      # epoch idx owning best.pt after each save: tie overwrites, lower score does not
    assert last["epoch"] == -1 and best["epoch"] == -1   # upstream strips both at the end; ownership recorded above
