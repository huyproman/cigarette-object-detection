"""Per-seed complete B0 nc=1 initial state: COCO-pretrained yolo26s transfer + seeded new single-class head."""
import hashlib, io, json, sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make(seed, pretrained, out_dir):
    torch.manual_seed(seed)
    m = DetectionModel("yolo26s.yaml", nc=1, verbose=False)          # random init under seed (incl. new 1-class head)
    src = YOLO(str(pretrained)).model.float().state_dict()
    own = m.state_dict()
    xfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    m.load_state_dict(xfer, strict=False)
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    path = Path(out_dir) / f"B0_seed{seed}_init_state.pt"
    torch.save(sd, path)
    return dict(seed=seed, path=str(path), n_tensors=len(sd), n_transferred=len(xfer), n_new=len(sd) - len(xfer),
                new_tensor_names=sorted(k for k in sd if k not in xfer), state_sha256=state_hash(sd),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pretrained=str(pretrained),
                pretrained_sha256=hashlib.sha256(Path(pretrained).read_bytes()).hexdigest())


if __name__ == "__main__":
    recs = [make(s, sys.argv[1], sys.argv[2]) for s in (101, 202, 303)]
    Path(sys.argv[3]).write_text(json.dumps(recs, indent=2))
    for r in recs:
        print(r["seed"], r["n_tensors"], r["n_transferred"], r["n_new"], r["state_sha256"][:16])
