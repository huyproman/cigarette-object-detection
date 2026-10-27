"""Three-profile registry: resolves profile -> checkpoint file, verifies SHA256 and size before any load (fail closed)."""
import hashlib, json, os
from pathlib import Path

PROFILES = ("b0", "e2_carafe", "e3_combined")
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs/registry/model_registry.json"


class InferenceError(Exception):
    """Deterministic, user-facing error (bad checkpoint, device, precision or image)."""


def load_registry(path=REGISTRY_PATH):
    return json.loads(Path(path).read_text())


def get_profile(profile_id, registry=None):
    reg = registry or load_registry()
    for p in reg["profiles"]:
        if p["profile_id"] == profile_id:
            return p
    raise InferenceError(f"unknown profile '{profile_id}'; valid profiles: {', '.join(PROFILES)}")


def weights_dir():
    return Path(os.environ.get("P05_WEIGHTS_DIR") or Path.cwd() / "weights")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_checkpoint(profile_id, checkpoint_override=None, registry=None, verify_profile_hash=True):
    """Return (path, profile_dict). Overrides are hash-verified against the registry if verify_profile_hash=True."""
    prof = get_profile(profile_id, registry)
    path = Path(checkpoint_override) if checkpoint_override else weights_dir() / prof["checkpoint_file"]
    if not path.is_file() or path.stat().st_size == 0:
        raise InferenceError(f"checkpoint for '{profile_id}' not found at {path}. Place the authorized file '{prof['checkpoint_file']}' "
                             f"in the weights directory (env P05_WEIGHTS_DIR, default weights) and verify: sha256 = {prof['checkpoint_sha256']}")
    if verify_profile_hash:
        if path.stat().st_size != prof["checkpoint_bytes"]:
            raise InferenceError(f"checkpoint size mismatch for '{profile_id}': {path.stat().st_size} != {prof['checkpoint_bytes']}")
        got = sha256_file(path)
        if got != prof["checkpoint_sha256"]:
            raise InferenceError(f"checkpoint SHA256 mismatch for '{profile_id}': got {got}, expected {prof['checkpoint_sha256']}")
    else:
        prof = dict(prof)
        prof["checkpoint_bytes"] = path.stat().st_size
        prof["checkpoint_sha256"] = sha256_file(path)
        prof["checkpoint_file"] = path.name
    return path, prof
