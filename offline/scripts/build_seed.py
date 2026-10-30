"""Build (or confirm) the seed symptom index for the shipped books.

    python offline/scripts/build_seed.py MODELS_FOLDER

Embeds every symptom of the shipped books with the bundled ONNX model and writes
``emb_seed_<model>.npz`` into MODELS_FOLDER, where the app (run from source), the tests and the
installer build all pick it up, so the books are embedded once instead of at every step.
Does nothing if the seed there already holds exactly the current books' symptoms.
"""
import sys
import time
from pathlib import Path

OFFLINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFLINE))


def shipped_symptom_keys() -> list[str]:
    """The symptoms the installer's seed covers (same as packaging/build_app.py)."""
    from horus.books import discover_books
    from horus.config import Paths
    from horus.dataset import load_base_dataset
    repo = OFFLINE.parent
    paths = Paths(resources=repo, user=repo / "build" / "seed-user")
    return sorted(load_base_dataset(paths, discover_books(paths)).s2r)


def main() -> int:
    models = Path(sys.argv[1] if len(sys.argv) > 1 else OFFLINE.parent / "models").resolve()
    from horus.config import EMBEDDING_MODEL_DIRNAME
    from horus.embeddings import OnnxEmbedder, seed_matches, store_tag, write_seed
    keys = shipped_symptom_keys()
    embedder = OnnxEmbedder(models / EMBEDDING_MODEL_DIRNAME)
    path = models / f"emb_seed_{store_tag(embedder.model_id)}.npz"
    if seed_matches(path, keys):
        print(f"Seed index up to date: {len(keys):,} symptoms in {path.name}")
        return 0
    t = time.time()
    write_seed(embedder, keys, models)
    print(f"Seed index: {len(keys):,} symptoms in {time.time() - t:.0f}s → {path.name} "
          f"({path.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
