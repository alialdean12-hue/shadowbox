from __future__ import annotations
import json, pickle, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import analyze_image


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main(argv):
    if len(argv) != 3:
        raise SystemExit("Usage: analyze_worker.py INPUT OUTPUT")
    with Path(argv[1]).open("rb") as handle:
        payload = pickle.load(handle)
    def progress(message, current, total):
        emit({"type":"progress","message":message,"current":current,"total":total})
    project = analyze_image(
        payload["source_path"],
        payload["page_width_mm"],
        payload["page_height_mm"],
        payload.get("color_count", 6),
        payload.get("margin_mm", 8.0),
        payload.get("max_dimension", 1200),
        progress,
    )
    with Path(argv[2]).open("wb") as handle:
        pickle.dump(project, handle, protocol=5)
    emit({"type":"finished","layers":len(project["layers"])})


if __name__ == "__main__":
    try:
        main(sys.argv)
    except BaseException as exc:
        emit({"type":"error","message":str(exc),"traceback":traceback.format_exc()})
        raise
