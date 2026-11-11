from __future__ import annotations
import json, pickle, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.exporter import export_package


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main(argv):
    if len(argv) != 3:
        raise SystemExit("Usage: export_worker.py INPUT OUTPUT")
    with Path(argv[1]).open("rb") as handle:
        payload = pickle.load(handle)
    def progress(message, current, total):
        emit({"type":"progress","message":message,"current":current,"total":total})
    root = export_package(
        payload["project"],
        payload["destination"],
        payload["project_name"],
        payload.get("source_path"),
        progress,
    )
    with Path(argv[2]).open("wb") as handle:
        pickle.dump({"export_root": str(root)}, handle, protocol=5)
    emit({"type":"finished","export_root":str(root)})


if __name__ == "__main__":
    try:
        main(sys.argv)
    except BaseException as exc:
        emit({"type":"error","message":str(exc),"traceback":traceback.format_exc()})
        raise
