from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path


def _append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def run_worker(mode: str, input_path: str, output_path: str, progress_path: str) -> int:
    progress_file = Path(progress_path)
    try:
        with Path(input_path).open("rb") as handle:
            payload = pickle.load(handle)

        def progress(message, current, total):
            _append(
                progress_file,
                {
                    "type": "progress",
                    "message": str(message),
                    "current": int(current),
                    "total": int(total),
                },
            )

        if mode == "--analyze-worker":
            from core.engine import analyze_image

            project = analyze_image(
                payload["source_path"],
                payload["page_width_mm"],
                payload["page_height_mm"],
                payload.get("color_count", 6),
                payload.get("margin_mm", 8.0),
                payload.get("max_dimension", 1200),
                progress,
            )
            result = project
            finished = {"type": "finished", "layers": len(project["layers"])}
        elif mode == "--export-worker":
            from core.exporter import export_package

            root = export_package(
                payload["project"],
                payload["destination"],
                payload["project_name"],
                payload.get("source_path"),
                progress,
            )
            result = {"export_root": str(root)}
            finished = {"type": "finished", "export_root": str(root)}
        else:
            raise ValueError(f"Unknown worker mode: {mode}")

        with Path(output_path).open("wb") as handle:
            pickle.dump(result, handle, protocol=5)
        _append(progress_file, finished)
        return 0
    except BaseException as exc:
        _append(
            progress_file,
            {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
