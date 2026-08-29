from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] in {"--analyze-worker", "--export-worker"}:
        from worker_runtime import run_worker

        return run_worker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    from app import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
