"""Watch the already-authorized C3 queue and generate its fixed-protocol report when complete."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parent / "MT_DS_PL_C3"


def write_status(data):
    data["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    temporary = PROJECT / "queue_status.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(PROJECT / "queue_status.json")


def main():
    control = json.loads((PROJECT / "queue_control.json").read_text())
    while True:
        files = sorted((PROJECT / "Results_transfer/launch_logs").glob("*/manifest.json"))
        if not files:
            raise RuntimeError("No C3 queue manifest")
        path = files[-1]
        jobs = json.loads(path.read_text())["jobs"]
        counts = {key: sum(j["status"] == key for j in jobs) for key in ("pending", "running", "finished", "failed")}
        status = dict(manifest=str(path), counts=counts, phase="running")
        if counts["failed"]:
            status["phase"] = "failed"
            write_status(status)
            raise SystemExit("Training failed; inspect queue manifest and run log")
        if counts["finished"] == len(jobs) == 24:
            result = subprocess.run([sys.executable, str(PROJECT / "scripts/report_transfer.py")],
                                    capture_output=True, text=True)
            status.update(phase="reported" if result.returncode == 0 else "report_failed",
                          report=str(PROJECT / "summary/transfer_report.md"),
                          report_stdout=result.stdout, report_stderr=result.stderr)
            write_status(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            raise SystemExit(result.returncode)
        try:
            os.kill(control["pid"], 0)
        except ProcessLookupError:
            status["phase"] = "queue_stopped_before_completion"
            write_status(status)
            raise SystemExit("C3 queue ended before all jobs completed")
        write_status(status)
        time.sleep(30)


if __name__ == "__main__":
    main()
