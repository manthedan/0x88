from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_PROFILE = "tiny-leela"
DEFAULT_REGION = "us-west-2"


@dataclass
class Aws:
    profile: str = DEFAULT_PROFILE
    region: str = DEFAULT_REGION

    @classmethod
    def from_env(cls, profile: str | None = None, region: str | None = None) -> "Aws":
        return cls(
            profile=profile or os.environ.get("AWS_PROFILE", DEFAULT_PROFILE),
            region=region or os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION),
        )

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["AWS_PROFILE"] = self.profile
        env["AWS_DEFAULT_REGION"] = self.region
        return env

    def run(self, args: list[str], *, json_out: bool = False, check: bool = True) -> Any:
        cmd = ["aws", *args, "--region", self.region]
        if json_out and "--output" not in args:
            cmd.extend(["--output", "json"])
        proc = subprocess.run(cmd, env=self.env(), text=True, capture_output=True)
        if check and proc.returncode != 0:
            raise RuntimeError(f"aws command failed rc={proc.returncode}: {' '.join(cmd)}\n{proc.stderr.strip()}")
        if json_out:
            if not proc.stdout.strip():
                return None
            return json.loads(proc.stdout)
        return proc.stdout

    def batch_list_jobs(self, queue: str, status: str) -> list[dict[str, Any]]:
        data = self.run(
            ["batch", "list-jobs", "--job-queue", queue, "--job-status", status],
            json_out=True,
        )
        return data.get("jobSummaryList", []) if data else []

    def batch_describe_job(self, job_id: str) -> dict[str, Any] | None:
        data = self.run(["batch", "describe-jobs", "--jobs", job_id], json_out=True)
        jobs = data.get("jobs", []) if data else []
        return jobs[0] if jobs else None

    def batch_wait_terminal(self, job_id: str, interval: int = 60, max_seconds: int | None = None) -> dict[str, Any]:
        started = time.time()
        while True:
            job = self.batch_describe_job(job_id)
            if not job:
                raise RuntimeError(f"job not found: {job_id}")
            status = job.get("status")
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job_id} {status}", flush=True)
            if status in {"SUCCEEDED", "FAILED"}:
                return job
            if max_seconds is not None and time.time() - started > max_seconds:
                return job
            time.sleep(interval)

    def cloudwatch_tail_for_job(self, job: dict[str, Any], lines: int = 80) -> str:
        attempts = job.get("attempts") or []
        stream = None
        for attempt in reversed(attempts):
            stream = ((attempt.get("container") or {}).get("logStreamName"))
            if stream:
                break
        if not stream:
            stream = ((job.get("container") or {}).get("logStreamName"))
        if not stream:
            return ""
        try:
            data = self.run(
                [
                    "logs",
                    "get-log-events",
                    "--log-group-name",
                    "/aws/batch/job",
                    "--log-stream-name",
                    stream,
                    "--limit",
                    str(lines),
                ],
                json_out=True,
            )
        except Exception as exc:
            return f"<unable to read CloudWatch log {stream}: {exc}>"
        events = data.get("events", []) if data else []
        return "\n".join(str(e.get("message", "")) for e in events)


def human_ms(ms: int | None) -> str:
    if not ms:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ms / 1000))
