from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import cold_store, inventory
from .aws_cli import Aws, human_ms
from .paths import now_iso, rel, repo_root
from .registry import RunEvent, RunRegistry, default_run_id, git_sha
from .state import PipelineState

JOB_STATUSES = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_run_record(args: argparse.Namespace) -> int:
    reg = RunRegistry(args.registry)
    run_id = args.run_id or default_run_id(args.kind, args.name)
    attrs: dict[str, Any] = {
        "kind": args.kind,
        "name": args.name or run_id,
        "status": args.status,
        "git_sha": git_sha(),
    }
    for item in args.attr or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"bad --attr {item!r}; expected key=value")
        attrs[key] = value
    reg.append(RunEvent(run_id=run_id, event=args.event, attrs=attrs))
    print(run_id)
    return 0


def cmd_run_update(args: argparse.Namespace) -> int:
    reg = RunRegistry(args.registry)
    attrs: dict[str, Any] = {}
    if args.status:
        attrs["status"] = args.status
    for item in args.attr or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"bad --attr {item!r}; expected key=value")
        attrs[key] = value
    reg.append(RunEvent(run_id=args.run_id, event=args.event, attrs=attrs))
    return 0


def cmd_run_list(args: argparse.Namespace) -> int:
    reg = RunRegistry(args.registry)
    rows = reg.list(kind=args.kind, status=args.status, active=args.active)
    if args.json:
        print_json(rows)
        return 0
    if not rows:
        print("no runs")
        return 0
    for r in rows:
        print(
            f"{r.get('updated_at','')}\t{r.get('status','')}\t{r.get('kind','')}\t"
            f"{r.get('run_id','')}\t{r.get('name','')}\t{r.get('aws_job_id','')}"
        )
    return 0


def cmd_run_show(args: argparse.Namespace) -> int:
    run = RunRegistry(args.registry).get(args.run_id)
    if not run:
        raise SystemExit(f"run not found: {args.run_id}")
    print_json(run)
    return 0


def cmd_phase_mark(args: argparse.Namespace) -> int:
    path = PipelineState(args.root).mark(args.phase, args.state, note=args.note)
    print(path)
    return 0


def cmd_phase_status(args: argparse.Namespace) -> int:
    print_json(PipelineState(args.root).status())
    return 0


def cmd_cloud_jobs(args: argparse.Namespace) -> int:
    aws = Aws.from_env(args.profile, args.region)
    rows = []
    statuses = args.statuses or JOB_STATUSES
    for status in statuses:
        for job in aws.batch_list_jobs(args.queue, status):
            name = job.get("jobName", "")
            if args.match and args.match not in name and args.match not in job.get("jobId", ""):
                continue
            rows.append(
                {
                    "status": job.get("status", status),
                    "jobName": name,
                    "jobId": job.get("jobId"),
                    "createdAt": human_ms(job.get("createdAt")),
                }
            )
    if args.json:
        print_json(rows)
    else:
        for r in rows:
            print(f"{r['status']}\t{r['createdAt']}\t{r['jobName']}\t{r['jobId']}")
    return 0


def cmd_cloud_describe(args: argparse.Namespace) -> int:
    aws = Aws.from_env(args.profile, args.region)
    job = aws.batch_describe_job(args.job_id)
    if not job:
        raise SystemExit(f"job not found: {args.job_id}")
    if args.json:
        print_json(job)
    else:
        print(f"jobId: {job.get('jobId')}")
        print(f"jobName: {job.get('jobName')}")
        print(f"status: {job.get('status')}")
        print(f"createdAt: {human_ms(job.get('createdAt'))}")
        print(f"startedAt: {human_ms(job.get('startedAt'))}")
        print(f"stoppedAt: {human_ms(job.get('stoppedAt'))}")
        if job.get("statusReason"):
            print(f"statusReason: {job.get('statusReason')}")
        cont = job.get("container") or {}
        if cont.get("reason"):
            print(f"container.reason: {cont.get('reason')}")
        if cont.get("logStreamName"):
            print(f"logStreamName: {cont.get('logStreamName')}")
        if args.logs:
            print("--- CloudWatch tail ---")
            print(aws.cloudwatch_tail_for_job(job, lines=args.log_lines))
    return 0


def cmd_cloud_progress(args: argparse.Namespace) -> int:
    aws = Aws.from_env(args.profile, args.region)
    job = aws.batch_describe_job(args.job_id)
    if not job:
        raise SystemExit(f"job not found: {args.job_id}")
    stream = (job.get("container") or {}).get("logStreamName")
    if not stream:
        print(f"{args.job_id}\t{job.get('status')}\t(no log stream yet)")
        return 0
    try:
        data = aws.run(
            [
                "logs",
                "filter-log-events",
                "--log-group-name",
                "/aws/batch/job",
                "--log-stream-names",
                stream,
                "--filter-pattern",
                args.filter,
            ],
            json_out=True,
        )
    except Exception as exc:
        raise SystemExit(str(exc))
    events = data.get("events", []) if data else []
    lines = [str(e.get("message", "")) for e in events]
    print(f"jobId: {job.get('jobId')}")
    print(f"status: {job.get('status')}")
    print(f"startedAt: {human_ms(job.get('startedAt'))}")
    print(f"logStreamName: {stream}")
    print("--- matched log lines ---")
    for line in lines[-args.lines :]:
        print(line)
    return 0


def cmd_cloud_watch(args: argparse.Namespace) -> int:
    aws = Aws.from_env(args.profile, args.region)
    job = aws.batch_wait_terminal(args.job_id, interval=args.interval, max_seconds=args.max_seconds)
    if args.record_run_id:
        status = str(job.get("status", "")).lower()
        RunRegistry(args.registry).append(
            RunEvent(
                run_id=args.record_run_id,
                event="cloud_watch",
                attrs={"status": "succeeded" if status == "succeeded" else status, "aws_status": job.get("status")},
            )
        )
    if job.get("status") == "FAILED" or args.logs:
        print("--- CloudWatch tail ---")
        print(aws.cloudwatch_tail_for_job(job, lines=args.log_lines))
    return 0 if job.get("status") == "SUCCEEDED" else 2


def parse_submit_log(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    m = re.search(r'"jobId"\s*:\s*"([^"]+)"', text)
    if m:
        out["aws_job_id"] = m.group(1)
    m = re.search(r'"jobName"\s*:\s*"([^"]+)"', text)
    if m:
        out["aws_job_name"] = m.group(1)
    m = re.search(r"DATASET_S3_PREFIX=(\S+)", text)
    if m:
        out["s3_prefix"] = m.group(1)
    m = re.search(r"COMMAND:\s*(.+)", text)
    if m:
        out["command"] = m.group(1).strip()
    return out


def _load_local_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _load_s3_json(aws: Aws, uri: str) -> dict[str, Any]:
    text = aws.run(["s3", "cp", uri, "-"], json_out=False)
    return json.loads(text)


def _manifest_validation(m: dict[str, Any], *, expect_history: int, expect_train: int, expect_dev: int, expect_train_shards: int | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if m.get("history_plies") != expect_history:
        problems.append(f"history_plies={m.get('history_plies')} expected={expect_history}")
    if m.get("total_train_rows") != expect_train:
        problems.append(f"total_train_rows={m.get('total_train_rows')} expected={expect_train}")
    if m.get("total_dev_rows") != expect_dev:
        problems.append(f"total_dev_rows={m.get('total_dev_rows')} expected={expect_dev}")
    shards = m.get("train_shards") or []
    if expect_train_shards is not None and len(shards) != expect_train_shards:
        problems.append(f"train_shards={len(shards)} expected={expect_train_shards}")
    if not m.get("dev"):
        problems.append("dev path missing")
    return not problems, problems


def _manifest_inputs(base_dataset_dir: str) -> list[str]:
    m = _load_local_manifest(Path(base_dataset_dir))
    inputs = m.get("reproducibility", {}).get("inputs") or []
    if not isinstance(inputs, list):
        raise SystemExit("manifest reproducibility.inputs is not a list")
    return [str(x) for x in inputs]


def cmd_cloud_adopt_log(args: argparse.Namespace) -> int:
    attrs = parse_submit_log(Path(args.log))
    if not attrs.get("aws_job_id"):
        raise SystemExit(f"no jobId found in {args.log}")
    run_id = args.run_id or f"cloud-{attrs.get('aws_job_name','job')}-{attrs['aws_job_id'][:8]}"
    attrs.update(
        {
            "kind": args.kind,
            "name": args.name or attrs.get("aws_job_name", run_id),
            "status": args.status,
            "log": rel(args.log),
            "git_sha": git_sha(),
        }
    )
    RunRegistry(args.registry).append(RunEvent(run_id=run_id, event="adopt_log", attrs=attrs))
    print(run_id)
    return 0


def cmd_cloud_validate_h8_manifest(args: argparse.Namespace) -> int:
    if bool(args.s3_prefix) == bool(args.manifest):
        raise SystemExit("provide exactly one of --manifest or --s3-prefix")
    if args.s3_prefix:
        uri = args.s3_prefix.rstrip("/") + "/manifest.json"
        m = _load_s3_json(Aws.from_env(args.profile, args.region), uri)
        source = uri
    else:
        m = _load_local_manifest(Path(args.manifest))
        source = str(args.manifest)
    ok, problems = _manifest_validation(
        m,
        expect_history=args.expect_history,
        expect_train=args.expect_train,
        expect_dev=args.expect_dev,
        expect_train_shards=args.expect_train_shards,
    )
    out = {
        "ok": ok,
        "source": source,
        "name": m.get("name"),
        "history_plies": m.get("history_plies"),
        "total_train_rows": m.get("total_train_rows"),
        "total_dev_rows": m.get("total_dev_rows"),
        "train_shards": len(m.get("train_shards") or []),
        "problems": problems,
    }
    if args.json:
        print_json(out)
    else:
        print(f"ok={ok} source={source}")
        print(f"name={out['name']} h={out['history_plies']} train={out['total_train_rows']} dev={out['total_dev_rows']} shards={out['train_shards']}")
        for p in problems:
            print(f"PROBLEM {p}")
    return 0 if ok else 2


def cmd_cloud_preflight_h8_dataset(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = _load_local_manifest(Path(args.base_dataset_dir))
    inputs = _manifest_inputs(args.base_dataset_dir)
    missing = [p for p in inputs if not (root / p).exists() and not Path(p).exists()]
    raw_bytes = sum((root / p if (root / p).exists() else Path(p)).stat().st_size for p in inputs if (root / p).exists() or Path(p).exists())
    compression_problems = [p for p in inputs if not str(p).endswith(".zst")]
    target_train_margin = math.ceil(args.max_rows * args.margin)
    target_dev_margin = math.ceil(args.dev_rows * args.margin)
    problems: list[str] = []
    warnings: list[str] = []
    if missing:
        problems.append(f"missing raw inputs: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
    if args.require_compress_inputs and compression_problems and not args.compress_inputs:
        problems.append("100M raw inputs include uncompressed .jsonl files; pass/use --compress-inputs")
    if manifest.get("history_plies") == args.expect_history:
        warnings.append("base manifest is already h8; make sure this is intentional for a rebuild")
    if manifest.get("total_train_rows") and int(manifest.get("total_train_rows")) < args.max_rows:
        warnings.append(f"base manifest train rows {manifest.get('total_train_rows')} < target {args.max_rows}; raw-input capacity estimate is recommended")
    capacity: dict[str, Any] | None = None
    if args.estimate_capacity and not missing:
        cmd = [
            sys.executable,
            "scripts/preflight_supervised_dataset_capacity.py",
            "--input",
            *inputs,
            "--max-rows",
            str(args.max_rows),
            "--dev-rows",
            str(args.dev_rows),
            "--margin",
            str(args.margin),
            "--max-rows-per-game",
            str(args.max_rows_per_game),
            "--max-rows-per-opening",
            str(args.max_rows_per_opening),
            "--max-rows-per-source",
            str(args.max_rows_per_source),
            "--skip-plies",
            str(args.skip_plies),
            "--history-plies",
            str(args.expect_history),
            "--seed",
            str(args.seed),
            "--json",
        ]
        for cap in args.source_cap or []:
            if cap:
                cmd += ["--source-cap", cap]
        if args.dedupe_fen:
            cmd.append("--dedupe-fen")
        proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
        if proc.stderr.strip() and not args.json:
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.stdout.strip():
            capacity = json.loads(proc.stdout)
        if proc.returncode != 0:
            problems.append("capacity estimate did not reach requested safety margin")
    elif not args.estimate_capacity:
        msg = "source capacity was not estimated; rerun with --estimate-capacity before submit"
        if args.max_rows >= 100_000_000 and not args.allow_without_capacity_estimate:
            problems.append(msg)
        else:
            warnings.append(msg)
    out = {
        "ok": not problems,
        "base_dataset_dir": args.base_dataset_dir,
        "base_name": manifest.get("name"),
        "base_history_plies": manifest.get("history_plies"),
        "raw_inputs": len(inputs),
        "raw_bytes": raw_bytes,
        "target": {"train_rows": args.max_rows, "dev_rows": args.dev_rows, "history_plies": args.expect_history, "train_rows_with_margin": target_train_margin, "dev_rows_with_margin": target_dev_margin},
        "selection": {"max_rows_per_game": args.max_rows_per_game, "max_rows_per_opening": args.max_rows_per_opening, "max_rows_per_source": args.max_rows_per_source, "source_cap": args.source_cap or [], "skip_plies": args.skip_plies, "seed": args.seed},
        "compression": {"require_compress_inputs": args.require_compress_inputs, "compress_inputs_planned": args.compress_inputs, "uncompressed_inputs": len(compression_problems)},
        "capacity": capacity,
        "warnings": warnings,
        "problems": problems,
    }
    if args.json:
        print_json(out)
    else:
        print(f"ok={out['ok']} base={args.base_dataset_dir} inputs={len(inputs)} raw_bytes={raw_bytes}")
        print(f"target h{args.expect_history} train={args.max_rows} dev={args.dev_rows} margin={args.margin} train_margin={target_train_margin}")
        print(f"compression require={args.require_compress_inputs} planned={args.compress_inputs} uncompressed_inputs={len(compression_problems)}")
        if capacity:
            cap = capacity.get("capacity_observed_until_stop", {})
            print(f"capacity train={cap.get('train_rows')} dev={cap.get('dev_rows')} ok={capacity.get('ok')}")
        for w in warnings:
            print(f"WARN {w}")
        for p in problems:
            print(f"PROBLEM {p}")
    return 0 if not problems else 2


def cmd_cloud_repair_plan_h8_dataset(args: argparse.Namespace) -> int:
    missing = max(0, args.expect_train - args.actual_train)
    required_with_margin = math.ceil(args.expect_train * args.margin)
    shortfall_with_margin = max(0, required_with_margin - args.actual_train)
    suggested_rows = max(shortfall_with_margin, math.ceil(missing * args.repair_multiplier))
    out = {
        "ok": False,
        "failure": {"actual_train_rows": args.actual_train, "expect_train_rows": args.expect_train, "missing_rows": missing},
        "repair_target": {"margin": args.margin, "required_train_rows_with_margin": required_with_margin, "additional_usable_rows_needed": suggested_rows},
        "preferred_actions": [
            "add more high-quality elite/TCEC raw months first",
            "run tlops cloud preflight-h8-dataset --estimate-capacity on the repaired input manifest before submit",
            "resubmit only with --compress-inputs/--upload-inputs and a fresh output prefix or clearly intentional overwrite",
            "validate S3 manifest before cache fanout",
        ],
        "fallback_actions_requiring_approval": [
            "increase --max-rows-per-game",
            "lower --skip-plies",
            "disable/raise dedupe or opening/source caps",
            "accepting a sub-100M dataset",
        ],
    }
    if args.json:
        print_json(out)
    else:
        print(f"missing_rows={missing}")
        print(f"additional_usable_rows_needed_with_margin≈{suggested_rows}")
        print("preferred_actions:")
        for item in out["preferred_actions"]:
            print(f"  - {item}")
        print("fallback_actions_requiring_approval:")
        for item in out["fallback_actions_requiring_approval"]:
            print(f"  - {item}")
    return 0


def cmd_cloud_status_h8_100m(args: argparse.Namespace) -> int:
    aws = Aws.from_env(args.profile, args.region)
    active = []
    for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
        for job in aws.batch_list_jobs(args.queue, status):
            if args.match in job.get("jobName", "") or args.match in job.get("jobId", ""):
                active.append({"status": status, "jobName": job.get("jobName"), "jobId": job.get("jobId"), "createdAt": human_ms(job.get("createdAt"))})
    manifest_status: dict[str, Any] = {"checked": False}
    if args.dataset_s3_prefix:
        try:
            m = _load_s3_json(aws, args.dataset_s3_prefix.rstrip("/") + "/manifest.json")
            ok, problems = _manifest_validation(m, expect_history=args.expect_history, expect_train=args.expect_train, expect_dev=args.expect_dev, expect_train_shards=args.expect_train_shards)
            manifest_status = {"checked": True, "exists": True, "ok": ok, "history_plies": m.get("history_plies"), "total_train_rows": m.get("total_train_rows"), "total_dev_rows": m.get("total_dev_rows"), "train_shards": len(m.get("train_shards") or []), "problems": problems}
        except Exception as exc:
            manifest_status = {"checked": True, "exists": False, "ok": False, "error": str(exc)}
    log_tail = None
    if args.log and Path(args.log).exists():
        lines = Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = lines[-args.log_lines :]
    phase = "cache_fanout_allowed" if manifest_status.get("ok") else "dataset_pending_or_invalid"
    if active:
        phase = "dataset_job_active"
    out = {"phase": phase, "active_jobs": active, "manifest": manifest_status, "log_tail": log_tail}
    if args.write_md:
        path = Path(args.write_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# h8 100M cloud status", "", f"phase: `{phase}`", "", "## Active jobs"]
        if active:
            lines += [f"- `{j['status']}` {j['createdAt']} `{j['jobName']}` `{j['jobId']}`" for j in active]
        else:
            lines.append("- none")
        lines += ["", "## Manifest", "", "```json", json.dumps(manifest_status, indent=2, sort_keys=True), "```"]
        if log_tail:
            lines += ["", "## Log tail", "", "```", *log_tail, "```"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.json:
        print_json(out)
    else:
        print(f"phase={phase}")
        print(f"active_jobs={len(active)}")
        for j in active:
            print(f"  {j['status']} {j['createdAt']} {j['jobName']} {j['jobId']}")
        if manifest_status.get("checked"):
            print(f"manifest_exists={manifest_status.get('exists')} ok={manifest_status.get('ok')} train={manifest_status.get('total_train_rows')} dev={manifest_status.get('total_dev_rows')} h={manifest_status.get('history_plies')}")
            for p in manifest_status.get("problems", []):
                print(f"PROBLEM {p}")
        if log_tail:
            print("--- log tail ---")
            print("\n".join(log_tail))
    return 0 if phase == "cache_fanout_allowed" else 1


def cmd_cloud_submit_h8(args: argparse.Namespace) -> int:
    root = repo_root()
    run_id = args.run_id or default_run_id("h8_dataset", args.job_name)
    log = Path(args.log or (root / "artifacts" / "ops" / f"{run_id}.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cloud/aws/submit_h8_dataset_job.sh",
        "--bucket-uri",
        args.bucket_uri,
        "--job-queue",
        args.queue,
        "--base-dataset-dir",
        args.base_dataset_dir,
        "--dataset-name",
        args.dataset_name,
        "--job-name",
        args.job_name,
        "--vcpus",
        str(args.vcpus),
        "--memory",
        str(args.memory),
        "--region",
        args.region,
        "--max-rows",
        str(args.max_rows),
        "--dev-rows",
        str(args.dev_rows),
        "--rows-per-shard",
        str(args.rows_per_shard),
        "--max-rows-per-game",
        str(args.max_rows_per_game),
        "--max-rows-per-opening",
        str(args.max_rows_per_opening),
        "--max-rows-per-source",
        str(args.max_rows_per_source),
        "--source-caps",
        args.source_caps,
        "--skip-plies",
        str(args.skip_plies),
        "--seed",
        str(args.seed),
        "--parallel-uploads",
        str(args.parallel_uploads),
    ]
    if args.compressed_dir:
        cmd += ["--compressed-dir", args.compressed_dir]
    if args.compress_inputs:
        cmd.append("--compress-inputs")
    if args.upload_inputs:
        cmd.append("--upload-inputs")
    if args.submit:
        cmd.append("--submit")
    RunRegistry(args.registry).append(
        RunEvent(
            run_id=run_id,
            event="submit_start",
            attrs={"kind": "h8_dataset", "name": args.job_name, "status": "running", "log": rel(log), "command": " ".join(cmd)},
        )
    )
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=fh, stderr=subprocess.STDOUT)
    attrs = parse_submit_log(log)
    attrs["status"] = "submitted" if proc.returncode == 0 else "failed"
    attrs["returncode"] = proc.returncode
    RunRegistry(args.registry).append(RunEvent(run_id=run_id, event="submit_done", attrs=attrs))
    print(run_id)
    return proc.returncode


def cmd_cloud_submit_cache_h7_h8(args: argparse.Namespace) -> int:
    root = repo_root()
    run_id = args.run_id or default_run_id("squareformer_cache", Path(args.dataset_s3_prefix).name or "h7_h8")
    log = Path(args.log or (root / "artifacts" / "ops" / f"{run_id}.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cloud/aws/submit_h7_h8_cache_after_h8_dataset.sh",
        "--bucket-uri",
        args.bucket_uri,
        "--dataset-s3-prefix",
        args.dataset_s3_prefix,
        "--histories",
        args.histories,
        "--job-queue",
        args.queue,
        "--job-definition",
        args.job_definition,
        "--region",
        args.region,
        "--expect-train",
        str(args.expect_train),
        "--expect-dev",
        str(args.expect_dev),
    ]
    if args.allow_partial:
        cmd.append("--allow-partial")
    cmd.append("--submit" if args.submit else "--prepare" if args.prepare else "--dry-run")
    RunRegistry(args.registry).append(
        RunEvent(
            run_id=run_id,
            event="cache_submit_start",
            attrs={"kind": "squareformer_cache", "status": "running", "log": rel(log), "command": " ".join(cmd)},
        )
    )
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    job_ids = re.findall(r'"jobId"\s*:\s*"([^"]+)"', text)
    RunRegistry(args.registry).append(
        RunEvent(
            run_id=run_id,
            event="cache_submit_done",
            attrs={"status": "submitted" if proc.returncode == 0 and args.submit else "prepared" if proc.returncode == 0 and args.prepare else "dry_run" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "aws_job_ids": job_ids},
        )
    )
    print(run_id)
    return proc.returncode


def cmd_artifact_inventory(args: argparse.Namespace) -> int:
    rows = [x.as_dict(repo_root()) for x in inventory(args.paths, depth=args.depth)]
    if args.json:
        print_json(rows)
    else:
        for r in rows[: args.limit]:
            flag = "ACTIVE" if r["active"] else "cold-ok"
            print(f"{r['bytes']:>14}\t{flag}\t{r['path']}\t{r.get('reason','')}")
    return 0


def parse_size(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    for suffix, val in [("tb", 1024**4), ("gb", 1024**3), ("mb", 1024**2), ("kb", 1024)]:
        if s.endswith(suffix):
            mult = val
            s = s[: -len(suffix)]
            break
    return int(float(s) * mult)


def cmd_artifact_cold_store(args: argparse.Namespace) -> int:
    items = inventory(args.paths, depth=args.depth)
    manifest = cold_store(
        items,
        args.to,
        dry_run=not args.execute,
        older_than_days=args.older_than_days,
        min_bytes=parse_size(args.min_size),
    )
    print_json(manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tlops", description="Tiny Leela ops CLI")
    p.add_argument("--registry", default=None, help="registry JSONL path; default artifacts/ops/runs.jsonl")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="run_cmd", required=True)
    rr = run_sub.add_parser("record")
    rr.add_argument("--run-id")
    rr.add_argument("--kind", required=True)
    rr.add_argument("--name")
    rr.add_argument("--status", default="running")
    rr.add_argument("--event", default="record")
    rr.add_argument("--attr", action="append")
    rr.set_defaults(func=cmd_run_record)
    ru = run_sub.add_parser("update")
    ru.add_argument("run_id")
    ru.add_argument("--status")
    ru.add_argument("--event", default="update")
    ru.add_argument("--attr", action="append")
    ru.set_defaults(func=cmd_run_update)
    rl = run_sub.add_parser("list")
    rl.add_argument("--kind")
    rl.add_argument("--status")
    rl.add_argument("--active", action="store_true")
    rl.add_argument("--json", action="store_true")
    rl.set_defaults(func=cmd_run_list)
    rs = run_sub.add_parser("show")
    rs.add_argument("run_id")
    rs.set_defaults(func=cmd_run_show)

    phase = sub.add_parser("phase")
    phase_sub = phase.add_subparsers(dest="phase_cmd", required=True)
    pm = phase_sub.add_parser("mark")
    pm.add_argument("root")
    pm.add_argument("phase")
    pm.add_argument("--state", default="done")
    pm.add_argument("--note")
    pm.set_defaults(func=cmd_phase_mark)
    ps = phase_sub.add_parser("status")
    ps.add_argument("root")
    ps.set_defaults(func=cmd_phase_status)

    cloud = sub.add_parser("cloud")
    cloud.add_argument("--profile")
    cloud.add_argument("--region", default="us-west-2")
    cloud_sub = cloud.add_subparsers(dest="cloud_cmd", required=True)
    cj = cloud_sub.add_parser("jobs")
    cj.add_argument("--queue", default="tiny-leela-cache-queue")
    cj.add_argument("--match")
    cj.add_argument("--statuses", nargs="+", choices=JOB_STATUSES)
    cj.add_argument("--json", action="store_true")
    cj.set_defaults(func=cmd_cloud_jobs)
    cd = cloud_sub.add_parser("describe")
    cd.add_argument("job_id")
    cd.add_argument("--json", action="store_true")
    cd.add_argument("--logs", action="store_true")
    cd.add_argument("--log-lines", type=int, default=80)
    cd.set_defaults(func=cmd_cloud_describe)
    cp = cloud_sub.add_parser("progress")
    cp.add_argument("job_id")
    cp.add_argument("--filter", default="progress")
    cp.add_argument("--lines", type=int, default=10)
    cp.set_defaults(func=cmd_cloud_progress)
    cw = cloud_sub.add_parser("watch")
    cw.add_argument("job_id")
    cw.add_argument("--interval", type=int, default=60)
    cw.add_argument("--max-seconds", type=int)
    cw.add_argument("--logs", action="store_true")
    cw.add_argument("--log-lines", type=int, default=80)
    cw.add_argument("--record-run-id")
    cw.set_defaults(func=cmd_cloud_watch)
    ca = cloud_sub.add_parser("adopt-log")
    ca.add_argument("log")
    ca.add_argument("--run-id")
    ca.add_argument("--kind", default="h8_dataset")
    ca.add_argument("--name")
    ca.add_argument("--status", default="submitted")
    ca.set_defaults(func=cmd_cloud_adopt_log)
    vh = cloud_sub.add_parser("validate-h8-manifest")
    vh.add_argument("--manifest", help="local manifest path or dataset directory")
    vh.add_argument("--s3-prefix", help="S3 dataset prefix; /manifest.json is appended")
    vh.add_argument("--expect-history", type=int, default=8)
    vh.add_argument("--expect-train", type=int, default=100000000)
    vh.add_argument("--expect-dev", type=int, default=1000000)
    vh.add_argument("--expect-train-shards", type=int, default=100)
    vh.add_argument("--json", action="store_true")
    vh.set_defaults(func=cmd_cloud_validate_h8_manifest)
    pf = cloud_sub.add_parser("preflight-h8-dataset")
    pf.add_argument("--base-dataset-dir", default="data/datasets/supervised_100m_elite_tcec_v1")
    pf.add_argument("--max-rows", type=int, default=100000000)
    pf.add_argument("--dev-rows", type=int, default=1000000)
    pf.add_argument("--expect-history", type=int, default=8)
    pf.add_argument("--rows-per-shard", type=int, default=1000000)
    pf.add_argument("--margin", type=float, default=1.15)
    pf.add_argument("--max-rows-per-game", type=int, default=64)
    pf.add_argument("--max-rows-per-opening", type=int, default=100000)
    pf.add_argument("--max-rows-per-source", type=int, default=0)
    pf.add_argument("--source-cap", action="append")
    pf.add_argument("--skip-plies", type=int, default=10)
    pf.add_argument("--seed", type=int, default=7)
    pf.add_argument("--dedupe-fen", action="store_true")
    pf.add_argument("--require-compress-inputs", action="store_true", default=True)
    pf.add_argument("--no-require-compress-inputs", dest="require_compress_inputs", action="store_false")
    pf.add_argument("--compress-inputs", action="store_true", help="declare that submit plan includes --compress-inputs")
    pf.add_argument("--estimate-capacity", action="store_true", help="stream raw inputs and simulate cloud row selection")
    pf.add_argument("--allow-without-capacity-estimate", action="store_true", help="do not fail 100M preflight when the expensive capacity pass is omitted")
    pf.add_argument("--json", action="store_true")
    pf.set_defaults(func=cmd_cloud_preflight_h8_dataset)
    rp = cloud_sub.add_parser("repair-plan-h8-dataset")
    rp.add_argument("--actual-train", type=int, required=True)
    rp.add_argument("--expect-train", type=int, default=100000000)
    rp.add_argument("--margin", type=float, default=1.15)
    rp.add_argument("--repair-multiplier", type=float, default=1.25)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_cloud_repair_plan_h8_dataset)
    st = cloud_sub.add_parser("status-h8-100m")
    st.add_argument("--queue", default="tiny-leela-cache-queue")
    st.add_argument("--match", default="h8-dataset-100m")
    st.add_argument("--dataset-s3-prefix", default="s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1")
    st.add_argument("--expect-history", type=int, default=8)
    st.add_argument("--expect-train", type=int, default=100000000)
    st.add_argument("--expect-dev", type=int, default=1000000)
    st.add_argument("--expect-train-shards", type=int, default=100)
    st.add_argument("--log")
    st.add_argument("--log-lines", type=int, default=20)
    st.add_argument("--write-md", help="write a small durable status dashboard markdown file")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_cloud_status_h8_100m)
    sh = cloud_sub.add_parser("submit-h8")
    sh.add_argument("--run-id")
    sh.add_argument("--bucket-uri", required=True)
    sh.add_argument("--queue", default="tiny-leela-cache-queue")
    sh.add_argument("--base-dataset-dir", required=True)
    sh.add_argument("--dataset-name", required=True)
    sh.add_argument("--job-name", default="h8-dataset")
    sh.add_argument("--compressed-dir")
    sh.add_argument("--vcpus", type=int, default=16)
    sh.add_argument("--memory", type=int, default=65536)
    sh.add_argument("--max-rows", type=int, required=True)
    sh.add_argument("--dev-rows", type=int, default=1000000)
    sh.add_argument("--rows-per-shard", type=int, default=1000000)
    sh.add_argument("--max-rows-per-game", type=int, default=64)
    sh.add_argument("--max-rows-per-opening", type=int, default=100000)
    sh.add_argument("--max-rows-per-source", type=int, default=0)
    sh.add_argument("--source-caps", default="")
    sh.add_argument("--skip-plies", type=int, default=10)
    sh.add_argument("--seed", type=int, default=7)
    sh.add_argument("--parallel-uploads", type=int, default=4)
    sh.add_argument("--compress-inputs", action="store_true")
    sh.add_argument("--upload-inputs", action="store_true")
    sh.add_argument("--submit", action="store_true")
    sh.add_argument("--log")
    sh.set_defaults(func=cmd_cloud_submit_h8)
    sc = cloud_sub.add_parser("submit-cache-h7-h8")
    sc.add_argument("--run-id")
    sc.add_argument("--bucket-uri", required=True)
    sc.add_argument("--dataset-s3-prefix", required=True)
    sc.add_argument("--histories", default="7,8")
    sc.add_argument("--queue", default="tiny-leela-cache-queue")
    sc.add_argument("--job-definition", default="tiny-leela-cache-squareformer-cache")
    sc.add_argument("--expect-train", type=int, required=True)
    sc.add_argument("--expect-dev", type=int, required=True)
    sc.add_argument("--allow-partial", action="store_true")
    sc.add_argument("--prepare", action="store_true")
    sc.add_argument("--submit", action="store_true")
    sc.add_argument("--log")
    sc.set_defaults(func=cmd_cloud_submit_cache_h7_h8)

    art = sub.add_parser("artifact")
    art_sub = art.add_subparsers(dest="artifact_cmd", required=True)
    ai = art_sub.add_parser("inventory")
    ai.add_argument("paths", nargs="*", default=["artifacts", "data/datasets"])
    ai.add_argument("--depth", type=int, default=1)
    ai.add_argument("--limit", type=int, default=50)
    ai.add_argument("--json", action="store_true")
    ai.set_defaults(func=cmd_artifact_inventory)
    ac = art_sub.add_parser("cold-store")
    ac.add_argument("--to", required=True)
    ac.add_argument("paths", nargs="*", default=["artifacts", "data/datasets"])
    ac.add_argument("--depth", type=int, default=1)
    ac.add_argument("--older-than-days", type=float, default=14)
    ac.add_argument("--min-size", default="0")
    ac.add_argument("--execute", action="store_true", help="actually move files; default is dry-run")
    ac.set_defaults(func=cmd_artifact_cold_store)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
