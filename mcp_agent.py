from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AGENTD_ROOT = Path(__file__).resolve().parent  # .../DrugDiscoveryAgent/AgentD


def _ensure_agentd_importable() -> None:
    """Ensure `import agentD` works without breaking fastmcp's `import mcp.*`."""
    if str(AGENTD_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENTD_ROOT))


def _should_force_conda_run() -> bool:
    """
    If set, subprocess jobs run via `conda run -n <env> ...` even if the MCP
    server itself isn't started from that conda env.
    """
    return os.getenv("AGENTD_FORCE_CONDA_RUN", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _conda_env_name() -> str:
    return os.getenv("AGENTD_CONDA_ENV", "agentd").strip() or "agentd"


def _python_worker_prefix() -> List[str]:
    """
    Prefix for launching python workers.
    - If MCP server is launched inside the desired env, sys.executable is enough.
    - If not, AGENTD_FORCE_CONDA_RUN=1 will force `conda run -n agentd python ...`.
    """
    if _should_force_conda_run():
        return ["conda", "run", "-n", _conda_env_name(), "python"]
    return [sys.executable]


def _command_in_env(cmd: List[str]) -> List[str]:
    """
    Wrap a non-python command to run inside the conda env if requested.
    """
    if _should_force_conda_run():
        return ["conda", "run", "-n", _conda_env_name()] + cmd
    return cmd


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ProgressTracker:
    """
    Tracks pipeline progress to a status.json file that can be monitored externally.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.status_file = run_dir / "status.json"
        self.steps: List[Dict[str, Any]] = []
        self.current_step = ""
        self.started_at = _now_utc_iso()

    def update(self, step: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Update the current step and write to status file."""
        self.current_step = step
        step_entry = {
            "step": step,
            "timestamp": _now_utc_iso(),
            "details": details or {},
        }
        self.steps.append(step_entry)
        self._write()

    def _write(self) -> None:
        status = {
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "current_step": self.current_step,
            "last_updated": _now_utc_iso(),
            "steps": self.steps,
        }
        _safe_json_dump(self.status_file, status)


def _parse_fasta_to_sequence(fasta_text: str) -> str:
    """
    Convert UniProt FASTA (with header + newlines) into a single sequence string.
    """
    if not fasta_text:
        return ""
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if lines[0].startswith(">"):
        lines = lines[1:]
    seq = "".join(lines)
    # Ensure no whitespace
    return "".join(seq.split())


def _load_api_keys_fallback() -> None:
    """
    Prefer env vars; if missing, optionally fall back to configs/secret_keys.py.
    Never prints secrets.
    """
    if os.getenv("OPENAI_API_KEY") and os.getenv("SERPER_API_KEY"):
        return
    try:
        # configs.secret_keys.py 
        from configs import secret_keys

        if not os.getenv("SERPER_API_KEY") and getattr(secret_keys, "serper_api_key", None):
            os.environ["SERPER_API_KEY"] = secret_keys.serper_api_key
        if not os.getenv("OPENAI_API_KEY") and getattr(secret_keys, "openai_api_key", None):
            os.environ["OPENAI_API_KEY"] = secret_keys.openai_api_key
    except Exception:
        return


# -----------------------------
# Job management (subprocesses)
# -----------------------------


@dataclass(frozen=True)
class JobInfo:
    job_id: str
    pid: int
    command: List[str]
    cwd: str
    log_path: str
    started_at: str


class JobManager:
    """
    Minimal job manager: start subprocesses with stdout/stderr to log file,
    persist metadata to runs/<run_id>/jobs/<job_id>.json, and provide status/logs.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.jobs_dir = run_dir / "jobs"
        self.logs_dir = run_dir / "logs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _job_meta_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def start(
        self,
        *,
        name: str,
        command: List[str],
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> JobInfo:
        job_id = f"{name}_{uuid.uuid4().hex[:12]}"
        log_path = self.logs_dir / f"{job_id}.log"
        log_f = open(log_path, "wb")

        # Ensure cwd exists
        cwd.mkdir(parents=True, exist_ok=True)

        # Merge env
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        # Start process
        p = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=proc_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

        info = JobInfo(
            job_id=job_id,
            pid=int(p.pid),
            command=command,
            cwd=str(cwd),
            log_path=str(log_path),
            started_at=_now_utc_iso(),
        )
        _safe_json_dump(self._job_meta_path(job_id), {"status": "running", **info.__dict__})
        return info

    def _pid_is_alive(self, pid: int) -> bool:
        """
        Check if a process is alive, handling zombie processes on Linux.
        A zombie process (defunct) still has a PID but is not truly running.
        """
        try:
            os.kill(pid, 0)
        except OSError:
            return False

        # On Linux, check /proc/<pid>/status for zombie state
        proc_status = Path(f"/proc/{pid}/status")
        if proc_status.exists():
            try:
                content = proc_status.read_text()
                for line in content.splitlines():
                    if line.startswith("State:"):
                        # State: Z (zombie) or State: R (running), etc.
                        if "Z" in line or "zombie" in line.lower():
                            return False
                        break
            except Exception:
                pass

        return True

    def status(self, job_id: str) -> Dict[str, Any]:
        meta_path = self._job_meta_path(job_id)
        if not meta_path.exists():
            return {"job_id": job_id, "status": "unknown", "error": "job id not found"}

        meta = _safe_json_load(meta_path)
        pid = int(meta.get("pid", -1))
        alive = pid > 0 and self._pid_is_alive(pid)

        if meta.get("status") == "running" and not alive:
            # We don't have exit code portably without psutil; infer from log footer if worker writes it.
            meta["status"] = "finished"
            meta["finished_at"] = _now_utc_iso()
            _safe_json_dump(meta_path, meta)

        return meta

    def tail_logs(self, job_id: str, max_bytes: int = 100_000) -> Dict[str, Any]:
        meta = self.status(job_id)
        log_path = meta.get("log_path")
        if not log_path:
            return {"job_id": job_id, "error": "no log_path"}
        lp = Path(log_path)
        if not lp.exists():
            return {"job_id": job_id, "error": "log file not found", "log_path": log_path}
        data = lp.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = repr(data)
        return {"job_id": job_id, "log_path": log_path, "tail": text}

    def cancel(self, job_id: str) -> Dict[str, Any]:
        meta = self.status(job_id)
        pid = int(meta.get("pid", -1))
        if pid <= 0:
            return {"job_id": job_id, "status": meta.get("status", "unknown"), "error": "no pid"}
        if not self._pid_is_alive(pid):
            return {"job_id": job_id, "status": "finished"}
        try:
            os.kill(pid, signal.SIGTERM)
            meta["status"] = "cancelled"
            meta["cancelled_at"] = _now_utc_iso()
            _safe_json_dump(self._job_meta_path(job_id), meta)
            return meta
        except Exception as e:
            return {"job_id": job_id, "error": str(e)}


# -----------------------------
# Worker implementations
# -----------------------------


def _worker_predict_affinity(*, sequence: str, smiles_csv: str, out_dir: str) -> int:
    """
    Runs BAPULM affinity prediction in a separate process to release GPU memory after exit.
    """
    # Suppress warnings to keep output clean
    import warnings
    warnings.filterwarnings("ignore")
    
    # Redirect stderr to suppress RDKit warnings
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        _load_api_keys_fallback()
        os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)

        _ensure_agentd_importable()
        from agentD.tools.prediction import predict_affinity_batch  # type: ignore

        payload = json.dumps({"sequence": sequence, "smiles_path": smiles_csv})
        result = predict_affinity_batch(payload)
        return 0
    finally:
        sys.stderr = old_stderr


def _worker_boltz_predict(*, yaml_path: str, out_dir: str, extra_args: List[str]) -> int:
    """
    Runs `boltz predict` as a subprocess worker.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    cmd = ["boltz", "predict", yaml_path] + extra_args
    # Run with suppressed output (logs go to boltz's own output files)
    completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return int(completed.returncode)


def _worker_predict_admet(*, smiles_csv: str, out_dir: str) -> int:
    """
    Runs ADMET prediction in a separate process.
    This is necessary because the DeepPK API can take several minutes and would
    otherwise cause MCP client timeouts.
    """
    # Suppress warnings to keep output clean
    import warnings
    warnings.filterwarnings("ignore")
    
    # Redirect stderr to suppress RDKit warnings
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        _load_api_keys_fallback()
        os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)

        _ensure_agentd_importable()
        from agentD.tools.prediction import get_admet_predictions  # type: ignore

        result = get_admet_predictions(smiles_csv)
        return 0
    finally:
        sys.stderr = old_stderr


def _run_worker_from_argv(argv: List[str]) -> int:
    # Suppress ALL stderr output at the earliest possible point
    # This prevents RDKit C++ warnings from appearing in terminal
    import io
    import os
    import warnings
    warnings.filterwarnings("ignore")
    
    # Redirect stderr to /dev/null at OS level to catch C-level output
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)  # 2 = stderr file descriptor
    os.close(devnull_fd)
    sys.stderr = io.StringIO()
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("_worker", nargs="?")
    parser.add_argument("--kind", required=True, choices=["affinity", "boltz", "admet"])
    parser.add_argument("--sequence", default="")
    parser.add_argument("--smiles_csv", default="")
    parser.add_argument("--yaml_path", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--extra_args_json", default="[]")
    args = parser.parse_args(argv)

    if args.kind == "affinity":
        return _worker_predict_affinity(
            sequence=args.sequence, smiles_csv=args.smiles_csv, out_dir=args.out_dir
        )
    if args.kind == "admet":
        return _worker_predict_admet(
            smiles_csv=args.smiles_csv, out_dir=args.out_dir
        )
    if args.kind == "boltz":
        extra_args = json.loads(args.extra_args_json)
        return _worker_boltz_predict(
            yaml_path=args.yaml_path, out_dir=args.out_dir, extra_args=list(extra_args)
        )
    raise ValueError(f"Unknown worker kind: {args.kind}")


# -----------------------------
# Pipeline helpers (in-process)
# -----------------------------


def _ensure_run_dir(run_root: Path, run_id: Optional[str] = None) -> Tuple[str, Path]:
    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _copy_into_run_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _merge_affinity_admet(affinity_csv: Path, admet_csv: Path, out_csv: Path) -> Path:
    """
    Merge affinity and ADMET CSVs.
    
    Strategy: Use row index for merge since:
    - Both predictions use the same input CSV in the same order
    - ADMET API may return slightly different SMILES (canonicalization)
    - Affinity uses input SMILES directly
    
    We use affinity's SMILES as the canonical one and merge on row position.
    """
    import pandas as pd

    df_aff = pd.read_csv(affinity_csv)
    df_admet = pd.read_csv(admet_csv)
    
    if "SMILES" not in df_aff.columns:
        raise ValueError("Affinity CSV must include a 'SMILES' column")
    
    # Both files should have same number of rows (same input)
    # Merge by position (index), use affinity SMILES as canonical
    if len(df_aff) != len(df_admet):
        # Fallback to SMILES-based merge if row counts don't match
        if "SMILES" not in df_admet.columns:
            raise ValueError("ADMET CSV must include a 'SMILES' column for SMILES-based merge")
        merged = pd.merge(df_admet, df_aff, on="SMILES", how="inner")
    else:
        # Index-based merge: drop ADMET's SMILES, use affinity's SMILES
        df_admet_no_smiles = df_admet.drop(columns=["SMILES"], errors="ignore")
        df_admet_no_smiles = df_admet_no_smiles.reset_index(drop=True)
        df_aff = df_aff.reset_index(drop=True)
        merged = pd.concat([df_aff, df_admet_no_smiles], axis=1)
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


def _score_candidates(property_csv: Path, out_csv: Path) -> Path:
    """
    Add rule-based booleans and QED + num_passed_rules.
    Uses RDKit-derived properties primarily, and CSV values for specific FDA fields.
    """
    import io
    import warnings
    import pandas as pd
    
    # Suppress RDKit warnings including C-level output
    warnings.filterwarnings("ignore")
    
    # OS-level stderr suppression to catch C-level RDKit output
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        from agentD.analysis.drug_likeness_analyzer import DrugLikenessAnalyzer  # type: ignore

        df = pd.read_csv(property_csv)
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            d = r.to_dict()
            smiles = d.get("SMILES")
            analyzer = DrugLikenessAnalyzer(data=d, smiles=smiles)
            report = analyzer.get_summary_report()

            # Normalize into flat columns
            lip = bool(report.get("lipinski_rule_of_5"))
            veb = bool(report.get("veber_rule"))
            gho = bool(report.get("ghose_filter"))
            ro3 = bool(report.get("rule_of_3"))
            opr = bool(report.get("oprea_lead_like"))
            num_passed = int(sum([lip, veb, gho, ro3, opr]))
            qed = analyzer.calculated_properties.get("qed")

            out_row = dict(d)
            out_row.update(
                {
                    "lipinski_rule_of_5": lip,
                    "veber_rule": veb,
                    "ghose_filter": gho,
                    "rule_of_3": ro3,
                    "oprea_lead_like": opr,
                    "num_passed_rules": num_passed,
                    "QED": qed,
                }
            )
            rows.append(out_row)
        scored = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_csv, index=False)
        return out_csv
    finally:
        sys.stderr = old_stderr
        # Restore OS-level stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


def _select_final_candidates(
    scored_csv: Path,
    out_csv: Path,
    *,
    top_k: int = 10,
    min_pkd: float = 6.0,
    min_qed: float = 0.55,
) -> Path:
    """
    Select final candidates for Boltz based on drug-likeness criteria.
    
    Filters applied:
    - (i) satisfy Oprea's lead-likeness filter
    - (ii) pass at least 2 of 3 drug-likeness rules (Lipinski, Veber, Ghose)
    - (iii) have predicted pKd value greater than min_pkd (default 6.0)
    - (iv) have QED score greater than min_qed (default 0.55)
    
    Candidates passing all filters are sorted by pKd and top_k are selected.
    """
    import pandas as pd

    df = pd.read_csv(scored_csv)
    original_count = len(df)

    # Filter by Oprea lead-like
    if "oprea_lead_like" in df.columns:
        df = df[df["oprea_lead_like"] == True]

    # Filter by at least 2 of 3 drug-likeness rules
    rule_cols = ["lipinski_rule_of_5", "veber_rule", "ghose_filter"]
    existing_rules = [c for c in rule_cols if c in df.columns]
    if existing_rules:
        df["drug_rules_passed"] = df[existing_rules].sum(axis=1)
        df = df[df["drug_rules_passed"] >= 2]

    # Filter by pKd
    if "Affinity [pKd]" in df.columns:
        df = df[df["Affinity [pKd]"] > min_pkd]

    # Filter by QED
    if "QED" in df.columns:
        df = df[df["QED"] >= min_qed]

    # Sort by affinity (pKd) descending, take top_k
    if "Affinity [pKd]" in df.columns:
        df = df.sort_values(by="Affinity [pKd]", ascending=False)

    selected = df.head(top_k)
    
    # Keep only useful columns for final output (clean and simple)
    keep_cols = [
        "SMILES",
        "Affinity [pKd]",
        "QED",
        "lipinski_rule_of_5",
        "veber_rule", 
        "ghose_filter",
        "oprea_lead_like",
    ]
    # Only keep columns that exist
    final_cols = [c for c in keep_cols if c in selected.columns]
    selected = selected[final_cols]
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)
    return out_csv


def _refine_smiles(
    in_csv: Path,
    out_csv: Path,
    *,
    protein: str,
    model: str = "gpt-4o",
) -> Path:
    """
    Run the LLM-driven SMILES refinement. This is the step that was manual in notebooks.
    We automate it and save a deterministic mapping file with Updated_SMILES + rationale.
    Output format: SMILES, Updated_SMILES, Property, Rationale
    """
    import io
    import warnings
    
    warnings.filterwarnings("ignore")
    
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        _load_api_keys_fallback()
        from agentD.agents import agentD as AgentDClass  # type: ignore
        from agentD.tools.prediction import check_smiles_validity  # type: ignore
        from agentD.prompts.molecular_refinement import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS  # type: ignore
        from agentD.utils import process_dataset_with_agent  # type: ignore

        tools = [check_smiles_validity]
        refinement_agent = AgentDClass(
            tools,
            model=model,
            prefix=PREFIX,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
        ).agent

        df_out = process_dataset_with_agent(str(in_csv), protein, tools, refinement_agent)

        keep_cols = ["SMILES", "Updated_SMILES", "Property", "Rationale"]
        available_cols = [c for c in keep_cols if c in df_out.columns]
        df_out = df_out[available_cols]

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        return out_csv
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


def _extract_updated_smiles(refinement_csv: Path, out_smiles_csv: Path) -> Path:
    """Extract Updated_SMILES from refinement.csv and rename to SMILES for next iteration."""
    import pandas as pd

    df = pd.read_csv(refinement_csv)
    if "Updated_SMILES" not in df.columns:
        raise ValueError("Refinement CSV missing Updated_SMILES column")
    updated = df[["Updated_SMILES"]].dropna()
    updated = updated.rename(columns={"Updated_SMILES": "SMILES"})
    updated = updated[updated["SMILES"].astype(str).str.strip() != ""]
    out_smiles_csv.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(out_smiles_csv, index=False)
    return out_smiles_csv


def _cleanup_intermediate_files(prop_dir: Path, keep_files: List[str]) -> None:
    """
    Remove intermediate prediction files from property directory.
    Only keeps the files specified in keep_files list.
    """
    if not prop_dir.exists():
        return
    
    for f in prop_dir.iterdir():
        if f.is_file() and f.suffix == ".csv":
            if f.name not in keep_files:
                try:
                    f.unlink()
                except Exception:
                    pass  


def _write_smiles_csv(smiles: str, out_csv: Path) -> Path:
    import pandas as pd

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"SMILES": smiles}]).to_csv(out_csv, index=False)
    return out_csv


def _combine_pool_csvs(pool_dir: Path, out_csv: Path) -> Path:
    """
    Combine SMILES from Reinvent_sampling.csv and Mol2Mol_sampling.csv into a single CSV.
    Removes duplicates and ensures only valid SMILES are included.
    """
    import pandas as pd

    reinvent_csv = pool_dir / "Reinvent_sampling.csv"
    mol2mol_csv = pool_dir / "Mol2Mol_sampling.csv"

    dfs = []
    for csv_path in [reinvent_csv, mol2mol_csv]:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "SMILES" in df.columns:
                    dfs.append(df[["SMILES"]])
            except Exception:
                pass

    if not dfs:
        return out_csv  
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["SMILES"])
    combined = combined[combined["SMILES"].astype(str).str.strip() != ""]
    combined = combined.drop_duplicates(subset=["SMILES"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return out_csv


def _run_reinvent_pooling(
    *,
    run_dir: Path,
    initial_smiles: str,
    model: str = "gpt-4o",
    num_smiles: int = 20,  # Number of SMILES to sample (reduced for faster testing)
) -> Dict[str, Any]:
    """
    Pooling wrapper. Uses existing tool functions in generation.py.

    Uses REINVENT_PATH from configs/tool_globals.py.
    """
    _load_api_keys_fallback()
    os.chdir(run_dir)

    _ensure_agentd_importable()
    import agentD.tools.generation as gen  
    from configs.tool_globals import REINVENT_PATH  

    # Validate that generation.py can find expected file
    expected_sampling = Path(REINVENT_PATH) / "configs" / "toml" / "sampling.toml"
    if not expected_sampling.exists():
        return {
            "ok": False,
            "error": "REINVENT sampling.toml not found",
            "hint": "Update REINVENT_PATH in configs/tool_globals.py",
            "expected": str(expected_sampling),
        }

    # Mol2Mol
    gen.save_smi_for_mol2mol(initial_smiles)
    mol2mol_toml = gen.update_reinvent_config("Mol2Mol")
    reinvent_toml = gen.update_reinvent_config("Reinvent")

    # Override num_smiles in generated TOML files for faster testing
    def _patch_num_smiles(toml_path: str, count: int):
        """Patch num_smiles in TOML file without modifying original configs."""
        import re
        with open(toml_path, "r") as f:
            content = f.read()

        content = re.sub(r"num_smiles\s*=\s*\d+", f"num_smiles = {count}", content)
        with open(toml_path, "w") as f:
            f.write(content)

    _patch_num_smiles(mol2mol_toml, num_smiles)
    _patch_num_smiles(reinvent_toml, num_smiles)

    # Execute REINVENT. 
    def _run_reinvent_cfg(cfg: str) -> str:
        log_file = cfg.replace(".toml", ".log")
        cmd = _command_in_env(["reinvent", "-l", log_file, cfg])
        try:
            subprocess.run(cmd, check=True, cwd=str(run_dir))
            return "REINVENT execution completed successfully."
        except subprocess.CalledProcessError as e:
            return f"Error occurred while running REINVENT: {e}"
        except FileNotFoundError:
            return "Error: 'reinvent' command not found. Ensure REINVENT is installed in the active environment."

    mol2mol_result = _run_reinvent_cfg(mol2mol_toml)
    reinvent_result = _run_reinvent_cfg(reinvent_toml)
    return {
        "ok": True,
        "configs": {"Mol2Mol": mol2mol_toml, "Reinvent": reinvent_toml},
        "results": {"Mol2Mol": mol2mol_result, "Reinvent": reinvent_result},
        "pool_dir": str(run_dir / "pool"),
    }


def _generate_boltz_yamls(
    *,
    run_dir: Path,
    protein_sequence: str,
    smiles_list: List[str],
) -> List[str]:
    os.chdir(run_dir)
    _ensure_agentd_importable()
    from agentD.tools.generation import generate_complex_structure  # type: ignore

    yaml_paths: List[str] = []
    for smi in smiles_list:
        payload = json.dumps({"sequence": protein_sequence, "smiles": smi})
        _ = generate_complex_structure(payload)
        cfg_dir = run_dir / "configs"
        if not cfg_dir.exists():
            continue
        yamls = sorted(cfg_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if yamls:
            yaml_paths.append(str(yamls[0].resolve()))
    return yaml_paths


def _wait_for_jobs(
    jm: JobManager,
    job_ids: List[str],
    progress: ProgressTracker,
    progress_key: str,
    poll_interval: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Wait for multiple jobs to complete, updating progress periodically.
    Returns dict of job_id -> final status.
    """
    wait_count = 0
    while True:
        statuses = {jid: jm.status(jid) for jid in job_ids}
        running = [jid for jid, s in statuses.items() if s.get("status") == "running"]
        if not running:
            break
        time.sleep(poll_interval)
        wait_count += 1
        # Update progress every 30 seconds
        if wait_count % 6 == 0:
            progress.update(progress_key, {
                "substep": "waiting",
                "wait_seconds": wait_count * poll_interval,
                "statuses": {jid: s.get("status") for jid, s in statuses.items()},
            })
    return {jid: jm.status(jid) for jid in job_ids}


def _run_prediction_jobs(
    *,
    jm: JobManager,
    iter_dir: Path,
    smiles_csv: Path,
    seq: str,
    iteration: int,
    suffix: str = "",
) -> Tuple[JobInfo, JobInfo]:
    """
    Launch affinity and ADMET prediction jobs for given SMILES CSV.
    Returns (affinity_job, admet_job).
    """
    job_name_suffix = f"_i{iteration:02d}{suffix}"

    job_aff = jm.start(
        name=f"affinity{job_name_suffix}",
        command=[
            *_python_worker_prefix(),
            str(AGENTD_ROOT / "mcp_agent.py"),
            "_worker",
            "--kind",
            "affinity",
            "--sequence",
            seq,
            "--smiles_csv",
            str(smiles_csv),
            "--out_dir",
            str(iter_dir),
        ],
        cwd=iter_dir,
        env={"PYTHONPATH": str(AGENTD_ROOT)},
    )

    job_admet = jm.start(
        name=f"admet{job_name_suffix}",
        command=[
            *_python_worker_prefix(),
            str(AGENTD_ROOT / "mcp_agent.py"),
            "_worker",
            "--kind",
            "admet",
            "--smiles_csv",
            str(smiles_csv),
            "--out_dir",
            str(iter_dir),
        ],
        cwd=iter_dir,
        env={"PYTHONPATH": str(AGENTD_ROOT)},
    )

    return job_aff, job_admet


def _find_prediction_outputs(iter_dir: Path, smiles_csv_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find affinity and ADMET output files for a given iteration."""
    prop_dir = iter_dir / "property"

    affinity_out = prop_dir / f"affinity_{smiles_csv_name}"
    if not affinity_out.exists():
        candidates = list(prop_dir.glob("affinity_*.csv"))
        affinity_out = candidates[0] if candidates else None

    admet_out = prop_dir / f"admet_{smiles_csv_name}"
    if not admet_out.exists():
        candidates = list(prop_dir.glob("admet_*.csv"))
        admet_out = candidates[0] if candidates else None

    return affinity_out, admet_out


# -----------------------------
# MCP server
# -----------------------------


def _make_mcp():
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in ("", str(AGENTD_ROOT))]
        from fastmcp import FastMCP 
    finally:
        sys.path = saved_path

    mcp = FastMCP("AgentD MCP")

    @mcp.tool
    def agentd_run_pipeline(
        protein: str,
        disease: str,
        iterations: int = 2,
        num_smiles: int = 20,
        run_boltz: bool = True,
        boltz_top_k: int = 10,
        boltz_extra_args_json: str = "[\"--use_msa_server\",\"--accelerator\",\"gpu\",\"--num_workers\",\"20\"]",
        model: str = "gpt-4o",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end drug discovery pipeline automation.

        Parameters:
        - protein: Target protein name (e.g., "BCL-2", "EGFR")
        - disease: Disease context (e.g., "chronic lymphocytic leukemia")
        - iterations: Number of refinement iterations (default 2)
        - num_smiles: Number of SMILES to sample per model in REINVENT (default 20)
        - run_boltz: Whether to run Boltz structure generation (default True)
        - boltz_top_k: Number of top candidates for Boltz (default 10)
        - boltz_extra_args_json: Extra args for Boltz as JSON string
        - model: LLM model for refinement (default "gpt-4o")
        - run_id: Custom run ID (optional, auto-generated if not provided)

        Note: REINVENT_PATH is read from configs/tool_globals.py
        
        Pipeline steps:
        1. Extraction (LLM-based): Discovers drug name, UniProt ID, FASTA, SMILES
        2. Pooling: Generate candidate molecules via REINVENT
        3. Iterative refinement loop
        4. Boltz structure generation for top candidates
        """
        _load_api_keys_fallback()
        _ensure_agentd_importable()

        run_root = AGENTD_ROOT / "runs"
        run_id2, run_dir = _ensure_run_dir(run_root, run_id)
        jm = JobManager(run_dir)
        progress = ProgressTracker(run_dir)

        progress.update("initializing", {"protein": protein, "disease": disease})

        _safe_json_dump(
            run_dir / "run_config.json",
            {
                "protein": protein,
                "disease": disease,
                "iterations": iterations,
                "num_smiles": num_smiles,
                "run_boltz": run_boltz,
                "boltz_top_k": boltz_top_k,
                "boltz_extra_args_json": boltz_extra_args_json,
                "model": model,
                "created_at": _now_utc_iso(),
            },
        )

        # --- LLM-based Extraction (same as notebook) ---
        progress.update("extraction", {"substep": "starting_llm_extraction"})
        os.chdir(run_dir)
        
        from agentD.agents import agentD as AgentDClass  # type: ignore
        from agentD.utils import get_tool_decorated_functions  # type: ignore
        from agentD.prompts.data_extraction import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS  # type: ignore
        
        # Get retrieval tools for extraction
        tools = get_tool_decorated_functions(str(AGENTD_ROOT / "agentD" / "tools" / "retrieval.py"))
        tool_names = [tool.name for tool in tools]
        tool_desc = [tool.description for tool in tools]
        
        # Create extraction agent
        extraction_agent = AgentDClass(
            tools,
            model=model,
            prefix=PREFIX,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
        ).agent
        
        # Run extraction - LLM discovers drug name, UniProt ID, FASTA, SMILES
        human_prompt = f"Suggest the potential drug molecules for {disease} targeting the protein {protein}."
        input_data = {
            "input": human_prompt,
            "tools": tools,
            "tool_names": tool_names,
            "tool_desc": tool_desc,
        }
        
        extraction_result = extraction_agent.invoke(input_data)
        progress.update("extraction", {"substep": "llm_extraction_completed"})
        
        # Save full extraction response (same as notebook)
        from agentD.utils import custom_serializer  # type: ignore
        with open(run_dir / "extraction_response.json", "w", encoding="utf-8") as f:
            json.dump(extraction_result, f, indent=2, default=custom_serializer)
        
        # Extract data from agent's intermediate steps (same approach as notebook)
        # The agent uses tools: get_uniprot_ids, fetch_uniprot_fasta, search, get_drug_smiles
        # Each step is [action, observation] where action can be an object or string
        import re
        
        drug_name = None
        smiles = None
        uniprot_id = None
        fasta = None
        
        # Parse intermediate steps to get the data the agent retrieved
        if "intermediate_steps" in extraction_result:
            for step in extraction_result.get("intermediate_steps", []):
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    
                    # Action can be an object with .tool/.tool_input or a string
                    tool_name = None
                    tool_input = None
                    
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                        tool_input = action.tool_input
                    elif isinstance(action, str):
                        # Parse from string like "tool='get_drug_smiles' tool_input='Venetoclax'..."
                        tool_match = re.search(r"tool='([^']+)'", action)
                        input_match = re.search(r"tool_input='([^']+)'", action)
                        if tool_match:
                            tool_name = tool_match.group(1)
                        if input_match:
                            tool_input = input_match.group(1)
                    
                    if tool_name == "get_uniprot_ids" and observation:
                        # Parse UniProt ID from observation like [['P10415', 'Apoptosis regulator Bcl-2']]
                        if isinstance(observation, list) and len(observation) > 0:
                            first = observation[0]
                            if isinstance(first, (list, tuple)) and len(first) > 0:
                                uniprot_id = first[0]
                            elif isinstance(first, str):
                                uniprot_id = first
                        elif isinstance(observation, str) and "P" in observation:
                            match = re.search(r"['\"]([A-Z][0-9A-Z]{4,})['\"]", observation)
                            if match:
                                uniprot_id = match.group(1)
                    
                    elif tool_name == "fetch_uniprot_fasta" and observation:
                        # FASTA sequence returned
                        fasta = observation if isinstance(observation, str) else str(observation)
                    
                    elif tool_name == "get_drug_smiles" and observation:
                        # Drug name is the input, SMILES is the observation
                        drug_name = tool_input
                        if isinstance(observation, str) and len(observation) > 10:
                            smiles = observation
        
        # Fallback: use retrieval tools directly if agent didn't get all data
        from agentD.tools import retrieval_new as R  
        
        if not uniprot_id:
            ids = R.get_uniprot_ids(protein)
            if isinstance(ids, str) or not ids:
                return {"ok": False, "run_id": run_id2, "error": f"UniProt lookup failed: {ids}"}
            uniprot_id = ids[0][0]
        
        if not fasta:
            fasta = R.fetch_uniprot_fasta(uniprot_id) or ""
        
        seq = _parse_fasta_to_sequence(fasta)
        
        # Fallback for drug name: parse from output text if not found in steps
        if not drug_name or not smiles:
            output_text = extraction_result.get("output", "")
            
            # Try common patterns like "Venetoclax" (drug names typically capitalized)
            patterns = [
                r"([A-Z][a-z]+(?:inib|mab|nib|zole|tide|stat|clax|lib|zumab))\b",  # Common drug suffixes
                r"drug[:\s]+([A-Z][a-z]+)",
                r"found\s+([A-Z][a-z]+)",
                r"identified\s+([A-Z][a-z]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, output_text)
                if match:
                    potential_drug = match.group(1).strip()
                    # Verify it's a real drug by trying to get SMILES from ChEMBL
                    test_smiles = R.get_drug_smiles(potential_drug)
                    if test_smiles and len(test_smiles) > 10:
                        drug_name = potential_drug
                        smiles = test_smiles
                        break
        
        if not drug_name or not smiles:
            return {
                "ok": False,
                "run_id": run_id2,
                "error": "Could not extract drug name from LLM response",
                "extraction_output": extraction_result.get("output", "")[:500],
            }

        extraction = {
            "protein": protein,
            "disease": disease,
            "uniprot_id": uniprot_id,
            "fasta": seq,
            "drug_name": drug_name,
            "SMILES": smiles,
        }
        _safe_json_dump(run_dir / "extraction.json", extraction)
        progress.update("extraction", {
            "substep": "completed",
            "uniprot_id": uniprot_id,
            "drug_name": drug_name,
            "smiles": smiles[:50] + "...",
        })

        # --- Pooling (REINVENT) ---
        progress.update("pooling", {"substep": "running_reinvent", "num_smiles": num_smiles})
        pool_result = _run_reinvent_pooling(
            run_dir=run_dir, initial_smiles=smiles, model=model, num_smiles=num_smiles
        )
        progress.update("pooling", {"substep": "completed", "ok": pool_result.get("ok", False)})
        _safe_json_dump(run_dir / "pooling_result.json", pool_result)

        pool_dir = run_dir / "pool"
        reinvent_csv = pool_dir / "Reinvent_sampling.csv"
        mol2mol_csv = pool_dir / "Mol2Mol_sampling.csv"

        if reinvent_csv.exists() or mol2mol_csv.exists():
            current_smiles_csv = pool_dir / "combined_candidates.csv"
            _combine_pool_csvs(pool_dir, current_smiles_csv)
        else:
            current_smiles_csv = pool_dir / "seed.csv"
            _write_smiles_csv(smiles, current_smiles_csv)

        # --- Iterative loop ---
        artifacts: Dict[str, Any] = {
            "run_id": run_id2,
            "run_dir": str(run_dir),
            "extraction_json": str(run_dir / "extraction.json"),
            "iterations": [],
        }

        total_iterations = max(1, int(iterations))
        for i in range(total_iterations):
            is_last_iteration = (i == total_iterations - 1)
            progress.update(f"iteration_{i}", {
                "substep": "starting",
                "iteration": i,
                "total_iterations": total_iterations,
                "is_last": is_last_iteration,
            })

            iter_dir = run_dir / f"iter_{i:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(iter_dir)

            iter_pool_dir = iter_dir / "pool"
            iter_pool_dir.mkdir(parents=True, exist_ok=True)
            smiles_csv_iter = iter_pool_dir / "candidates.csv"
            _copy_into_run_dir(current_smiles_csv, smiles_csv_iter)

            try:
                import pandas as pd
                num_candidates = len(pd.read_csv(smiles_csv_iter))
            except Exception:
                num_candidates = -1

            progress.update(f"iteration_{i}_prediction", {
                "substep": "starting_affinity_and_admet",
                "num_candidates": num_candidates,
            })

            job_aff, job_admet = _run_prediction_jobs(
                jm=jm,
                iter_dir=iter_dir,
                smiles_csv=smiles_csv_iter,
                seq=seq,
                iteration=i,
                suffix="_before" if is_last_iteration else "",
            )

            job_statuses = _wait_for_jobs(
                jm=jm,
                job_ids=[job_aff.job_id, job_admet.job_id],
                progress=progress,
                progress_key=f"iteration_{i}_prediction",
            )

            progress.update(f"iteration_{i}_prediction", {
                "substep": "predictions_completed",
                "affinity_status": job_statuses[job_aff.job_id].get("status"),
                "admet_status": job_statuses[job_admet.job_id].get("status"),
            })

            affinity_out, admet_out = _find_prediction_outputs(iter_dir, smiles_csv_iter.name)

            if not admet_out or not admet_out.exists():
                progress.update(f"iteration_{i}_error", {
                    "substep": "missing_admet",
                    "error": "ADMET prediction required but output not found",
                })
                _safe_json_dump(iter_dir / "errors" / "missing_admet.json", {
                    "error": "ADMET is required but output file not found",
                    "admet_job": job_admet.__dict__,
                    "admet_status": job_statuses[job_admet.job_id],
                })
                artifacts["iterations"].append({
                    "iter": i,
                    "ok": False,
                    "error": "ADMET prediction required but failed. See errors/missing_admet.json",
                })
                break

            if not affinity_out or not affinity_out.exists():
                progress.update(f"iteration_{i}_error", {
                    "substep": "missing_affinity",
                    "error": "Affinity prediction output not found",
                })
                _safe_json_dump(iter_dir / "errors" / "missing_affinity.json", {
                    "error": "Affinity prediction output file not found",
                    "affinity_job": job_aff.__dict__,
                    "affinity_status": job_statuses[job_aff.job_id],
                })
                artifacts["iterations"].append({
                    "iter": i,
                    "ok": False,
                    "error": "Affinity prediction failed. See errors/missing_affinity.json",
                })
                break

            prop_dir = iter_dir / "property"
            if is_last_iteration:
                final_affinity = prop_dir / "affinity_before.csv"
                final_admet = prop_dir / "admet_before.csv"
            else:
                final_affinity = prop_dir / "affinity.csv"
                final_admet = prop_dir / "admet.csv"

            if affinity_out != final_affinity:
                shutil.copy(affinity_out, final_affinity)
            if admet_out != final_admet:
                shutil.copy(admet_out, final_admet)

            progress.update(f"iteration_{i}_scoring", {"substep": "completed"})

            progress.update(f"iteration_{i}_refinement", {"substep": "starting_llm_refinement"})
            refinement_dir = iter_dir / "refinement"
            refinement_dir.mkdir(parents=True, exist_ok=True)
            refinement_csv = refinement_dir / "refinement.csv"
            
            _refine_smiles(smiles_csv_iter, refinement_csv, protein=protein, model=model)
            progress.update(f"iteration_{i}_refinement", {"substep": "completed"})

            if is_last_iteration:
                updated_smiles_csv = refinement_dir / "updated_smiles_temp.csv"
                _extract_updated_smiles(refinement_csv, updated_smiles_csv)
                progress.update(f"iteration_{i}_after_prediction", {
                    "substep": "starting_after_predictions",
                })

                job_aff_after, job_admet_after = _run_prediction_jobs(
                    jm=jm,
                    iter_dir=iter_dir,
                    smiles_csv=updated_smiles_csv,
                    seq=seq,
                    iteration=i,
                    suffix="_after",
                )

                job_statuses_after = _wait_for_jobs(
                    jm=jm,
                    job_ids=[job_aff_after.job_id, job_admet_after.job_id],
                    progress=progress,
                    progress_key=f"iteration_{i}_after_prediction",
                )

                affinity_after, admet_after = _find_prediction_outputs(iter_dir, updated_smiles_csv.name)

                if affinity_after and affinity_after.exists():
                    shutil.copy(affinity_after, prop_dir / "affinity_after.csv")
                if admet_after and admet_after.exists():
                    shutil.copy(admet_after, prop_dir / "admet_after.csv")

                progress.update(f"iteration_{i}_after_prediction", {
                    "substep": "completed",
                    "affinity_after_exists": (prop_dir / "affinity_after.csv").exists(),
                    "admet_after_exists": (prop_dir / "admet_after.csv").exists(),
                })

            if is_last_iteration:
                _cleanup_intermediate_files(prop_dir, [
                    "affinity_before.csv", "admet_before.csv",
                    "affinity_after.csv", "admet_after.csv",
                ])
            else:
                _cleanup_intermediate_files(prop_dir, [
                    "affinity.csv", "admet.csv",
                ])
            
            _cleanup_intermediate_files(refinement_dir, ["refinement.csv"])

            iter_artifact = {
                "iter": i,
                "ok": True,
                "is_last": is_last_iteration,
                "input_smiles_csv": str(smiles_csv_iter),
                "affinity_job": job_aff.__dict__,
                "admet_job": job_admet.__dict__,
                "refinement_csv": str(refinement_csv),
            }

            if is_last_iteration:
                iter_artifact.update({
                    "affinity_before_csv": str(prop_dir / "affinity_before.csv"),
                    "admet_before_csv": str(prop_dir / "admet_before.csv"),
                    "affinity_after_csv": str(prop_dir / "affinity_after.csv"),
                    "admet_after_csv": str(prop_dir / "admet_after.csv"),
                })
            else:
                iter_artifact.update({
                    "affinity_csv": str(prop_dir / "affinity.csv"),
                    "admet_csv": str(prop_dir / "admet.csv"),
                })

            artifacts["iterations"].append(iter_artifact)

            # Next iteration input - extract updated SMILES from refinement.csv
            # This creates a temp file that will be copied to next iter's pool/candidates.csv
            if not is_last_iteration:
                next_iter_smiles = iter_dir / "next_iter_candidates.csv"
                _extract_updated_smiles(refinement_csv, next_iter_smiles)
                current_smiles_csv = next_iter_smiles

        # --- Boltz structure generation + execution ---
        # --- Boltz: Always generate configs, optionally run structure prediction ---
        # Selection criteria: Oprea filter + 2/3 drug rules + pKd > 6.0
        boltz_jobs: List[Dict[str, Any]] = []
        boltz_yamls: List[str] = []
        smiles_for_boltz: List[str] = []
        
        if artifacts["iterations"]:
            progress.update("boltz", {"substep": "selecting_candidates"})

            last_iter = artifacts["iterations"][-1]
            if not last_iter.get("ok"):
                progress.update("boltz", {"substep": "skipped_due_to_iteration_failure"})
            else:
                # Get affinity_after and admet_after from last iteration
                affinity_after_csv = last_iter.get("affinity_after_csv")
                admet_after_csv = last_iter.get("admet_after_csv")
                
                # Define intermediate files for cleanup
                merged_csv = run_dir / "boltz_merged.csv"
                scored_csv = run_dir / "boltz_scored.csv"
                final_candidates_csv = run_dir / "boltz_candidates.csv"
                
                if affinity_after_csv and admet_after_csv and \
                   Path(affinity_after_csv).exists() and Path(admet_after_csv).exists():
                    try:
                        import pandas as pd
                        
                        # Merge affinity + admet for scoring
                        _merge_affinity_admet(
                            Path(affinity_after_csv), 
                            Path(admet_after_csv), 
                            merged_csv
                        )
                        
                        # Check if merged has data (more than just header)
                        df_merged = pd.read_csv(merged_csv)
                        if len(df_merged) == 0:
                            progress.update("boltz", {
                                "substep": "no_data_after_merge",
                                "message": "No candidates found after merging affinity and ADMET"
                            })
                        else:
                            # Score candidates (adds drug-likeness columns: oprea, lipinski, veber, ghose, QED)
                            _score_candidates(merged_csv, scored_csv)
                            
                            # Select final candidates with strict filters:
                            # - Oprea lead-like = True
                            # - At least 2 of 3 drug rules (Lipinski, Veber, Ghose)
                            # - pKd > 6.0
                            # - QED >= 0.55
                            _select_final_candidates(
                                scored_csv,
                                final_candidates_csv,
                                top_k=int(boltz_top_k),
                                min_pkd=6.0,
                                min_qed=0.55,
                            )
                            
                            # Read selected candidates
                            df_selected = pd.read_csv(final_candidates_csv)
                            smiles_for_boltz = df_selected["SMILES"].dropna().astype(str).tolist()
                            
                            progress.update("boltz", {
                                "substep": "candidates_selected",
                                "passed_filters": len(smiles_for_boltz),
                            })
                        
                    except Exception as e:
                        _safe_json_dump(run_dir / "errors" / "boltz_selection.json", {"error": str(e)})
                    finally:
                        # Always clean up intermediate files - only keep boltz_candidates.csv
                        if merged_csv.exists():
                            merged_csv.unlink()
                        if scored_csv.exists():
                            scored_csv.unlink()

                # Always generate YAML configs (even if run_boltz=False)
                progress.update("boltz", {
                    "substep": "generating_yamls",
                    "num_candidates": len(smiles_for_boltz),
                })

                if smiles_for_boltz:
                    boltz_yamls = _generate_boltz_yamls(
                        run_dir=run_dir,
                        protein_sequence=seq,
                        smiles_list=smiles_for_boltz
                    )
                    progress.update("boltz", {
                        "substep": "yamls_generated",
                        "num_yamls": len(boltz_yamls),
                        "run_boltz": run_boltz,
                    })

                    # Only run Boltz structure prediction if run_boltz=True
                    if run_boltz:
                        progress.update("boltz", {
                            "substep": "running_boltz_jobs",
                            "num_yamls": len(boltz_yamls),
                        })

                        extra_args = json.loads(boltz_extra_args_json)
                        for idx, yp in enumerate(boltz_yamls):
                            progress.update("boltz", {
                                "substep": f"running_boltz_{idx+1}_of_{len(boltz_yamls)}",
                                "yaml": yp,
                            })

                            job = jm.start(
                                name="boltz",
                                command=[
                                    *_python_worker_prefix(),
                                    str(AGENTD_ROOT / "mcp_agent.py"),
                                    "_worker",
                                    "--kind",
                                    "boltz",
                                    "--yaml_path",
                                    yp,
                                    "--out_dir",
                                    str(run_dir),
                                    "--extra_args_json",
                                    json.dumps(extra_args),
                                ],
                                cwd=run_dir,
                                env={"PYTHONPATH": str(AGENTD_ROOT)},
                            )

                            # Wait for each Boltz job
                            while jm.status(job.job_id).get("status") == "running":
                                time.sleep(5)
                            boltz_jobs.append(job.__dict__)

                        progress.update("boltz", {"substep": "completed", "num_jobs": len(boltz_jobs)})
                    else:
                        progress.update("boltz", {"substep": "configs_only_structure_skipped"})
                else:
                    progress.update("boltz", {"substep": "no_candidates_passed_filters"})

        artifacts["boltz"] = {"yamls": boltz_yamls, "jobs": boltz_jobs}
        progress.update("pipeline_complete", {"total_iterations": len(artifacts["iterations"])})
        _safe_json_dump(run_dir / "summary.json", {"ok": True, **artifacts})
        return {"ok": True, **artifacts}

    @mcp.tool
    def agentd_qna(
        question: str,
        protein: Optional[str] = None,
        disease: Optional[str] = None,
        model: str = "gpt-4o",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Q&A mode: Ask questions about drug discovery, proteins, diseases, etc.
        Uses RAG (Retrieval-Augmented Generation) with downloaded research papers.

        Parameters:
        - question: The question to answer (use "_init_papers_" to just download papers)
        - protein: Target protein name (for paper search)
        - disease: Disease name (for paper search)
        - model: LLM model (default "gpt-4o")
        - run_id: Run ID for storing papers (creates new if not provided)

        If protein and disease are provided, relevant papers will be downloaded
        to runs/<run_id>/papers/ directory.

        This is a standalone mode separate from the full pipeline.
        """
        _load_api_keys_fallback()
        _ensure_agentd_importable()

        # Create or use existing run directory for Q&A session
        run_root = AGENTD_ROOT / "runs"
        run_id2, run_dir = _ensure_run_dir(run_root, run_id)
        papers_dir = run_dir / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Save Q&A session config
        if not (run_dir / "qna_config.json").exists():
            _safe_json_dump(
                run_dir / "qna_config.json",
                {
                    "mode": "qna",
                    "protein": protein,
                    "disease": disease,
                    "model": model,
                    "created_at": _now_utc_iso(),
                },
            )

        from agentD.agents import agentD as AgentDClass  # type: ignore
        from agentD.utils import get_tool_decorated_functions  # type: ignore
        from agentD.prompts.question_answering import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS  # type: ignore
        import configs.tool_globals as tool_globals  # type: ignore

        # Override PAPER_DIR to use run-specific directory
        original_paper_dir = tool_globals.PAPER_DIR
        tool_globals.PAPER_DIR = str(papers_dir)

        # Change working directory to run_dir
        original_cwd = os.getcwd()
        os.chdir(run_dir)

        try:
            import importlib
            import agentD.tools.retrieval as retrieval_module
            
            # Reload retrieval module to pick up the new PAPER_DIR
            importlib.reload(retrieval_module)
            
            # Download papers if protein/disease specified and papers dir is empty
            papers_downloaded = []
            existing_papers = list(papers_dir.glob("*.pdf"))
            if protein and disease and not existing_papers:
                try:
                    query = f"{disease} {protein} treatment therapy year: \"2022-\""
                    papers_downloaded = retrieval_module.download_relevant_papers(query)
                except Exception as e:
                    _safe_json_dump(run_dir / "errors" / "paper_download.json", {"error": str(e)})

            # Reload again after download to ensure question_answering sees the papers
            importlib.reload(retrieval_module)
            
            # If this is just an init call to download papers, return early
            if question == "_init_papers_":
                return {
                    "ok": True,
                    "run_id": run_id2,
                    "run_dir": str(run_dir),
                    "papers_downloaded": papers_downloaded,
                    "answer": f"Papers downloaded: {len(papers_downloaded)}",
                }
            
            # Get retrieval tools (includes question_answering which does RAG)
            tools = get_tool_decorated_functions(str(AGENTD_ROOT / "agentD" / "tools" / "retrieval.py"))
            tool_names = [tool.name for tool in tools]
            tool_desc = [tool.description for tool in tools]

            qna_agent = AgentDClass(
                tools,
                model=model,
                prefix=PREFIX,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
            ).agent

            input_data = {
                "input": question,
                "tools": tools,
                "tool_names": tool_names,
                "tool_desc": tool_desc,
            }

            result = qna_agent.invoke(input_data)
            
            # Log Q&A interaction
            qna_log = run_dir / "qna_log.jsonl"
            with open(qna_log, "a") as f:
                f.write(json.dumps({
                    "timestamp": _now_utc_iso(),
                    "question": question,
                    "answer": result.get("output", str(result)),
                    "papers_downloaded": len(papers_downloaded),
                }) + "\n")

            return {
                "ok": True,
                "run_id": run_id2,
                "run_dir": str(run_dir),
                "question": question,
                "answer": result.get("output", str(result)),
                "papers_downloaded": papers_downloaded,
            }
        except Exception as e:
            return {
                "ok": False,
                "run_id": run_id2,
                "run_dir": str(run_dir),
                "question": question,
                "error": str(e),
            }
        finally:
            # Restore original PAPER_DIR and working directory
            tool_globals.PAPER_DIR = original_paper_dir
            os.chdir(original_cwd)

    @mcp.tool
    def agentd_run_status(run_id: str) -> Dict[str, Any]:
        """Get the current status and progress of a pipeline run."""
        run_dir = AGENTD_ROOT / "runs" / run_id
        status_file = run_dir / "status.json"

        if not status_file.exists():
            return {"ok": False, "run_id": run_id, "error": "Status file not found"}

        try:
            status = _safe_json_load(status_file)
            return {"ok": True, "run_id": run_id, **status}
        except Exception as e:
            return {"ok": False, "run_id": run_id, "error": str(e)}

    @mcp.tool
    def agentd_list_runs() -> Dict[str, Any]:
        """List all pipeline runs with their status."""
        runs_dir = AGENTD_ROOT / "runs"
        if not runs_dir.exists():
            return {"ok": True, "runs": []}

        runs = []
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            run_info = {"run_id": run_dir.name}

            # Try to get status
            status_file = run_dir / "status.json"
            if status_file.exists():
                try:
                    status = _safe_json_load(status_file)
                    run_info["current_step"] = status.get("current_step", "unknown")
                    run_info["started_at"] = status.get("started_at")
                    run_info["last_updated"] = status.get("last_updated")
                except Exception:
                    run_info["current_step"] = "error_reading_status"

            # Try to get config
            config_file = run_dir / "run_config.json"
            if config_file.exists():
                try:
                    config = _safe_json_load(config_file)
                    run_info["protein"] = config.get("protein")
                    run_info["drug_name"] = config.get("drug_name")
                except Exception:
                    pass

            runs.append(run_info)

        return {"ok": True, "runs": runs[:20]}  # Limit to 20 most recent

    @mcp.tool
    def agentd_job_status(run_id: str, job_id: str) -> Dict[str, Any]:
        """Get job status for a run."""
        run_dir = AGENTD_ROOT / "runs" / run_id
        jm = JobManager(run_dir)
        return jm.status(job_id)

    @mcp.tool
    def agentd_job_logs(run_id: str, job_id: str, max_bytes: int = 100_000) -> Dict[str, Any]:
        """Tail job logs for a run."""
        run_dir = AGENTD_ROOT / "runs" / run_id
        jm = JobManager(run_dir)
        return jm.tail_logs(job_id, max_bytes=max_bytes)

    @mcp.tool
    def agentd_job_cancel(run_id: str, job_id: str) -> Dict[str, Any]:
        """Cancel a job for a run."""
        run_dir = AGENTD_ROOT / "runs" / run_id
        jm = JobManager(run_dir)
        return jm.cancel(job_id)

    return mcp


def main() -> int:
    # Worker entrypoint: `python AgentD/mcp_agent.py _worker --kind ...`
    # Suppress stderr at OS level BEFORE any imports for workers
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        import os
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)  # Redirect stderr to /dev/null at OS level
        os.close(devnull_fd)
        return _run_worker_from_argv(sys.argv[1:])

    # Check for Q&A mode from command line
    if len(sys.argv) > 1 and sys.argv[1] == "_qna":
        # Interactive Q&A mode
        _load_api_keys_fallback()
        _ensure_agentd_importable()

        print("=" * 60)
        print("AgentD Q&A Mode (RAG-based)")
        print("=" * 60)
        print("\nThis mode uses downloaded research papers to answer questions.")
        print("Type 'quit' or 'exit' to leave.\n")

        # Ask if user wants to download papers
        protein = input("Enter protein name (or press Enter to skip): ").strip() or None
        drug_name = input("Enter drug name (or press Enter to skip): ").strip() or None

        if protein and drug_name:
            print(f"\nDownloading papers for {drug_name} + {protein}...")
            try:
                from agentD.tools.retrieval import download_relevant_papers
                query = f"{drug_name} {protein} year: \"2022-\""
                papers = download_relevant_papers(query)
                print(f"Downloaded {len(papers)} papers.\n")
            except Exception as e:
                print(f"Paper download failed: {e}\n")

        # Setup Q&A agent
        from agentD.agents import agentD as AgentDClass
        from agentD.utils import get_tool_decorated_functions
        from agentD.prompts.question_answering import PREFIX, SUFFIX, FORMAT_INSTRUCTIONS

        tools = get_tool_decorated_functions(str(AGENTD_ROOT / "agentD" / "tools" / "retrieval.py"))
        tool_names = [tool.name for tool in tools]
        tool_desc = [tool.description for tool in tools]

        qna_agent = AgentDClass(
            tools,
            model="gpt-4o",
            prefix=PREFIX,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
        ).agent

        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not question:
                    continue

                input_data = {
                    "input": question,
                    "tools": tools,
                    "tool_names": tool_names,
                    "tool_desc": tool_desc,
                }

                print("\nThinking...")
                result = qna_agent.invoke(input_data)
                print("\n" + "=" * 40)
                print("Answer:", result.get("output", str(result)))
                print("=" * 40)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

        return 0

    # Server entrypoint (stdio)
    # Suppress stderr to prevent library warnings from breaking MCP JSON-RPC protocol
    import io
    import warnings
    warnings.filterwarnings("ignore")
    sys.stderr = io.StringIO()  # Redirect stderr to null
    
    _load_api_keys_fallback()
    mcp = _make_mcp()
    # show_banner=False is critical for stdio transport - banner output breaks JSON-RPC protocol
    mcp.run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
