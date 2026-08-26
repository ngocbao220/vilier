import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile

import numpy as np
import soundfile as sf


TERMINAL_SUCCESS = ("complete", "completed", "success")
TERMINAL_FAILURE = ("error", "failed", "failure", "cancel", "canceled")
SOURCE_ARCHIVE_NAME = "vilier_source.zip"
PIPELINE_RESULT_NAME = "pipeline_result.zip"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
SOURCE_FILES = (
    "config.json",
    "run_pipeline.sh",
    "tools/run_asr_bundle.py",
    "pipeline/__init__.py",
    "pipeline/asr.py",
    "pipeline/audio.py",
    "pipeline/cli.py",
    "pipeline/diarization.py",
    "pipeline/labeling.py",
    "pipeline/music.py",
    "pipeline/overlap_separation.py",
    "pipeline/schema.py",
    "pipeline/timeline.py",
    "pipeline/tree_log.py",
    "pipeline/vad.py",
    "pipeline/visualization.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload one Vilier input audio to Kaggle, run GPU pipeline, and download outputs")
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-slug", default=os.environ.get("KAGGLE_DATASET_SLUG", "ngocbaotrinhtuan/vilier-pipeline-bundle"))
    parser.add_argument("--kernel-slug", default=os.environ.get("KAGGLE_KERNEL_SLUG", "ngocbaotrinhtuan/vilier-gpu-pipeline"))
    parser.add_argument("--repo-url", default=os.environ.get("KAGGLE_REPO_URL", ""))
    parser.add_argument("--repo-ref", default=os.environ.get("KAGGLE_REPO_REF", "main"))
    parser.add_argument("--accelerator", default=os.environ.get("KAGGLE_ACCELERATOR", DEFAULT_ACCELERATOR))
    parser.add_argument("--work-dir", default=os.environ.get("KAGGLE_PIPELINE_WORK_DIR", ".kaggle_pipeline_work"))
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("KAGGLE_POLL_SECONDS", "30")))
    parser.add_argument("--max-wait-seconds", type=float, default=float(os.environ.get("KAGGLE_MAX_WAIT_SECONDS", "21600")))
    parser.add_argument("--dataset-ready-seconds", type=float, default=float(os.environ.get("KAGGLE_DATASET_READY_SECONDS", "240")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    runner = KagglePipelineRunner(
        audio_id=args.audio_id,
        audio_path=audio,
        config=config,
        output_dir=output_dir,
        dataset_slug=args.dataset_slug,
        kernel_slug=args.kernel_slug,
        repo_url=args.repo_url,
        repo_ref=args.repo_ref,
        accelerator=args.accelerator,
        work_dir=work_dir,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        dataset_ready_seconds=args.dataset_ready_seconds,
        dry_run=args.dry_run,
    )
    result_dir = runner.run()
    print(f"output_dir={result_dir}")
    return 0


class KagglePipelineRunner:
    def __init__(
        self,
        audio_id: str,
        audio_path: Path,
        config: dict,
        output_dir: Path,
        dataset_slug: str,
        kernel_slug: str,
        repo_url: str,
        repo_ref: str,
        accelerator: str,
        work_dir: Path,
        poll_seconds: float = 30,
        max_wait_seconds: float = 21600,
        dataset_ready_seconds: float = 240,
        dry_run: bool = False,
    ):
        self.audio_id = audio_id
        self.audio_path = audio_path
        self.config = config
        self.output_dir = output_dir
        self.dataset_slug = dataset_slug
        self.kernel_slug = kernel_slug
        self.repo_url = repo_url
        self.repo_ref = repo_ref
        self.accelerator = accelerator
        self.work_dir = work_dir
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.dataset_ready_seconds = dataset_ready_seconds
        self.dry_run = dry_run

    def run(self) -> Path:
        if self.dry_run:
            return self._dry_run()
        dataset_dir = self.prepare_dataset_dir()
        kernel_dir = self.prepare_kernel_dir()
        self.upload_dataset(dataset_dir)
        self.wait_for_dataset_files()
        self.push_kernel(kernel_dir)
        print("Kaggle đang chạy pipeline trên GPU", flush=True)
        self.wait_for_kernel()
        print("Đã có kết quả, đang tải outputs xuống", flush=True)
        result_dir = self.download_output()
        return self.merge_outputs(result_dir)

    def _dry_run(self) -> Path:
        print("Kaggle đang chạy pipeline trên GPU", flush=True)
        print("Đã có kết quả, đang tải outputs xuống", flush=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self.config.get("entrypoint", {}).get("sample_rate", 16000))
        sf.write(self.output_dir / "audio.standardized.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
        vad = [{"id": "vad_00000", "start": 0.0, "end": 1.0, "duration": 1.0}]
        asr_segments = [
            {
                "id": "asrseg_00000",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "audio": "asr_audio/SPEAKER_00/audio_00001.wav",
            }
        ]
        asr_dir = self.output_dir / "asr_audio" / "SPEAKER_00"
        asr_dir.mkdir(parents=True, exist_ok=True)
        sf.write(asr_dir / "audio_00001.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
        (self.output_dir / "vad.json").write_text(json.dumps(vad, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "audio_id": self.audio_id,
            "source_audio": self.audio_path.name,
            "sample_rate": sample_rate,
            "duration": 1.0,
            "vad_segments": vad,
            "asr_segments": asr_segments,
            "segments": [{"id": "00000_SPEAKER_00", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "duration": 1.0}],
            "transcript": [],
        }
        transcript = [
            {
                "id": "asr_00000",
                "asr_segment_id": "asrseg_00000",
                "audio": "asr_audio/SPEAKER_00/audio_00001.wav",
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "speaker": "SPEAKER_00",
                "text": "dry-run kaggle pipeline transcript 1",
                "model": self.config.get("asr", {}).get("model", "vinai/PhoWhisper-large"),
                "language": self.config.get("asr", {}).get("language", "vi"),
            }
        ]
        if self.config.get("asr", {}).get("enabled", False):
            manifest["transcript"] = transcript
            (self.output_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "manifest.timeline.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        validate_pipeline_result(self.output_dir, asr_enabled=bool(self.config.get("asr", {}).get("enabled", False)))
        return self.output_dir

    def prepare_dataset_dir(self) -> Path:
        if not self.audio_path.exists():
            raise FileNotFoundError(self.audio_path)
        dataset_dir = self.work_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True)
        shutil.copy2(self.audio_path, dataset_dir / self.audio_path.name)
        remote_config = remote_config_for_kaggle(self.config, self.audio_path.name)
        (dataset_dir / "config.json").write_text(json.dumps(remote_config, ensure_ascii=False, indent=2), encoding="utf-8")
        overlap_config = self.config.get("overlap_separation", {})
        include_sepreformer = bool(overlap_config.get("enabled", False))
        write_source_archive(
            dataset_dir / SOURCE_ARCHIVE_NAME,
            include_sepreformer=include_sepreformer,
            sepreformer_model_name=str(overlap_config.get("model_name", "SepReformer_Base_WSJ0")),
        )
        owner, slug = split_slug(self.dataset_slug)
        metadata = {
            "title": f"Vilier pipeline bundle {self.audio_id}",
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (dataset_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return dataset_dir

    def prepare_kernel_dir(self) -> Path:
        kernel_dir = self.work_dir / "kernel"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.py").write_text(self.kernel_code(), encoding="utf-8")
        owner, slug = split_slug(self.kernel_slug)
        metadata = {
            "id": f"{owner}/{slug}",
            "title": "Vilier GPU Pipeline",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": self.accelerator,
            "dataset_sources": [self.dataset_slug],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return kernel_dir

    def kernel_code(self) -> str:
        return textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            import shutil
            import subprocess
            import sys
            import zipfile

            AUDIO_ID = {self.audio_id!r}
            WORK_ROOT = Path("/kaggle/working")
            INPUT_ROOT = Path("/kaggle/input")
            SOURCE_DIR = WORK_ROOT / "vilier_source"
            OUTPUT_ROOT = WORK_ROOT / "outputs"
            SOURCE_ARCHIVE_NAME = {SOURCE_ARCHIVE_NAME!r}

            def load_hf_secret():
                try:
                    from kaggle_secrets import UserSecretsClient
                    token = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
                except Exception:
                    token = ""
                if token:
                    os.environ["HUGGINGFACE_TOKEN"] = token
                    os.environ["HF_TOKEN"] = token
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

            def first_match(pattern):
                matches = sorted(INPUT_ROOT.rglob(pattern))
                if not matches:
                    visible = [str(path.relative_to(INPUT_ROOT)) for path in sorted(INPUT_ROOT.rglob("*"))[:200]]
                    raise FileNotFoundError(f"No {{pattern}} found under {{INPUT_ROOT}}. Visible inputs: {{visible}}")
                return matches[0]

            load_hf_secret()
            source_zip = first_match(SOURCE_ARCHIVE_NAME)
            if SOURCE_DIR.exists():
                shutil.rmtree(SOURCE_DIR)
            SOURCE_DIR.mkdir(parents=True)
            with zipfile.ZipFile(source_zip) as zf:
                zf.extractall(SOURCE_DIR)

            config_path = first_match("config.json")
            audio_path = first_match({self.audio_path.name!r})
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "setuptools<81",
                    "transformers",
                    "accelerate",
                    "soundfile",
                    "librosa",
                    "pyyaml",
                    "tqdm",
                    "pyannote.audio[separation]==3.3.2",
                    "demucs",
                ],
                check=True,
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SOURCE_DIR)
            env["MPLCONFIGDIR"] = str(WORK_ROOT / "mplconfig")
            env["NUMBA_CACHE_DIR"] = str(WORK_ROOT / "numba_cache")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.cli",
                    "--config",
                    str(config_path),
                    "--input",
                    str(audio_path),
                    "--output",
                    str(OUTPUT_ROOT),
                ],
                check=True,
                env=env,
                cwd=str(SOURCE_DIR),
            )
            result_dir = OUTPUT_ROOT / AUDIO_ID
            result_zip = WORK_ROOT / f"{{AUDIO_ID}}_{PIPELINE_RESULT_NAME}"
            with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in result_dir.rglob("*"):
                    if item.is_file():
                        zf.write(item, item.relative_to(result_dir).as_posix())
            print("pipeline_result_zip", result_zip)
            """
        ).strip() + "\n"

    def upload_dataset(self, dataset_dir: Path) -> None:
        metadata_dir = self.work_dir / "dataset_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        exists = run_command(["kaggle", "datasets", "metadata", self.dataset_slug, "-p", str(metadata_dir)], check=False).returncode == 0
        if exists:
            run_command(["kaggle", "datasets", "version", "-p", str(dataset_dir), "-m", f"Vilier pipeline bundle {self.audio_id}"], check=True)
        else:
            run_command(["kaggle", "datasets", "create", "-p", str(dataset_dir), "-q"], check=True)

    def push_kernel(self, kernel_dir: Path) -> None:
        cmd = ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
        if self.accelerator:
            cmd.extend(["--accelerator", self.accelerator])
        result = run_command(cmd, check=False)
        if result.returncode == 0:
            return
        text = (result.stdout + "\n" + result.stderr).strip()
        if self.accelerator and "accelerator" in text.lower():
            print(f"Kaggle CLI did not accept --accelerator; retrying with machine_shape={self.accelerator} only", flush=True)
            run_command(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)
            return
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)

    def wait_for_dataset_files(self) -> None:
        ready_tokens = (self.audio_path.name, "config.json", SOURCE_ARCHIVE_NAME)
        started = time.monotonic()
        last_text = ""
        saw_successful_listing = False
        while True:
            returncode, last_text = read_dataset_files_listing(self.dataset_slug)
            if returncode == 0:
                saw_successful_listing = True
            if returncode == 0 and all(token in last_text for token in ready_tokens):
                return
            if time.monotonic() - started > self.dataset_ready_seconds:
                if saw_successful_listing:
                    print(
                        "Kaggle dataset listing is paginated or delayed; continuing to kernel push after successful listing response",
                        flush=True,
                    )
                    return
                raise TimeoutError(f"Timed out waiting for Kaggle pipeline dataset files in {self.dataset_slug}.\nLast response:\n{last_text}")
            time.sleep(min(10.0, max(1.0, self.poll_seconds)))

    def wait_for_kernel(self) -> None:
        started = time.monotonic()
        while True:
            result = run_command(["kaggle", "kernels", "status", self.kernel_slug], check=True)
            text = (result.stdout + "\n" + result.stderr).lower()
            print(text.strip())
            if any(token in text for token in TERMINAL_SUCCESS):
                return
            if any(token in text for token in TERMINAL_FAILURE):
                detail = self.download_failure_output()
                raise RuntimeError(f"Kaggle kernel failed: {text.strip()}{detail}")
            if time.monotonic() - started > self.max_wait_seconds:
                raise TimeoutError(f"Timed out waiting for Kaggle kernel: {self.kernel_slug}")
            time.sleep(self.poll_seconds)

    def download_output(self) -> Path:
        result_dir = self.work_dir / "result"
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True)
        run_command(["kaggle", "kernels", "output", self.kernel_slug, "-p", str(result_dir), "-o"], check=True)
        return result_dir

    def download_failure_output(self) -> str:
        result_dir = self.work_dir / "result"
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True)
        result = run_command(["kaggle", "kernels", "output", self.kernel_slug, "-p", str(result_dir), "-o"], check=False)
        if result.returncode != 0:
            text = (result.stdout + "\n" + result.stderr).strip()
            return f"\nCould not download Kaggle failure output:\n{text}"
        logs = sorted(result_dir.glob("*.log"))
        if logs:
            return f"\nKaggle log downloaded to: {logs[0]}"
        return f"\nKaggle output downloaded to: {result_dir}"

    def merge_outputs(self, result_dir: Path) -> Path:
        result_zip = find_pipeline_result_zip(result_dir, self.audio_id)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        with zipfile.ZipFile(result_zip) as zf:
            zf.extractall(self.output_dir)
        validate_pipeline_result(self.output_dir, asr_enabled=bool(self.config.get("asr", {}).get("enabled", False)))
        return self.output_dir


def remote_config_for_kaggle(config: dict, audio_name: str) -> dict:
    remote = json.loads(json.dumps(config))
    remote.setdefault("entrypoint", {})["input_path"] = audio_name
    remote["entrypoint"]["output_path"] = "/kaggle/working/outputs"
    remote.setdefault("runtime", {})["backend"] = "local"
    remote["runtime"]["dry_run"] = False
    if remote.get("asr", {}).get("enabled", False):
        remote.setdefault("asr", {})["asr_backend"] = "local"
        if str(remote["asr"].get("device", "cpu")) == "cpu":
            remote["asr"]["device"] = "0"
    for key in ("diarization", "music_separation", "overlap_separation"):
        if str(remote.get(key, {}).get("device", "")) == "cpu":
            remote[key]["device"] = "cuda"
    if remote.get("overlap_separation", {}).get("enabled", False):
        remote["overlap_separation"]["sepreformer_path"] = "SepReFormer"
    if "state_labeling" in remote:
        remote["state_labeling"]["enabled"] = False
    return remote


def validate_pipeline_result(output_dir: Path, asr_enabled: bool) -> None:
    required = ["manifest.timeline.json", "vad.json", "audio.standardized.wav"]
    if asr_enabled:
        required.append("transcript.json")
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Kaggle pipeline result missing required files: {missing}")


def find_pipeline_result_zip(result_dir: Path, audio_id: str) -> Path:
    preferred = sorted(result_dir.glob(f"{audio_id}_*{PIPELINE_RESULT_NAME}"))
    if preferred:
        return preferred[0]
    matches = sorted(result_dir.rglob(f"*{PIPELINE_RESULT_NAME}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No {PIPELINE_RESULT_NAME} found in Kaggle output: {result_dir}")


def split_slug(slug: str) -> tuple[str, str]:
    parts = slug.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Kaggle slug must be owner/slug: {slug}")
    return parts[0], parts[1]


def read_dataset_files_listing(dataset_slug: str, max_pages: int = 25) -> tuple[int, str]:
    pages = []
    page_token = ""
    seen_tokens = set()
    for _ in range(max_pages):
        cmd = ["kaggle", "datasets", "files", dataset_slug, "--page-size", "200"]
        if page_token:
            cmd.extend(["--page-token", page_token])
        result = run_command(cmd, check=False)
        text = (result.stdout + "\n" + result.stderr).strip()
        pages.append(text)
        if result.returncode != 0:
            return result.returncode, "\n".join(page for page in pages if page)
        page_token = extract_next_page_token(text)
        if not page_token or page_token in seen_tokens:
            return 0, "\n".join(page for page in pages if page)
        seen_tokens.add(page_token)
    return 0, "\n".join(page for page in pages if page)


def extract_next_page_token(text: str) -> str:
    for line in text.splitlines():
        if "Next Page Token" not in line:
            continue
        _, _, token = line.partition("=")
        return token.strip()
    return ""


def write_source_archive(target: Path, include_sepreformer: bool, sepreformer_model_name: str) -> None:
    target.write_bytes(source_archive_bytes(include_sepreformer=include_sepreformer, sepreformer_model_name=sepreformer_model_name))


def source_archive_bytes(include_sepreformer: bool, sepreformer_model_name: str) -> bytes:
    project_root = Path(__file__).resolve().parents[1]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relative in SOURCE_FILES:
            path = project_root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            zf.write(path, relative)
        if include_sepreformer:
            add_sepreformer_files(zf, project_root, sepreformer_model_name)
    return buffer.getvalue()


def add_sepreformer_files(zf: zipfile.ZipFile, project_root: Path, model_name: str) -> None:
    root = project_root / "SepReFormer"
    model_dir = root / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"SepReFormer model directory not found: {model_dir}")
    checkpoint_files = sorted((model_dir / "log").glob("*/*"))
    checkpoint_files = [path for path in checkpoint_files if path.is_file() and path.suffix in {".pt", ".pth"}]
    if not checkpoint_files:
        raise FileNotFoundError(f"No SepReFormer checkpoint found for {model_name}")
    for relative in ("LICENSE", "requirements.txt", "utils/decorators.py"):
        path = root / relative
        if path.exists():
            zf.write(path, f"SepReFormer/{relative}")
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            zf.write(path, path.relative_to(project_root).as_posix())


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = resolve_command(cmd)
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def resolve_command(cmd: list[str]) -> list[str]:
    if not cmd or cmd[0] != "kaggle":
        return cmd
    return [kaggle_executable()] + cmd[1:]


def kaggle_executable() -> str:
    configured = os.environ.get("KAGGLE_BIN", "").strip()
    candidates = [configured] if configured else []
    found = shutil.which("kaggle")
    if found:
        candidates.append(found)
    candidates.extend(["/opt/anaconda3/bin/kaggle", "/usr/local/bin/kaggle", "/opt/homebrew/bin/kaggle"])
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Kaggle CLI not found. Set KAGGLE_BIN=/path/to/kaggle or install it with `pip install kaggle`.")


if __name__ == "__main__":
    raise SystemExit(main())
