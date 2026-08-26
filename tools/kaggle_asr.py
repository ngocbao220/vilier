import argparse
import base64
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


TERMINAL_SUCCESS = ("complete", "completed", "success")
TERMINAL_FAILURE = ("error", "failed", "failure", "cancel", "canceled")
SOURCE_ARCHIVE_NAME = "vilier_source.zip"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
SOURCE_FILES = (
    "config.json",
    "tools/run_asr_bundle.py",
    "pipeline/__init__.py",
    "pipeline/asr.py",
    "pipeline/audio.py",
    "pipeline/schema.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a Vilier ASR bundle to Kaggle, run GPU ASR, and download transcript.json")
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True, help="Local output/<audio_id> directory where transcript.json will be written")
    parser.add_argument("--dataset-slug", default=os.environ.get("KAGGLE_DATASET_SLUG", "ngocbaotrinhtuan/vilier-asr-bundle"))
    parser.add_argument("--kernel-slug", default=os.environ.get("KAGGLE_KERNEL_SLUG", "ngocbaotrinhtuan/vilier-phowhisper-asr"))
    parser.add_argument("--repo-url", default=os.environ.get("KAGGLE_REPO_URL", ""))
    parser.add_argument("--repo-ref", default=os.environ.get("KAGGLE_REPO_REF", "main"))
    parser.add_argument("--accelerator", default=os.environ.get("KAGGLE_ACCELERATOR", DEFAULT_ACCELERATOR))
    parser.add_argument("--work-dir", default=os.environ.get("KAGGLE_ASR_WORK_DIR", ".kaggle_asr_work"))
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("KAGGLE_POLL_SECONDS", "30")))
    parser.add_argument("--max-wait-seconds", type=float, default=float(os.environ.get("KAGGLE_MAX_WAIT_SECONDS", "21600")))
    parser.add_argument("--dataset-ready-seconds", type=float, default=float(os.environ.get("KAGGLE_DATASET_READY_SECONDS", "240")))
    parser.add_argument("--dry-run", action="store_true", help="Exercise local orchestration without calling Kaggle")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    validate_bundle(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    runner = KaggleAsrRunner(
        audio_id=args.audio_id,
        bundle=bundle,
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
    transcript_path = runner.run()
    print(f"transcript={transcript_path}")
    return 0


class KaggleAsrRunner:
    def __init__(
        self,
        audio_id: str,
        bundle: Path,
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
        self.bundle = bundle
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
        self.wait_for_kernel()
        print("Đã có kết quả, đang tải transcript xuống", flush=True)
        result_dir = self.download_output()
        return self.merge_transcript(result_dir)

    def _dry_run(self) -> Path:
        print("Đã có kết quả, đang tải transcript xuống", flush=True)
        manifest = _read_json_from_zip(self.bundle, "manifest.timeline.json")
        transcript = []
        asr_segments = manifest.get("asr_segments") or []
        if asr_segments:
            source_segments = asr_segments
        else:
            source_segments = []
            for idx, vad in enumerate(manifest.get("vad_segments", []), start=1):
                item = dict(vad)
                item["audio"] = (manifest.get("vad_audio") or [f"vad_audio/audio_{idx}.wav"])[idx - 1]
                item["speaker"] = "SPEAKER_00"
                source_segments.append(item)
        for idx, segment in enumerate(source_segments, start=1):
            start = round(float(segment.get("start", 0.0)), 3)
            end = round(float(segment.get("end", 0.0)), 3)
            record = {
                "id": f"asr_{idx - 1:05d}",
                "audio": str(segment.get("audio", "")),
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "speaker": str(segment.get("speaker", "SPEAKER_UNKNOWN")),
                "text": f"dry-run kaggle transcript {idx}",
                "model": "vinai/PhoWhisper-large",
                "language": "vi",
            }
            if asr_segments:
                record["asr_segment_id"] = str(segment.get("id", ""))
            else:
                record["vad_id"] = str(segment.get("id", ""))
            transcript.append(
                record
            )
        transcript_path = self.output_dir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        validate_transcript(transcript_path, self.output_dir / "manifest.timeline.json")
        return transcript_path

    def prepare_dataset_dir(self) -> Path:
        dataset_dir = self.work_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True)
        shutil.copy2(self.bundle, dataset_dir / self.bundle.name)
        write_source_archive(dataset_dir / SOURCE_ARCHIVE_NAME)
        owner, slug = split_slug(self.dataset_slug)
        metadata = {
            "title": f"Vilier ASR bundle {self.audio_id}",
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
            "title": "Vilier PhoWhisper ASR",
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
        embedded_source_zip = base64.b64encode(source_archive_bytes()).decode("ascii")
        return textwrap.dedent(
            f"""
            import base64
            from pathlib import Path
            import os
            import shutil
            import subprocess
            import sys
            import zipfile

            AUDIO_ID = {self.audio_id!r}
            REPO_URL = {self.repo_url!r}
            REPO_REF = {self.repo_ref!r}
            WORK_ROOT = Path("/kaggle/working")
            INPUT_ROOT = Path("/kaggle/input")
            SOURCE_DIR = WORK_ROOT / "vilier_source"
            BUNDLE_DIR = WORK_ROOT / "asr_bundle" / AUDIO_ID
            ASR_OUTPUT_DIR = WORK_ROOT / "asr_result"
            SOURCE_ARCHIVE_NAME = {SOURCE_ARCHIVE_NAME!r}
            EMBEDDED_SOURCE_ZIP = {embedded_source_zip!r}

            def find_bundle_source():
                matches = sorted(INPUT_ROOT.rglob(f"{{AUDIO_ID}}*_asr_bundle.zip"))
                if not matches:
                    matches = sorted(path for path in INPUT_ROOT.rglob("*.zip") if path.name != SOURCE_ARCHIVE_NAME)
                if matches:
                    return ("zip", matches[0])
                manifest_matches = sorted(
                    path.parent
                    for path in INPUT_ROOT.rglob("manifest.timeline.json")
                    if AUDIO_ID in str(path.parent)
                )
                if not manifest_matches:
                    visible = [str(path.relative_to(INPUT_ROOT)) for path in sorted(INPUT_ROOT.rglob("*"))[:200]]
                    raise FileNotFoundError(f"No ASR bundle zip or extracted manifest found under {{INPUT_ROOT}}. Visible inputs: {{visible}}")
                return ("dir", manifest_matches[0])

            def prepare_source():
                source_archives = sorted(INPUT_ROOT.rglob(SOURCE_ARCHIVE_NAME))
                if source_archives:
                    if SOURCE_DIR.exists():
                        shutil.rmtree(SOURCE_DIR)
                    SOURCE_DIR.mkdir(parents=True)
                    with zipfile.ZipFile(source_archives[0]) as zf:
                        zf.extractall(SOURCE_DIR)
                    return SOURCE_DIR
                if EMBEDDED_SOURCE_ZIP:
                    if SOURCE_DIR.exists():
                        shutil.rmtree(SOURCE_DIR)
                    SOURCE_DIR.mkdir(parents=True)
                    source_zip_path = WORK_ROOT / SOURCE_ARCHIVE_NAME
                    source_zip_path.write_bytes(base64.b64decode(EMBEDDED_SOURCE_ZIP))
                    with zipfile.ZipFile(source_zip_path) as zf:
                        zf.extractall(SOURCE_DIR)
                    return SOURCE_DIR
                if not REPO_URL:
                    raise FileNotFoundError(f"No {{SOURCE_ARCHIVE_NAME}} found under {{INPUT_ROOT}} and REPO_URL is empty")
                repo_dir = WORK_ROOT / "vilier"
                if repo_dir.exists():
                    shutil.rmtree(repo_dir)
                subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
                subprocess.run(["git", "-C", str(repo_dir), "checkout", REPO_REF], check=True)
                return repo_dir

            REPO_DIR = prepare_source()
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "soundfile", "librosa", "pyyaml"], check=True)

            if BUNDLE_DIR.exists():
                shutil.rmtree(BUNDLE_DIR)
            BUNDLE_DIR.mkdir(parents=True)
            bundle_kind, bundle_source = find_bundle_source()
            if bundle_kind == "zip":
                with zipfile.ZipFile(bundle_source) as zf:
                    zf.extractall(BUNDLE_DIR)
            else:
                shutil.copytree(bundle_source, BUNDLE_DIR, dirs_exist_ok=True)

            if ASR_OUTPUT_DIR.exists():
                shutil.rmtree(ASR_OUTPUT_DIR)
            ASR_OUTPUT_DIR.mkdir(parents=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_DIR)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_DIR / "tools" / "run_asr_bundle.py"),
                    "--config",
                    str(REPO_DIR / "config.json"),
                    "--bundle-dir",
                    str(BUNDLE_DIR),
                    "--output-dir",
                    str(ASR_OUTPUT_DIR),
                    "--device",
                    "0",
                ],
                check=True,
                env=env,
            )

            result_zip = WORK_ROOT / f"{{AUDIO_ID}}_asr_result.zip"
            with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in ASR_OUTPUT_DIR.rglob("*"):
                    if item.is_file():
                        zf.write(item, item.relative_to(ASR_OUTPUT_DIR).as_posix())
            print("result_zip", result_zip)
            """
        ).strip() + "\n"

    def upload_dataset(self, dataset_dir: Path) -> None:
        metadata_dir = self.work_dir / "dataset_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        exists = run_command(["kaggle", "datasets", "metadata", self.dataset_slug, "-p", str(metadata_dir)], check=False).returncode == 0
        if exists:
            run_command(["kaggle", "datasets", "version", "-p", str(dataset_dir), "-m", f"Vilier ASR bundle {self.audio_id}"], check=True)
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
        bundle_ready_tokens = (self.bundle.name, f"{self.bundle.stem}/manifest.timeline.json", f"{self.bundle.stem}/asr_audio/")
        started = time.monotonic()
        last_text = ""
        saw_successful_listing = False
        while True:
            returncode, last_text = read_dataset_files_listing(self.dataset_slug)
            if returncode == 0:
                saw_successful_listing = True
            if returncode == 0 and any(token in last_text for token in bundle_ready_tokens):
                return
            if time.monotonic() - started > self.dataset_ready_seconds:
                if saw_successful_listing:
                    print(
                        "Kaggle dataset listing is paginated or delayed; continuing to kernel push after successful listing response",
                        flush=True,
                    )
                    return
                raise TimeoutError(
                    "Timed out waiting for Kaggle dataset bundle "
                    f"{self.bundle.name} or extracted folder {self.bundle.stem}/ in {self.dataset_slug}.\nLast response:\n{last_text}"
                )
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

    def merge_transcript(self, result_dir: Path) -> Path:
        transcript = find_transcript(result_dir, self.audio_id)
        target = self.output_dir / "transcript.json"
        shutil.copy2(transcript, target)
        validate_transcript(target, self.output_dir / "manifest.timeline.json")
        return target


def validate_bundle(bundle: Path) -> None:
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        missing = {"manifest.timeline.json", "vad.json"} - names
        if missing:
            raise ValueError(f"Bundle missing required entries: {sorted(missing)}")
        manifest = json.loads(zf.read("manifest.timeline.json").decode("utf-8"))
        if manifest.get("asr_segments"):
            if not any(name.startswith("asr_audio/") and name.endswith(".wav") for name in names):
                raise ValueError("Bundle has no asr_audio/*.wav entries")
        elif not any(name.startswith("vad_audio/") and name.endswith(".wav") for name in names):
            raise ValueError("Bundle has no vad_audio/*.wav entries")


def validate_transcript(transcript_path: Path, manifest_path: Path) -> None:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = len(manifest.get("asr_segments") or manifest.get("vad_segments", []))
    if expected and len(transcript) != expected:
        raise ValueError(f"Transcript count mismatch: got {len(transcript)}, expected {expected}")
    id_key = "asr_segment_id" if manifest.get("asr_segments") else "vad_id"
    required = {"id", id_key, "audio", "start", "end", "speaker", "text", "model", "language"}
    bad = [idx for idx, item in enumerate(transcript) if not required.issubset(item)]
    if bad:
        raise ValueError(f"Transcript records missing keys at indexes: {bad[:10]}")


def find_transcript(result_dir: Path, audio_id: str) -> Path:
    direct = result_dir / "transcript.json"
    if direct.exists():
        return direct
    for zip_path in sorted(result_dir.glob("*.zip")):
        extract_dir = result_dir / f"{zip_path.stem}_unzipped"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        found = sorted(extract_dir.rglob("transcript.json"))
        if found:
            return found[0]
    found = sorted(result_dir.rglob("transcript.json"))
    if found:
        return found[0]
    raise FileNotFoundError(f"No transcript.json found in Kaggle output for {audio_id}: {result_dir}")


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


def write_source_archive(target: Path) -> None:
    target.write_bytes(source_archive_bytes())


def source_archive_bytes() -> bytes:
    project_root = Path(__file__).resolve().parents[1]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relative in SOURCE_FILES:
            path = project_root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            zf.write(path, relative)
    return buffer.getvalue()


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
    raise FileNotFoundError(
        "Kaggle CLI not found. Set KAGGLE_BIN=/path/to/kaggle or install it with `pip install kaggle`."
    )


def _read_json_from_zip(bundle: Path, name: str):
    with zipfile.ZipFile(bundle) as zf:
        return json.loads(zf.read(name).decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
