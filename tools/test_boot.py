"""Run the complete setup script against a small, offline Linux image fixture.

Filesystem operations and setup control flow are real. Mount metadata, fixed
chmod behavior, downloads, package installation, and the image service handoff
are simulated; this does not replace booting the actual image on RunPod.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
MODEL_NAMES = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
}


@unittest.skipUnless(sys.platform.startswith("linux") and BASH, "Linux Bash integration fixture")
class BootTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="h3-boot-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.comfy = self.workspace / "runpod-slim/ComfyUI"
        self.baked = self.base / "baked"
        (self.baked / "comfy_extras").mkdir(parents=True)
        (self.baked / "main.py").write_text("# image fixture, no GPU service\n", encoding="utf-8")
        (self.baked / "comfy_extras/nodes_minimax_h3.py").write_text(
            'node_id="MiniMaxH3AddGuide"\n', encoding="utf-8")
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.site = self.base / "site"
        self.site.mkdir()
        self.handoff = self.base / "handoff.json"
        self.download_log = self.base / "downloads.log"
        self.apt_log = self.base / "apt.log"
        self.start = self.base / "start.sh"

        # Keep the shipped code and bundled nodes/workflow. Only container paths
        # are rewritten; no installation function or branch is replaced.
        shutil.copytree(ROOT / "custom_nodes", self.repo / "custom_nodes")
        shutil.copytree(ROOT / "workflows", self.repo / "workflows")
        storage = self.repo / "custom_nodes/comfyui-h3-longform/storage.py"
        storage.write_text(storage.read_text(encoding="utf-8").replace("/workspace", str(self.workspace)), encoding="utf-8")
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        for original, replacement in (
            ("/workspace", str(self.workspace)),
            ("/start.sh", str(self.start)),
            ("/opt/comfyui-baked", str(self.baked)),
            ("/tmp/h3-jupyter", str(self.base / "h3-jupyter")),
        ):
            setup = setup.replace(original, replacement)
        self.setup_script = self.repo / "setup.sh"
        self.setup_script.write_text(setup, encoding="utf-8", newline="\n")

        self.fixture = self.base / "fixture.safetensors"
        header = json.dumps({"tensor": {"dtype": "U8", "shape": [1000000], "data_offsets": [0, 1000000]}}).encode()
        header += b" " * (-len(header) % 8)
        self.fixture.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 1000000)

        python = """#!/bin/bash
if [ "${1-}" = -m ] && [ "${2-}" = venv ]; then
  exit 1  # Exercise the supported optional-downloader fallback, never pip/network.
fi
exec """ + shlex.quote(sys.executable) + ' "$@"\n'
        self.make_executable(self.bin / "python3", python)
        self.make_executable(self.bin / "python3.12", python)
        self.make_executable(self.bin / "apt-get", """#!/bin/bash
printf '%s\\n' "$*" >> "$BOOT_APT_LOG"
exit 0
""")
        self.make_executable(self.bin / "ffmpeg", "#!/bin/bash\nexit 0\n")
        self.make_executable(self.bin / "wget", """#!/bin/bash
set -e
destination=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = -O ]; then
    shift
    destination="$1"
  fi
  shift
done
test -n "$destination"
printf '%s\\n' "$destination" >> "$BOOT_DOWNLOAD_LOG"
cp "$BOOT_MODEL_FIXTURE" "$destination"
if [ -n "${BOOT_FAIL_MODEL:-}" ] && [[ "$destination" = *"$BOOT_FAIL_MODEL"* ]]; then
  exit 1  # A full-sized partial must not be promoted after a failed transfer.
fi
""")
        self.make_executable(self.start, """#!/bin/bash
set -e
start_jupyter() {
  if [ "${JUPYTER_DISABLE_AUTH:-}" = "true" ]; then JUPYTER_TOKEN=""; fi
}
# The image service boundary is replaced, not the template's setup script.
python3 - <<'PY'
import json, os
from pathlib import Path
keys = ('COMFYUI_PATH', 'HF_HOME', 'JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR',
        'JUPYTER_RUNTIME_DIR', 'IPYTHONDIR', 'H3_GLOBAL_STORAGE',
        'H3_GLOBAL_MOUNT', 'H3_GLOBAL_ROOT', 'H3_LOCAL_MIN_FREE_GB',
        'JUPYTER_DISABLE_AUTH')
Path(os.environ['BOOT_HANDOFF']).write_text(json.dumps({k: os.environ.get(k) for k in keys}))
PY
""")

        (self.site / "sitecustomize.py").write_text("""import os
from pathlib import Path

workspace = Path(os.environ['BOOT_WORKSPACE']).resolve()
read_text = Path.read_text
chmod = os.chmod

def fixture_read_text(path, *args, **kwargs):
    if path == Path('/proc/self/mountinfo'):
        text = '1 0 0:1 / / rw - overlay overlay rw\\n'
        if os.environ.get('BOOT_MISSING_VOLUME') != '1':
            text += f'2 1 0:2 / {workspace} rw - fuse.vstorage volume rw\\n'
        return text
    return read_text(path, *args, **kwargs)

def fixed_workspace_chmod(path, mode, *args, **kwargs):
    if not isinstance(path, int):
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved == workspace or workspace in resolved.parents:
            mode = 0o777
    return chmod(path, mode, *args, **kwargs)

Path.read_text = fixture_read_text
os.chmod = fixed_workspace_chmod
""", encoding="utf-8", newline="\n")
        # The image's PyTorch, as setup's CUDA 13 check sees it.
        (self.site / "torch").mkdir()
        (self.site / "torch/__init__.py").write_text("""import os
from types import SimpleNamespace

__version__ = "2.10.0+fixture"
version = SimpleNamespace(cuda=os.environ.get("BOOT_TORCH_CUDA", "13.0") or None)
cuda = SimpleNamespace(is_available=lambda: os.environ.get("BOOT_GPU", "1") == "1",
                       get_device_name=lambda index: "Fixture GPU")
""", encoding="utf-8", newline="\n")

        self.env = dict(os.environ)
        for key in ("COMFYUI_PATH", "HF_HOME", "HF_TOKEN", "H3_TEXT_ENCODER",
                    "DOWNLOAD_TURBO_LORA", "DOWNLOAD_REF2VA", "JUPYTER_NO_AUTH"):
            self.env.pop(key, None)
        self.env.update({
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(self.site),
            "PYTHONDONTWRITEBYTECODE": "1",
            "BOOT_WORKSPACE": str(self.workspace),
            "BOOT_HANDOFF": str(self.handoff),
            "BOOT_DOWNLOAD_LOG": str(self.download_log),
            "BOOT_APT_LOG": str(self.apt_log),
            "BOOT_MODEL_FIXTURE": str(self.fixture),
            "H3_GLOBAL_STORAGE": "1",
            "H3_GLOBAL_MOUNT": "/missing-global",
            "H3_GLOBAL_ROOT": "/missing-global/minimax-h3",
            "H3_LOCAL_MIN_FREE_GB": "999999",
        })

    @staticmethod
    def make_executable(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def boot(self, **overrides):
        self.handoff.unlink(missing_ok=True)
        return subprocess.run([BASH, str(self.setup_script)], env={**self.env, **overrides},
                              text=True, capture_output=True, timeout=60)

    def downloads(self):
        return self.download_log.read_text().splitlines() if self.download_log.exists() else []

    def test_first_boot_and_restart_preserve_user_files_and_reuse_models(self):
        first = self.boot()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertTrue(self.handoff.exists(), first.stdout + first.stderr)
        self.assertIn("PyTorch 2.10.0+fixture, CUDA build 13.0", first.stdout)
        self.assertIn("GPU: Fixture GPU", first.stdout)
        self.assertEqual(len(self.downloads()), 7)
        self.assertEqual({p.name for p in (self.comfy / "models").rglob("*.safetensors")}, MODEL_NAMES)
        self.assertTrue((self.comfy / "custom_nodes/comfyui-h3-longform/__init__.py").is_file())
        installed_workflow = self.comfy / "user/default/workflows/minimax_h3_talking_head.json"
        self.assertEqual(json.loads(installed_workflow.read_text()),
                         json.loads((ROOT / "workflows/minimax_h3_talking_head.json").read_text()))
        installed_fast = self.comfy / "user/default/workflows/minimax_h3_talking_head_fast.json"
        self.assertEqual(json.loads(installed_fast.read_text()),
                         json.loads((ROOT / "workflows/minimax_h3_talking_head_fast.json").read_text()))
        installed_draft = self.comfy / "user/default/workflows/minimax_h3_talking_head_draft.json"
        self.assertEqual(json.loads(installed_draft.read_text()),
                         json.loads((ROOT / "workflows/minimax_h3_talking_head_draft.json").read_text()))

        state = json.loads(self.handoff.read_text())
        self.assertEqual(state["COMFYUI_PATH"], str(self.comfy))
        self.assertEqual(state["JUPYTER_DISABLE_AUTH"], "true")
        self.assertIn('JUPYTER_DISABLE_AUTH', self.start.read_text())
        runtime = Path(state["JUPYTER_RUNTIME_DIR"])
        self.assertIn(self.base, runtime.parents)
        self.assertNotIn(self.workspace, runtime.parents)
        self.assertEqual(runtime.stat().st_mode & 0o777, 0o700)
        for key in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "IPYTHONDIR", "HF_HOME"):
            self.assertIn(self.workspace, Path(state[key]).parents)
        for key in ("H3_GLOBAL_STORAGE", "H3_GLOBAL_MOUNT", "H3_GLOBAL_ROOT", "H3_LOCAL_MIN_FREE_GB"):
            self.assertIsNone(state[key])

        edited_workflow = '{"edited_by_user": true}\n'
        installed_workflow.write_text(edited_workflow)
        edited_fast = '{"fast_edited_by_user": true}\n'
        installed_fast.write_text(edited_fast)
        edited_draft = '{"draft_edited_by_user": true}\n'
        installed_draft.write_text(edited_draft)
        notebook = self.workspace / "my-notebook.ipynb"
        notebook.write_text('{"cells": [], "metadata": {"keep": true}}\n')
        notebook_bytes = notebook.read_bytes()
        broken_log = self.base / "broken-interpreter-called"
        self.make_executable(self.comfy / ".venv/bin/python",
                             '#!/bin/bash\nprintf called > ' + shlex.quote(str(broken_log)) + '\nexit 97\n')

        restart = self.boot()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertTrue(self.handoff.exists())
        self.assertEqual(len(self.downloads()), 7, "Completed models must not be downloaded again")
        self.assertFalse(broken_log.exists(), "Model validation must use the working image interpreter")
        self.assertEqual(installed_workflow.read_text(), edited_workflow)
        self.assertEqual(installed_fast.read_text(), edited_fast)
        self.assertEqual(installed_draft.read_text(), edited_draft)
        self.assertEqual(notebook.read_bytes(), notebook_bytes)
        self.assertEqual(list(self.workspace.glob(".h3-posix-*")), [])

    def test_missing_volume_stops_before_installation(self):
        result = self.boot(BOOT_MISSING_VOLUME="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No separate volume mounted", result.stderr)
        self.assertFalse(self.comfy.exists())
        self.assertFalse(self.handoff.exists())
        self.assertEqual(self.downloads(), [])
        self.assertFalse(self.apt_log.exists())

    def test_cuda_12_image_stops_before_downloads(self):
        result = self.boot(BOOT_TORCH_CUDA="12.8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs the CUDA 13 image", result.stderr)
        self.assertEqual(self.downloads(), [])
        self.assertFalse(self.handoff.exists())

    def test_old_driver_host_stops_before_downloads(self):
        result = self.boot(BOOT_GPU="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("host driver is older than r580", result.stderr)
        self.assertEqual(self.downloads(), [])
        self.assertFalse(self.handoff.exists())

    def test_failed_model_transfer_never_hands_off_to_services(self):
        failed_model = "minimax_h3_video_vae_int8_convrot.safetensors"
        result = self.boot(BOOT_FAIL_MODEL=failed_model)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL: one or more model downloads failed", result.stdout)
        self.assertFalse(self.handoff.exists())
        self.assertFalse((self.comfy / "models/vae" / failed_model).exists())
        self.assertTrue((self.comfy / "models/vae" / (failed_model + ".hf.part")).exists())
        self.assertTrue((self.comfy / "models/vae" / (failed_model + ".ms.part")).exists())
        self.assertEqual(len(self.downloads()), 8)
        self.assertEqual(len(list((self.comfy / "models").rglob("*.safetensors"))), 6)

    def test_existing_incomplete_comfy_directory_is_not_deleted(self):
        self.comfy.mkdir(parents=True)
        saved_file = self.comfy / "keep-my-data.txt"
        saved_file.write_text("existing user data")
        result = self.boot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to delete user files", result.stdout)
        self.assertEqual(saved_file.read_text(), "existing user data")
        self.assertFalse((self.comfy / "main.py").exists())
        self.assertFalse(self.handoff.exists())
        self.assertEqual(self.downloads(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
