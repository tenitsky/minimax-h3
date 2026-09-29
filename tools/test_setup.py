"""Exercise real Bash download helpers without network access or model weights."""
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
BASH = os.environ.get("TEST_BASH") or ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
HELPERS = SETUP[SETUP.index("file_ok() {"):SETUP.index("# 3. Model downloads")]


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class DownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="h3-download-")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.fixture = self.directory / "fixture.safetensors"
        header = json.dumps({"data": {"dtype": "U8", "shape": [1000000], "data_offsets": [0, 1000000]}}).encode()
        header += b" " * (-len(header) % 8)
        self.fixture.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 1000000)

    def run_script(self, command):
        script = self.directory / "check.sh"
        preamble = ("set -e\n" + "PY=" + shlex.quote(Path(sys.executable).as_posix()) + "\n"
                    + "FIXTURE=" + shlex.quote(self.fixture.as_posix()) + "\n"
                    + "DEST=" + shlex.quote((self.directory / "model.safetensors").as_posix()) + "\n"
                    + "HF_TOKEN=''\n")
        script.write_text(preamble + HELPERS + "\n" + command, encoding="utf-8", newline="\n")
        result = subprocess.run([BASH, str(script)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_valid_and_truncated_safetensors(self):
        self.run_script('file_ok "$FIXTURE"\n')
        self.fixture.write_bytes(self.fixture.read_bytes()[:-1])
        self.run_script('if file_ok "$FIXTURE"; then exit 1; fi\n')

    def test_successful_transfer_is_promoted(self):
        self.run_script('''
wget() { cp "$FIXTURE" "$DEST.hf.part"; }
fetch_url "$DEST" https://example.invalid/model hf
file_ok "$DEST"
test ! -e "$DEST.hf.part"
''')

    def test_failed_transfer_never_promotes_partial_even_if_large(self):
        self.run_script('''
wget() { cp "$FIXTURE" "$DEST.hf.part"; return 1; }
if fetch_url "$DEST" https://example.invalid/model hf; then exit 1; fi
test ! -e "$DEST"
test -e "$DEST.hf.part"
''')


if __name__ == "__main__":
    unittest.main(verbosity=2)
