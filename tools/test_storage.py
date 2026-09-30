"""Standard-library persistence tests: python tools/test_storage.py."""
import importlib.util
import asyncio
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("storage", Path(__file__).resolve().parents[1] / "custom_nodes/comfyui-h3-longform/storage.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="h3-storage-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.mount = self.base / "global"
        self.remote = self.mount / "minimax-h3"
        self.local = self.base / "pod1"
        self.newpod = self.base / "pod2"
        self.local.mkdir()
        env = patch.dict(os.environ, {"H3_GLOBAL_MOUNT": str(self.mount), "H3_GLOBAL_ROOT": str(self.remote)})
        env.start()
        self.addCleanup(env.stop)
        mounted = patch.object(storage.os.path, "ismount", return_value=True)
        mounted.start()
        self.addCleanup(mounted.stop)

    def write(self, path, content=b"complete video"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_new_pod_restores_verified_chunk_tail_workflow_and_final(self):
        session = "speaker's take"
        chunk = Path("h3_longform") / session / "chunk_0000.mp4"
        tail = Path("h3_longform") / session / ".carry/tail_0000.npz"
        self.write(self.local / chunk)
        self.write(self.local / tail, b"tail frames")
        self.write(self.local / "final.mp4", b"final video")
        storage.backup_workflow(self.local, session, {"workflow": {"nodes": [1]}})
        storage.backup_chunk(self.local, session, 0, True)
        storage.backup_final(self.local, "final.mp4")
        # A partial upload with no completion receipt must not become a chunk.
        self.write(self.remote / "output/h3_longform" / session / "chunk_0001.mp4", b"partial")
        storage.restore(self.newpod)
        for relative in [chunk, tail, Path("final.mp4"), Path("h3_longform") / session / "workflow.json"]:
            self.assertEqual((self.newpod / relative).read_bytes(), (self.local / relative).read_bytes())
        self.assertFalse((self.newpod / chunk.parent / "chunk_0001.mp4").exists())
        storage.restore(self.newpod)  # idempotent reboot

    def test_interrupted_copy_invalidates_old_receipt_and_retries(self):
        source = self.write(self.local / "video.mp4", b"old video")
        destination = self.remote / "video.mp4"
        storage.publish(source, destination)
        source.write_bytes(b"new video")
        def fail_copy(src, dst):
            Path(dst).write_bytes(b"new")
            raise OSError("upload interrupted")
        with patch.object(storage.shutil, "copyfile", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "interrupted"):
                storage.publish(source, destination)
        self.assertFalse(storage.valid(destination))
        storage.publish(source, destination)
        self.assertTrue(storage.valid(destination))
        with patch.object(storage.shutil, "copyfile", side_effect=AssertionError("duplicate upload")):
            storage.publish(source, destination)

    def test_corruption_and_truncated_receipts_are_ignored(self):
        source = self.write(self.local / "video.mp4")
        destination = self.remote / "output/video.mp4"
        storage.publish(source, destination)
        destination.write_bytes(b"x" * source.stat().st_size)
        storage.restore(self.newpod)
        self.assertFalse((self.newpod / "video.mp4").exists())
        storage.receipt_path(destination).write_text('{"size":')
        self.assertFalse(storage.valid(destination))

    def test_mount_loss_and_missing_tail_stop_backup(self):
        with patch.object(storage.os.path, "ismount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no longer mounted"):
                storage.backup_final(self.local, "final.mp4")
        self.write(self.local / "h3_longform/test/chunk_0000.mp4")
        with self.assertRaises(FileNotFoundError):
            storage.backup_chunk(self.local, "test", 0, True)
        self.assertFalse((self.remote / "output/h3_longform/test/chunk_0000.mp4").exists())

    def test_existing_local_render_is_not_overwritten(self):
        source = self.write(self.local / "final.mp4", b"old")
        storage.backup_final(self.local, "final.mp4")
        self.write(self.newpod / "final.mp4", b"new")
        storage.restore(self.newpod)
        self.assertEqual((self.newpod / "final.mp4").read_bytes(), b"new")

    def test_path_escape_is_rejected(self):
        for path in ("../elsewhere", str(self.base / "elsewhere")):
            with self.assertRaises(ValueError):
                storage.within(self.remote, path)

    def test_without_global_volume_backup_is_noop(self):
        with patch.dict(os.environ, {"H3_GLOBAL_ROOT": ""}):
            storage.backup_chunk(self.local, "unused", 0, True)
            storage.backup_final(self.local, "missing.mp4")
            storage.restore(self.newpod)
        self.assertFalse(self.remote.exists())

    def require_symlinks(self):
        target = self.write(self.base / "link-target")
        link = self.base / "test-link"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"OS does not permit symlinks: {error}")
        link.unlink()

    def test_models_link_to_global_and_reboot_is_idempotent(self):
        self.require_symlinks()
        relative = "diffusion_models/test.safetensors"
        source = self.write(self.local / "models" / relative, b"model weights")
        storage.publish(source, storage.model_path(relative))
        storage.link_model(self.local, relative)
        storage.link_model(self.local, relative)
        self.assertTrue(source.is_symlink())
        self.assertEqual(source.read_bytes(), b"model weights")
        storage.link_model(self.newpod, relative)
        self.assertEqual((self.newpod / "models" / relative).read_bytes(), b"model weights")

    def test_input_and_workflow_migration_and_new_pod(self):
        self.require_symlinks()
        self.write(self.local / "input/portrait.png", b"portrait")
        self.write(self.local / "user/default/workflows/my.json", b"{}")
        self.newpod.mkdir()
        with patch.dict(os.environ, {"H3_LOCAL_MIN_FREE_GB": "0"}):
            storage.configure(self.local)
            storage.configure(self.local)
            storage.configure(self.newpod)
        self.assertEqual((self.newpod / "input/portrait.png").read_bytes(), b"portrait")
        self.assertEqual((self.newpod / "user/default/workflows/my.json").read_bytes(), b"{}")
        self.assertTrue((self.local / "input.before-global/portrait.png").exists())
        self.assertFalse((self.local / "output").is_symlink())
        self.assertFalse((self.local / "user/default/workflows").is_symlink())
        # Match ComfyUI's actual save operation: mkstemp + atomic local replace.
        workflow = self.local / "user/default/workflows/my.json"
        temporary = workflow.with_suffix(".tmp")
        temporary.write_text('{"updated": true}')
        os.replace(temporary, workflow)
        storage.backup_workflows(self.local)
        self.assertTrue(storage.valid(self.remote / "workflows/my.json"))

    def test_old_workflow_symlink_is_migrated_to_local_edits(self):
        self.require_symlinks()
        self.write(self.remote / "workflows/legacy.json", b'{"nodes": []}')
        local = self.local / "user/default/workflows"
        local.parent.mkdir(parents=True)
        local.symlink_to(self.remote / "workflows", target_is_directory=True)
        storage.local_workflows(self.local)
        self.assertFalse(local.is_symlink())
        self.assertEqual((local / "legacy.json").read_bytes(), b'{"nodes": []}')
        self.assertTrue(storage.valid(self.remote / "workflows/legacy.json"))

    def test_same_size_corruption_is_repaired_on_republish(self):
        source = self.write(self.local / "file", b"good")
        remote = self.remote / "file"
        storage.publish(source, remote)
        remote.write_bytes(b"oops")
        storage.publish(source, remote)
        self.assertTrue(storage.valid(remote))

    def test_working_directory_rejects_object_filesystems(self):
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value="1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / /workspace rw - fuse.global global rw\n"):
            if os.name != "nt":
                with self.assertRaisesRegex(RuntimeError, "fuse.global"):
                    storage.check_working_path("/workspace/runpod-slim/ComfyUI")
        with self.assertRaisesRegex(RuntimeError, "on Global Storage"):
            storage.check_working_path(self.remote / "ComfyUI")

    def test_visibility_delay_is_retried(self):
        with patch.object(storage.time, "sleep") as sleep:
            attempts = iter([False, False, True])
            storage.wait_for(lambda: next(attempts), "delayed file")
            self.assertEqual(sleep.call_count, 2)

    def test_workspace_requires_its_own_mount(self):
        # ismount() is mocked true by the fixture; it must not bypass mountinfo.
        for mounts in ("1 0 0:1 / / rw - overlay overlay rw\n",
                       "1 0 0:1 / / rw - ext4 /dev/root rw\n",
                       f"2 1 0:2 / {self.newpod.as_posix()} rw - nfs4 server:/data rw\n"):
            with self.subTest(mounts=mounts), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts):
                with self.assertRaisesRegex(RuntimeError, "disposable container disk"):
                    storage.require_workspace_volume(self.local)
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                storage.require_workspace_volume(self.local)

    def test_workspace_rejects_ephemeral_and_object_mounts(self):
        for kind in ("overlay", "tmpfs", "ramfs", "fuse.global", "s3fs", "fuse.geesefs", "fuse.rclone", "fuse.gcsfuse"):
            mounts = f"2 1 0:2 / {self.local.as_posix()} rw - {kind} volume rw\n"
            with self.subTest(kind=kind), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts):
                with self.assertRaisesRegex(RuntimeError, "persistent POSIX working volume"):
                    storage.require_workspace_volume(self.local)

    def test_workspace_accepts_network_and_bind_mounts(self):
        for kind in ("nfs", "nfs4", "ext4", "xfs", "fuse", "fuse.juicefs", "fuse.vstorage"):
            # Bind mounts may share the parent device; the mountinfo entry still
            # identifies the separate volume. Also exercise escaped mount paths.
            workspace = self.local / "path with spaces"
            target = workspace.as_posix().replace(" ", "\\040")
            mounts = f"2 1 0:1 /volume/subdir {target} rw - {kind} volume rw\n"
            with self.subTest(kind=kind), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts), patch.object(storage, "probe_working_volume") as probe:
                storage.require_workspace_volume(workspace)
                storage.check_working_path(workspace / "ComfyUI/output")
                probe.assert_called_once_with(workspace.resolve())

    @unittest.skipIf(os.name == "nt", "Linux POSIX filesystem semantics required")
    def test_working_volume_probe_and_cleanup(self):
        preserved = self.write(self.local / "notebook.ipynb", b"existing user file")
        storage.probe_working_volume(self.local)
        self.assertEqual(preserved.read_bytes(), b"existing user file")
        self.assertEqual(list(self.local.iterdir()), [preserved])
        for operation in ("symlink_to", "chmod"):
            with self.subTest(operation=operation), patch.object(Path, operation, side_effect=OSError("not supported")):
                with self.assertRaisesRegex(RuntimeError, "failed the POSIX check"):
                    storage.probe_working_volume(self.local)
            self.assertEqual(list(self.local.iterdir()), [preserved])
        with patch.object(storage.os, "replace", side_effect=OSError("not supported")):
            with self.assertRaisesRegex(RuntimeError, "failed the POSIX check"):
                storage.probe_working_volume(self.local)
        self.assertEqual(list(self.local.iterdir()), [preserved])

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp required for middleware tests")
    def test_ui_save_backs_up_and_reports_backup_failure(self):
        from aiohttp import web
        middlewares = []
        server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(app=types.SimpleNamespace(middlewares=middlewares))))
        with patch.dict(sys.modules, {"server": server}), patch.dict(os.environ, {"COMFYUI_PATH": str(self.local)}):
            storage.install_workflow_backup()
            workflow = self.local / "user/default/workflows/test.json"

            async def handler(request):
                self.write(workflow, b'{"nodes": []}')
                return web.json_response({"saved": True})

            for path in ("/userdata/workflows/test.json", "/api/userdata/workflows/test.json"):
                request = types.SimpleNamespace(method="POST", path=path)
                response = asyncio.run(middlewares[0](request, handler))
                self.assertEqual(response.status, 200)
                self.assertTrue(storage.valid(self.remote / "workflows/test.json"))
            with patch.object(storage, "backup_workflows", side_effect=OSError("unavailable")):
                with self.assertRaises(web.HTTPServiceUnavailable):
                    asyncio.run(middlewares[0](request, handler))
                self.assertTrue(workflow.exists())

    def test_migration_conflicts_and_space_guard(self):
        self.write(self.local / "input/file.png", b"local")
        self.write(self.remote / "input/file.png", b"remote")
        with self.assertRaisesRegex(RuntimeError, "migration conflict"):
            storage.link_directory(self.local / "input", self.remote / "input")
        self.assertEqual((self.local / "input/file.png").read_bytes(), b"local")
        with patch.dict(os.environ, {"H3_LOCAL_MIN_FREE_GB": "10000000"}):
            with self.assertRaisesRegex(RuntimeError, "working disk space"):
                storage.configure(self.local)


if __name__ == "__main__":
    unittest.main(verbosity=2)
