"""Global Volume persistence using closed-file copies and checksum receipts.

No remote rename, locks, append, chmod, or background-only shutdown sync. A receipt
is written last; interrupted copies are rejected on restore. One writing Pod per
Global Volume/template namespace is supported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import threading
import uuid

LOG = "[H3 Storage]"
_workflow_lock = threading.Lock()


def check_working_path(path):
    """Reject object-backed scratch, including a Global Volume at /workspace."""
    path = Path(path).resolve()
    mount = Path(os.environ.get("H3_GLOBAL_MOUNT", "/workspace-global")).resolve()
    if path == mount or mount in path.parents:
        raise RuntimeError(f"Working path {path} is on Global Storage. Mount the Global Volume at /workspace-global and keep /workspace local.")
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        matches = []
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            target = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
            if target == path or target in path.parents:
                matches.append((len(str(target)), fields[fields.index("-") + 1]))
        kind = max(matches, default=(0, ""))[1].lower()
        if any(name in kind for name in ("fuse", "s3fs", "gcs", "rclone")):
            raise RuntimeError(f"Working path {path} uses {kind}. Set the Global Volume mount to /workspace-global; /workspace must be a local or regional working disk.")


def wait_for(check, description, attempts=5):
    """Allow short visibility delays without treating an incomplete copy as done."""
    for attempt in range(attempts):
        try:
            if check():
                return
        except OSError:
            pass
        if attempt + 1 < attempts:
            time.sleep(2)
    raise IOError(f"Global Storage verification failed: {description}. Local files were retained; retry once storage is available.")


def root():
    configured = os.environ.get("H3_GLOBAL_ROOT")
    if not configured:
        return None
    mount = Path(os.environ.get("H3_GLOBAL_MOUNT", "/workspace-global"))
    if not os.path.ismount(mount):
        raise RuntimeError(f"Global Volume is no longer mounted at {mount}; refusing a local-only backup.")
    result = Path(configured).resolve()
    if result == mount.resolve() or mount.resolve() not in result.parents:
        raise RuntimeError("H3_GLOBAL_ROOT must be a subfolder of H3_GLOBAL_MOUNT.")
    return result


def within(base, relative):
    base = Path(base).resolve()
    target = (base / relative).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"Path must stay inside {base}: {relative}")
    return target


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt_path(path):
    return Path(str(path) + ".h3-ready")


def valid(path, full=True):
    path = Path(path)
    try:
        info = json.loads(receipt_path(path).read_text(encoding="utf-8"))
        return (info["size"] == path.stat().st_size
                and (not full or info["sha256"] == digest(path)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish(source, destination):
    """Return only after the copy is closed, read back, and marked complete."""
    source, destination = Path(source), Path(destination)
    info = {"size": source.stat().st_size, "sha256": digest(source)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate the old receipt before overwriting a previously completed file.
    receipt = receipt_path(destination)
    try:
        if json.loads(receipt.read_text(encoding="utf-8")) == info and valid(destination):
            return
    except (OSError, ValueError):
        pass
    if receipt.exists():
        receipt.unlink()
    shutil.copyfile(source, destination)
    wait_for(lambda: destination.stat().st_size == info["size"] and digest(destination) == info["sha256"], str(destination))
    receipt.write_text(json.dumps(info) + "\n", encoding="utf-8")
    wait_for(lambda: valid(destination, full=False), str(receipt))


def backup_chunk(output_dir, session, index, carry=False):
    remote = root()
    if remote is None:
        return
    local = Path(output_dir)
    relative = Path("h3_longform") / session
    # Tail first: a completed remote chunk must also have its motion lead-in.
    names = [relative / ".carry" / f"tail_{index:04d}.npz"] if carry else []
    names.append(relative / f"chunk_{index:04d}.mp4")
    for name in names:
        publish(within(local, name), within(remote / "output", name))
    print(f"{LOG} chunk {index + 1} backed up and verified")


def backup_final(output_dir, filename):
    remote = root()
    if remote is not None:
        publish(within(output_dir, filename), within(remote / "output", filename))
        print(f"{LOG} final video backed up and verified: {remote / 'output' / filename}")


def backup_workflow(output_dir, session, extra_pnginfo):
    remote = root()
    workflow = (extra_pnginfo or {}).get("workflow")
    if remote is None or not isinstance(workflow, dict):
        return
    relative = Path("h3_longform") / session / "workflow.json"
    local = within(output_dir, relative)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    publish(local, within(remote / "output", relative))


def restore(output_dir, saved=None):
    remote = root()
    if remote is None:
        return
    saved = Path(saved) if saved is not None else remote / "output"
    restored = 0
    for receipt in sorted(saved.rglob("*.h3-ready")):
        source = Path(str(receipt)[:-len(".h3-ready")])
        relative = source.relative_to(saved)
        destination = within(output_dir, relative)
        if not valid(source):
            print(f"{LOG} ignoring incomplete/corrupt backup: {relative}")
            continue
        if destination.is_file():
            if digest(destination) == digest(source):
                continue
            # A local successful render may be newer than its last remote backup.
            # Never silently replace it with another render's data.
            print(f"{LOG} keeping local file that differs from its backup: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".restoring")
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != digest(source):
                raise IOError(f"Restore verification failed: {source}")
            os.replace(temporary, destination)  # local filesystem only
        finally:
            temporary.unlink(missing_ok=True)
        restored += 1
    print(f"{LOG} restored {restored} completed render files")


def backup_workflows(comfy_dir):
    remote = root()
    if remote is None:
        return
    local = Path(comfy_dir) / "user/default/workflows"
    check_working_path(local)
    with _workflow_lock:
        for source in local.rglob("*.json"):
            # ComfyUI writes via a temporary file, then replaces the .json locally.
            # Invalid/incomplete manually copied JSON must not replace a good backup.
            json.loads(source.read_text(encoding="utf-8"))
            publish(source, within(remote / "workflows", source.relative_to(local)))


def install_workflow_backup():
    """Back up successful UI saves before returning success to the browser."""
    if not os.environ.get("H3_GLOBAL_ROOT"):
        return
    import asyncio
    from aiohttp import web
    from server import PromptServer

    @web.middleware
    async def persist_workflow(request, handler):
        response = await handler(request)
        path = request.path.removeprefix("/api")
        if request.method == "POST" and path.startswith("/userdata/") and response.status < 400:
            try:
                await asyncio.to_thread(backup_workflows, os.environ["COMFYUI_PATH"])
            except Exception as error:
                raise web.HTTPServiceUnavailable(text=f"Saved locally, but Global Storage backup failed: {error}") from error
        return response

    PromptServer.instance.app.middlewares.append(persist_workflow)


def local_workflows(comfy):
    """Migrate the earlier global workflow symlink back to local editing safely."""
    remote = root() / "workflows"
    local = comfy / "user/default/workflows"
    if local.is_symlink():
        if local.resolve() != remote.resolve():
            raise RuntimeError(f"Workflow link points elsewhere: {local}")
        local.unlink()  # remove only the local link; global files remain untouched
    local.mkdir(parents=True, exist_ok=True)
    # Earlier versions stored workflows directly, with no receipt. Import valid
    # legacy JSON once; new backups use receipts and restore verification.
    for source in remote.rglob("*.json"):
        if receipt_path(source).exists():
            continue
        try:
            json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        destination = within(local, source.relative_to(remote))
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".restoring")
            shutil.copyfile(source, temporary)
            if digest(temporary) != digest(source):
                temporary.unlink()
                raise IOError(f"Workflow migration verification failed: {source}")
            os.replace(temporary, destination)
    restore(local, remote)
    backup_workflows(comfy)


def probe(comfy):
    remote = root()
    remote.mkdir(parents=True, exist_ok=True)
    # All POSIX-only operations happen locally. Test only ordinary remote I/O.
    target = remote / ".checks" / uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".h3-check-", dir=comfy) as directory:
        source = Path(directory) / "sample.bin"
        try:
            source.write_bytes(os.urandom(4096))
            publish(source, target)
            link = Path(directory) / "model-link"
            link.symlink_to(target)
            if digest(link) != digest(source):
                raise IOError("Cannot read the Global Volume through a local model link")
            source.write_bytes(os.urandom(4096))
            publish(source, target)  # replacing existing objects must also work
        finally:
            receipt_path(target).unlink(missing_ok=True)
            target.unlink(missing_ok=True)
    print(f"{LOG} Global Volume read/write/checksum probe passed")


def link_directory(local, remote):
    """Keep a local migration backup; never delete an existing input/workflow tree."""
    local, remote = Path(local), Path(remote)
    remote.mkdir(parents=True, exist_ok=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_symlink():
        if local.resolve() != remote.resolve():
            raise RuntimeError(f"Existing link points elsewhere: {local}")
        return
    if local.exists():
        for source in local.rglob("*"):
            if not source.is_file():
                continue
            dest = within(remote, source.relative_to(local))
            if dest.exists():
                if digest(source) != digest(dest):
                    raise RuntimeError(f"Input/workflow migration conflict: {dest}")
            else:
                publish(source, dest)
        # Preserve the original files, including if linking is interrupted.
        backup = local.with_name(local.name + ".before-global")
        if backup.exists():
            raise RuntimeError(f"Migration backup already exists: {backup}")
        local.rename(backup)
    local.symlink_to(remote, target_is_directory=True)


def configure(comfy_dir):
    remote = root()
    if remote is None:
        raise RuntimeError("Global storage was not enabled")
    comfy = Path(comfy_dir).resolve()
    for path in (comfy, comfy / "output", comfy / "temp", os.environ.get("HF_HOME", comfy / ".cache")):
        check_working_path(path)
    # 40 GiB of free scratch covers one staged model download plus render space.
    required = float(os.environ.get("H3_LOCAL_MIN_FREE_GB", "40")) * 1024 ** 3
    if shutil.disk_usage(comfy).free < required:
        raise RuntimeError("Not enough working disk space. Allocate a larger local disk (100 GB recommended).")
    probe(comfy)
    link_directory(comfy / "input", remote / "input")
    local_workflows(comfy)
    restore(comfy / "output")
    print(f"{LOG} enabled at {remote}; render scratch remains at {comfy / 'output'}")


def model_path(relative):
    remote = root()
    if remote is None:
        raise RuntimeError("Global storage was not enabled")
    return within(remote / "models", relative)


def link_model(comfy_dir, relative):
    destination = model_path(relative)
    if not valid(destination):
        raise RuntimeError(f"Model is incomplete or corrupt: {destination}")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Model path must be relative to models/")
    local = Path(comfy_dir) / "models" / relative
    if local.is_symlink():
        if local.resolve() == destination.resolve():
            return
        raise RuntimeError(f"Model link points elsewhere: {local}")
    if local.exists():
        if digest(local) != digest(destination):
            raise RuntimeError(f"Local model differs from the global model: {local}")
        local.unlink()  # verified identical copy is already stored globally
    local.parent.mkdir(parents=True, exist_ok=True)
    local.symlink_to(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["configure", "preflight", "backup-workflows", "model-ready", "publish-model", "link-model"])
    parser.add_argument("path")
    parser.add_argument("relative", nargs="?")
    args = parser.parse_args()
    if args.action == "preflight":
        check_working_path(args.path)
    elif args.action == "backup-workflows":
        backup_workflows(args.path)
    elif args.action == "configure":
        configure(args.path)
    elif args.action == "model-ready":
        raise SystemExit(0 if valid(model_path(args.path)) else 1)
    elif args.action == "publish-model":
        publish(args.path, model_path(args.relative))
    else:
        link_model(args.path, args.relative)


if __name__ == "__main__":
    main()
