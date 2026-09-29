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

LOG = "[H3 Storage]"


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
        if json.loads(receipt.read_text(encoding="utf-8")) == info and valid(destination, full=False):
            return
    except (OSError, ValueError):
        pass
    if receipt.exists():
        receipt.unlink()
    shutil.copyfile(source, destination)
    if destination.stat().st_size != info["size"] or digest(destination) != info["sha256"]:
        raise IOError(f"Global copy verification failed: {destination}")
    receipt.write_text(json.dumps(info) + "\n", encoding="utf-8")
    if not valid(destination, full=False):
        raise IOError(f"Global completion receipt failed: {destination}")


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


def restore(output_dir):
    remote = root()
    if remote is None:
        return
    saved = remote / "output"
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
    mount = Path(os.environ.get("H3_GLOBAL_MOUNT", "/workspace-global")).resolve()
    if comfy == mount or mount in comfy.parents:
        raise RuntimeError("ComfyUI must remain on the working disk, outside the Global Volume.")
    # 40 GiB of free scratch covers one staged model download plus render space.
    required = float(os.environ.get("H3_LOCAL_MIN_FREE_GB", "40")) * 1024 ** 3
    if shutil.disk_usage(comfy).free < required:
        raise RuntimeError("Not enough working disk space. Allocate a larger local disk (100 GB recommended).")
    link_directory(comfy / "input", remote / "input")
    link_directory(comfy / "user/default/workflows", remote / "workflows")
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
    parser.add_argument("action", choices=["configure", "model-ready", "publish-model", "link-model"])
    parser.add_argument("path")
    parser.add_argument("relative", nargs="?")
    args = parser.parse_args()
    if args.action == "configure":
        configure(args.path)
    elif args.action == "model-ready":
        raise SystemExit(0 if valid(model_path(args.path)) else 1)
    elif args.action == "publish-model":
        publish(args.path, model_path(args.relative))
    else:
        link_model(args.path, args.relative)


if __name__ == "__main__":
    main()
