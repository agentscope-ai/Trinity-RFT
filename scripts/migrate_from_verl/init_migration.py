import argparse
import shutil
import subprocess
from pathlib import Path


def main(args):
    # cd to verl repo dir and checkout the specified version
    subprocess.run(["git", "fetch", "origin"], cwd=args.repo_dir, check=True)
    subprocess.run(["git", "checkout", args.version], cwd=args.repo_dir, check=True)

    # copy files from verl repo to trinity repo with new names, and add them to git
    trinity_path_prefix = Path(__file__).parent.parent.parent / "trinity" / "trainer" / "verl"
    verl_path_prefix = Path(args.repo_dir)
    file_maps = {
        "fsdp_workers": ("verl", "workers", "fsdp_workers.py"),
        "dp_actor": ("verl", "workers", "actor", "dp_actor.py"),
        "fsdp_checkpoint_manager": ("verl", "utils", "checkpoint", "fsdp_checkpoint_manager.py"),
        "megatron_workers": ("verl", "workers", "megatron_workers.py"),
        "megatron_actor": ("verl", "workers", "actor", "megatron_actor.py"),
        "megatron_checkpoint_manager": (
            "verl",
            "utils",
            "checkpoint",
            "megatron_checkpoint_manager.py",
        ),
        "ray_trainer": ("verl", "trainer", "ppo", "ray_trainer.py"),
    }

    for filename, path_parts in file_maps.items():
        src_path = verl_path_prefix / Path(*path_parts)
        dst_path = trinity_path_prefix / f"{filename}-{args.version}.py"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(src_path.read_bytes())
        subprocess.run(["git", "add", str(dst_path)], cwd=trinity_path_prefix, check=True)
        print(f"Copied {src_path} to {dst_path}")

    print("Running pre-commit on the migrated files...")
    subprocess.run(
        ["pre-commit", "run", "--all-files"], cwd=Path(__file__).parent.parent.parent, check=True
    )

    # move the files to the build directory and reset git status to keep history clean
    target_dir = trinity_path_prefix / "build" / f"{args.version}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, path_parts in file_maps.items():
        src_path = trinity_path_prefix / f"{filename}-{args.version}.py"
        dst_path = target_dir / f"{filename}.py"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "add", str(src_path)], cwd=trinity_path_prefix, check=True)
        subprocess.run(["git", "reset", "HEAD", str(src_path)], cwd=trinity_path_prefix, check=True)
        shutil.move(src_path, dst_path)
        print(f"Moved {src_path} to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, help="Path to the verl repository")
    parser.add_argument("--version", required=True, help="Version of the verl repository")

    args = parser.parse_args()
    main(args)
