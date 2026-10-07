#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path


FILES = (
    "obs.npy",
    "states.npy",
    "actions.npy",
    "rewards.npy",
    "legals.npy",
    "terminals.npy",
    "truncations.npy",
    "filled.npy",
    "path_lengths.npy",
    "episode_returns.npy",
    "metadata.json",
)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare_tier(raw_root: Path, output_root: Path, tier: str, copy: bool) -> None:
    src_dir = raw_root / tier.lower()
    dst_dir = output_root / tier.capitalize()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing raw tier directory: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        src = src_dir / name
        if src.exists():
            link_or_copy(src, dst_dir / name, copy)

    info = {
        "source": str(src_dir.resolve()),
        "format": "native_smac_npy",
        "map_name": "25m",
        "obs_dim": 250,
        "n_agents": 25,
        "n_actions": 31,
        "note": "MAC-Flow appends one-hot agent ids internally; these observations stay native.",
    }
    with open(dst_dir / "macflow_metadata.json", "w") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="../outputs/25m_qmix_native/collections_5000",
        help="Directory containing good/medium/poor native raw datasets.",
    )
    parser.add_argument(
        "--output-root",
        default="data/native_smac/25m",
        help="MAC-Flow local native dataset root.",
    )
    parser.add_argument("--tiers", nargs="+", default=["good", "medium", "poor"])
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    for tier in args.tiers:
        prepare_tier(raw_root, output_root, tier, args.copy)
        print(f"prepared {tier} -> {output_root / tier.capitalize()}")


if __name__ == "__main__":
    main()
