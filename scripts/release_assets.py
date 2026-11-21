"""Check the versioned files and hash the GitHub Release assets."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tomllib
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313", "cp314", "cp315")
PLATFORM_TAGS = ("win_amd64", "manylinux_2_34_x86_64")
DLL_MEMBERS = {"umap_capi.dll", "umap_capi.lib", "umap_capi.h", "fit_graph.c", "README.md", "LICENSE"}
LOCAL_PATH = re.compile(
    rb"file://(?:/[a-z]:/|/)|"
    rb"(?<![a-z0-9])[a-z]:[\\/]+(?:a|build|git_repo|Program Files(?: \(x86\))?|"
    rb"runneradmin|target|temp|tmp|Users|workspace)[\\/]+|"
    rb"(?<![a-z0-9_])/(?:home|Users|mnt/[a-z]|tmp|var/tmp|workspace|__w|builds|opt/hostedtoolcache)/",
    re.IGNORECASE,
)
ABSOLUTE_PDB = re.compile(rb"RSDS.{20}(?:[a-zA-Z]:[\\/]|[\\/])", re.DOTALL)


def release_version(tag):
    with (ROOT / "Cargo.toml").open("rb") as manifest:
        version = tomllib.load(manifest)["workspace"]["package"]["version"]
    if tag != f"v{version}":
        raise SystemExit(f"Expected release tag v{version}, got {tag!r}.")
    if not (ROOT / "docs" / "releases" / f"{tag}.md").is_file():
        raise SystemExit(f"Missing release notes for {tag}.")
    return version


def hash_assets(directory, version):
    dll_name = f"umap-alternative-cabi-{version}-win-x64.zip"
    expected = {
        f"umap_alternative-{version}-{python}-{python}-{platform}.whl"
        for python in PYTHON_TAGS
        for platform in PLATFORM_TAGS
    } | {dll_name}
    checksum_file = directory / "SHA256SUMS"
    actual = {path.name for path in directory.iterdir() if path != checksum_file}
    if actual != expected:
        raise SystemExit(f"Unexpected release assets. Missing: {sorted(expected - actual)}. Extra: {sorted(actual - expected)}.")

    sources = {
        path.removeprefix("python/")
        for path in subprocess.check_output(
            ["git", "ls-files", "python/umap_alternative"], cwd=ROOT, text=True,
        ).splitlines()
    }
    source_contents = {
        name: subprocess.check_output(
            ["git", "show", f"HEAD:python/{name}"], cwd=ROOT,
        ).replace(b"\r\n", b"\n")
        for name in sources
    }
    dist_info = f"umap_alternative-{version}.dist-info"
    wheel_members = sources | {
        f"{dist_info}/{path}" for path in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE")
    }
    checksums = []
    for name in sorted(expected):
        path = directory / name
        with ZipFile(path) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise SystemExit(f"{name} contains a damaged entry: {bad_file}")
            names = archive.namelist()
            if name == dll_name:
                valid_members = len(names) == len(DLL_MEMBERS) and set(names) == DLL_MEMBERS
            else:
                extra = set(names) - wheel_members
                valid_members = (
                    wheel_members.issubset(names)
                    and len(names) == len(wheel_members) + 1
                    and len(extra) == 1
                    and all(
                        member.startswith("umap_alternative/_native.")
                        and member.endswith((".pyd", ".so"))
                        for member in extra
                    )
                )
            if not valid_members:
                raise SystemExit(f"{name} does not contain the expected release files.")
            for member in names:
                data = archive.read(member)
                if LOCAL_PATH.search(data) or ABSOLUTE_PDB.search(data) or (
                    b"\0" in data and any(LOCAL_PATH.search(data[offset::2]) for offset in (0, 1))
                ):
                    raise SystemExit(f"{name}: machine-specific path in {member}.")
                if member in source_contents and data.replace(b"\r\n", b"\n") != source_contents[member]:
                    raise SystemExit(f"{name}: {member} differs from the committed Python source.")
        with path.open("rb") as asset:
            digest = hashlib.file_digest(asset, "sha256").hexdigest()
        checksums.append(f"{digest}  {name}\n")
    checksum_file.write_text("".join(checksums), encoding="ascii", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag matching the workspace version")
    parser.add_argument("--assets", type=Path, help="Validate the assets and write SHA256SUMS into this directory")
    args = parser.parse_args()
    version = release_version(args.tag)
    if args.assets is not None:
        hash_assets(args.assets, version)
    print(f"Validated {args.tag}")
