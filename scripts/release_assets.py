"""Check the versioned files and hash the GitHub Release assets."""

import argparse
import hashlib
from pathlib import Path
import tomllib
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313", "cp314", "cp315")
PLATFORM_TAGS = ("win_amd64", "manylinux_2_34_x86_64")
DLL_MEMBERS = {"umap_capi.dll", "umap_capi.lib", "umap_capi.h", "fit_graph.c", "README.md", "LICENSE"}


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

    checksums = []
    for name in sorted(expected):
        path = directory / name
        with ZipFile(path) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise SystemExit(f"{name} contains a damaged entry: {bad_file}")
            if name == dll_name and set(archive.namelist()) != DLL_MEMBERS:
                raise SystemExit(f"{name} does not contain the expected C ABI files.")
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
