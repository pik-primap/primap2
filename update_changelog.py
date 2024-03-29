#!/usr/bin/env python3
import pathlib
import sys

version = sys.argv[1]

with open("CHANGELOG.rst") as fd:
    old_content = fd.read().splitlines(keepends=True)

with (
    open("CHANGELOG.rst", "w") as fd,
    open(".changelog_latest_version.rst", "w") as lfd,
):
    # Write header
    fd.writelines(old_content[:4])

    # New version
    new_version = f"{version}\n"
    new_version += "-" * len(version) + "\n"
    ch_dir = pathlib.Path("changelog_unreleased")
    for changelog_file in ch_dir.iterdir():
        new_version += changelog_file.read_text()
        changelog_file.unlink()

    fd.write(new_version)
    lfd.write(new_version)

    fd.write("\n")

    # Rest
    fd.writelines(old_content[4:])
