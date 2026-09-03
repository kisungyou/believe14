from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.artifact_hashes import verify_manifest, write_manifest
from scripts.release_info import load_release_info, project_root
from scripts.run_in_artifact_env import artifact_environment_variables
from scripts.verify_release import (
    ReleaseVerificationError,
    _normalized_remote,
    verify_metadata,
    verify_repository,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_release_info_is_canonical_and_final() -> None:
    info = load_release_info()
    assert info.name == "believe14"
    assert info.version == "0.1.0"
    assert info.tag == "v0.1.0"
    assert info.wheel == "believe14-0.1.0-py3-none-any.whl"
    assert info.sdist == "believe14-0.1.0.tar.gz"
    verify_metadata(project_root(), info)


def test_remote_normalization_ignores_checkout_credentials() -> None:
    expected = "https://github.com/kisungyou/believe14"
    assert _normalized_remote("git@github.com:kisungyou/believe14.git") == expected
    assert (
        _normalized_remote(
            "https://x-access-token:secret@github.com/kisungyou/believe14.git"
        )
        == expected
    )


def test_checksum_manifest_detects_artifact_mutation(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    manifest = tmp_path / "SHA256SUMS"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    artifacts = (wheel, sdist)

    write_manifest(manifest, artifacts)
    verify_manifest(manifest, artifacts)

    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksums changed"):
        verify_manifest(manifest, artifacts)


def test_artifact_environment_controls_child_kernel_interpreter(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "artifact-venv"
    variables = artifact_environment_variables(
        environment,
        {"PATH": "/outer/bin", "PYTHONNOUSERSITE": "0"},
    )
    expected_bin = environment / ("Scripts" if os.name == "nt" else "bin")
    assert variables["PATH"].split(os.pathsep)[0] == str(expected_bin)
    assert variables["VIRTUAL_ENV"] == str(environment)
    assert variables["PYTHONNOUSERSITE"] == "1"


def test_repository_verifier_requires_annotated_clean_tag(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    (tmp_path / "tracked.txt").write_text("release\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "Initial release")
    _git(tmp_path, "remote", "add", "origin", "git@github.com:kisungyou/believe14.git")
    _git(tmp_path, "tag", "-a", "v0.1.0", "-m", "believe14 0.1.0")

    info = replace(
        load_release_info(), repository="https://github.com/kisungyou/believe14"
    )
    verify_repository(
        tmp_path,
        info,
        check_clean=True,
        check_remote=True,
        require_tag=True,
        expected_tag="v0.1.0",
    )

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match="clean source tree"):
        verify_repository(
            tmp_path,
            info,
            check_clean=True,
            check_remote=True,
            require_tag=True,
            expected_tag="v0.1.0",
        )


def test_repository_verifier_rejects_lightweight_tag(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    (tmp_path / "tracked.txt").write_text("release\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "Initial release")
    _git(tmp_path, "tag", "v0.1.0")

    with pytest.raises(ReleaseVerificationError, match="annotated tag"):
        verify_repository(
            tmp_path,
            load_release_info(),
            check_clean=True,
            check_remote=False,
            require_tag=True,
            expected_tag="v0.1.0",
        )


def test_forced_tag_refetch_restores_annotated_release_tag(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    source = tmp_path / "source"
    runner = tmp_path / "runner"

    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(origin))
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Release Test")
    _git(source, "config", "user.email", "release@example.invalid")
    (source / "tracked.txt").write_text("release\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "Initial release")
    _git(source, "tag", "-a", "v0.1.0", "-m", "believe14 0.1.0")
    _git(source, "remote", "add", "origin", str(origin))
    _git(source, "push", "origin", "main", "refs/tags/v0.1.0")

    _git(tmp_path, "clone", "--no-tags", str(origin), str(runner))
    _git(runner, "tag", "v0.1.0")
    info = load_release_info()
    with pytest.raises(ReleaseVerificationError, match="annotated tag"):
        verify_repository(
            runner,
            info,
            check_clean=True,
            check_remote=False,
            require_tag=True,
            expected_tag="v0.1.0",
        )

    _git(
        runner,
        "fetch",
        "--force",
        "--no-tags",
        "origin",
        "refs/tags/v0.1.0:refs/tags/v0.1.0",
    )
    verify_repository(
        runner,
        info,
        check_clean=True,
        check_remote=False,
        require_tag=True,
        expected_tag="v0.1.0",
    )


def test_publish_refetches_annotated_tag_before_verification() -> None:
    publish = (project_root() / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )
    fetch_position = publish.index("git fetch --force --no-tags origin")
    refspec_position = publish.index(
        '"refs/tags/${GITHUB_REF_NAME}:refs/tags/${GITHUB_REF_NAME}"'
    )
    verification_position = publish.index("python -m scripts.verify_release")

    assert fetch_position < refspec_position < verification_position


def test_workflows_discover_artifact_names_and_cover_release_matrix() -> None:
    workflows = project_root() / ".github/workflows"
    ci = (workflows / "ci.yml").read_text(encoding="utf-8")
    publish = (workflows / "publish.yml").read_text(encoding="utf-8")
    nightly = (workflows / "nightly.yml").read_text(encoding="utf-8")

    for contents in (ci, publish, nightly):
        assert "believe14-0.1.0" not in contents
        assert "scripts.release_info" in contents
        assert "scripts/installed_smoke.py" in contents

    for operating_system in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert operating_system in ci
        assert operating_system in publish
    for python_version in ('"3.12"', '"3.13"', '"3.14"'):
        assert python_version in ci
        assert python_version in publish

    assert "https://test.pypi.org/legacy/" in publish
    assert "needs: [distribution, testpypi-smoke]" in publish
    assert "pypa/gh-action-pypi-publish@release/v1" in publish
    assert "actions/deploy-pages@v4" in ci
    assert "actions/deploy-pages@v4" in publish
