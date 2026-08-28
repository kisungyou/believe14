"""Run a command in a temporary environment containing one built artifact."""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sysconfig
import tempfile
import venv
from pathlib import Path


def _environment_python(directory: Path) -> Path:
    if os.name == "nt":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def artifact_environment_variables(
    environment: Path, inherited: dict[str, str] | None = None
) -> dict[str, str]:
    """Return variables that force child kernels to use the artifact venv."""

    variables = os.environ.copy() if inherited is None else inherited.copy()
    executable_directory = _environment_python(environment).parent
    previous_path = variables.get("PATH", "")
    variables["PATH"] = (
        str(executable_directory)
        if not previous_path
        else str(executable_directory) + os.pathsep + previous_path
    )
    variables["VIRTUAL_ENV"] = str(environment)
    variables["PYTHONNOUSERSITE"] = "1"
    return variables


def _expose_parent_dependencies(environment_python: Path) -> None:
    """Expose the invoking environment's dependencies after artifact packages."""

    result = subprocess.run(
        [
            str(environment_python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    child_site = Path(result.stdout.strip())
    parent_sites = [Path(path).resolve() for path in site.getsitepackages()]
    parent_purelib = Path(sysconfig.get_path("purelib")).resolve()
    if parent_purelib not in parent_sites:
        parent_sites.append(parent_purelib)
    child_site.joinpath("believe14-parent-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in parent_sites),
        encoding="utf-8",
    )


def run_in_artifact_environment(
    artifact: Path,
    command: tuple[str, ...],
    *,
    build_isolation: bool,
    with_dependencies: bool,
    extras: tuple[str, ...],
    installer: str,
) -> None:
    """Install ``artifact`` in a temporary venv and execute ``command``."""

    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    if not command:
        raise ValueError("A command is required.")
    with tempfile.TemporaryDirectory(prefix="believe14-artifact-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(
            with_pip=True,
            system_site_packages=False,
            clear=True,
        ).create(environment)
        python = _environment_python(environment)
        _expose_parent_dependencies(python)
        if installer == "uv":
            install = [
                os.environ.get("UV", "uv"),
                "pip",
                "install",
                "--python",
                str(python),
                "--offline",
                "--reinstall",
            ]
        else:
            install = [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--ignore-installed",
            ]
        if not with_dependencies:
            install.append("--no-deps")
        if not build_isolation:
            install.append("--no-build-isolation")
        requirement = str(artifact.resolve())
        if extras:
            requirement += f"[{','.join(extras)}]"
        install.append(requirement)
        subprocess.run(install, check=True)

        resolved = [str(python) if token == "{python}" else token for token in command]
        variables = artifact_environment_variables(environment)
        subprocess.run(resolved, check=True, env=variables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--installer", choices=("pip", "uv"), default="pip")
    parser.add_argument("--no-build-isolation", action="store_true")
    parser.add_argument("--with-dependencies", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = tuple(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    run_in_artifact_environment(
        arguments.artifact,
        command,
        build_isolation=not arguments.no_build_isolation,
        with_dependencies=arguments.with_dependencies,
        extras=tuple(arguments.extra),
        installer=arguments.installer,
    )


if __name__ == "__main__":
    main()
