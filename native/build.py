"""Build the optional native Monte Carlo backend.

Run it from anywhere::

    python native/build.py

The backend is optional. Plumbline runs on NumPy alone, and every part of the
audit works without this library. Building it is a choice you make when you
want the speed.

The library is plain C++ with a C ABI, not a Python extension module. That is
deliberate: there are no Python headers to find, no interpreter version to
match, and one built library serves every Python on the machine.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "plumbline_mc.cpp")
#: The loader looks here first. Keeping the library beside the package means a
#: development install finds it with no environment variable set.
OUTPUT_DIR = os.path.join(os.path.dirname(HERE), "plumbline", "engines", "_native")


def library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "plumbline_mc.dll"
    if system == "Darwin":
        return "libplumbline_mc.dylib"
    return "libplumbline_mc.so"


def find_compiler(preferred: str | None = None) -> tuple[str, str]:
    """Return (compiler path, family). Family is 'gnu' or 'msvc'."""
    if preferred:
        found = shutil.which(preferred)
        if not found:
            raise SystemExit(f"the compiler {preferred!r} is not on the PATH")
        return found, "msvc" if os.path.basename(found).lower() == "cl.exe" else "gnu"

    for candidate in ("g++", "clang++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found, "gnu"
    found = shutil.which("cl")
    if found:
        return found, "msvc"

    raise SystemExit(
        "no C++ compiler was found on the PATH.\n"
        "  Linux:   install g++ or clang++ from your package manager\n"
        "  macOS:   run 'xcode-select --install'\n"
        "  Windows: install MSVC build tools, or\n"
        "           winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT\n"
        "The backend is optional. Plumbline works without it."
    )


def gnu_command(compiler: str, output: str, native_arch: bool, debug: bool) -> list[str]:
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-fno-math-errno",  # std::exp and std::log never set errno here
        "-fvisibility=hidden",
        "-DPLUMBLINE_BUILDING",
        SOURCE,
        "-o",
        output,
    ]
    if native_arch:
        # -march=native is off by default: a library built for this exact CPU
        # can fault on an older one, and a wheel must not carry that risk.
        command.insert(4, "-march=native")
    if debug:
        command.insert(2, "-g")
    if platform.system() != "Windows":
        command.append("-pthread")
    else:
        # MinGW needs the threading runtime linked statically, or the library
        # cannot load without the compiler's own DLLs beside it.
        command += ["-static-libgcc", "-static-libstdc++", "-Wl,-Bstatic", "-lstdc++", "-lpthread", "-Wl,-Bdynamic"]
    return command


def msvc_command(output: str, debug: bool) -> list[str]:
    return [
        "cl",
        "/nologo",
        "/std:c++17",
        "/O2",
        "/EHsc",
        "/LD",
        "/DPLUMBLINE_BUILDING",
        SOURCE,
        f"/Fe:{output}",
    ] + (["/Zi"] if debug else [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="compiler to use instead of the first one found")
    parser.add_argument(
        "--march-native",
        action="store_true",
        help="tune for this exact CPU; the result may not run on another machine",
    )
    parser.add_argument("--debug", action="store_true", help="keep debug symbols")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="load the result and print its version")
    args = parser.parse_args()

    if not os.path.isfile(SOURCE):
        raise SystemExit(f"the source file is missing: {SOURCE}")

    compiler, family = find_compiler(args.compiler)
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, library_name())

    if family == "msvc":
        command = msvc_command(output, args.debug)
        cwd = args.output_dir
    else:
        command = gnu_command(compiler, output, args.march_native, args.debug)
        cwd = HERE

    print(f"compiler: {compiler}")
    print(f"command:  {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        return result.returncode

    print(f"built:    {output}")
    print(f"size:     {os.path.getsize(output) / 1024:.0f} kB")

    if args.check:
        sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
        from plumbline.engines import native

        native.reset()
        if not native.available():
            print(f"the library was built but did not load: {native.load_error()}")
            return 1
        print(f"loaded:   {native.backend_version()}")
        print(f"threads:  {native.backend_threads()}")

    return 0


if __name__ == "__main__":
    print(f"python:   {sys.version.split()[0]} on {sysconfig.get_platform()}")
    raise SystemExit(main())
