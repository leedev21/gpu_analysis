import os
import sys
import subprocess
import ctypes
import glob
import sysconfig
import importlib
from pathlib import Path
import platform

def get_te_path():
    """Find Transformer Engine install path using pip"""
    command = [sys.executable, "-m", "pip", "show", "transformer_engine"]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    if result and len(result) > 1:
        return Path(result[result.index("Location") + 1].strip())
    else:
        return None

def _get_sys_extension():
    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")

    return extension

def _load_cudnn():
    """Load CUDNN shared library."""

    lib_path = glob.glob(
        os.path.join(
            sysconfig.get_path("purelib"),
            f"nvidia/cudnn/lib/libcudnn.{_get_sys_extension()}.*[0-9]",
        )
    )

    if lib_path:
        assert (
            len(lib_path) == 1
        ), f"Found {len(lib_path)} libcudnn.{_get_sys_extension()}.x in nvidia-cudnn-cuXX."
        return ctypes.CDLL(lib_path[0], mode=ctypes.RTLD_GLOBAL)

    cudnn_home = os.environ.get("CUDNN_HOME") or os.environ.get("CUDNN_PATH")
    if cudnn_home:
        libs = glob.glob(f"{cudnn_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        libs = glob.glob(f"{cuda_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    return ctypes.CDLL(f"libcudnn.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)

def _load_library(te_path):
    """Load shared library with Transformer Engine C extensions"""

    so_path = te_path / "transformer_engine" / "wheel_lib" / f"libtransformer_engine.{_get_sys_extension()}"
    if not so_path.exists():
        so_path = te_path / "transformer_engine" / f"libtransformer_engine.{_get_sys_extension()}"
    if not so_path.exists():
        so_path = te_path / f"libtransformer_engine.{_get_sys_extension()}"
    if not so_path.exists():
        print(f"Could not find libtransformer_engine.{_get_sys_extension()}")
        return

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

def _load_nvrtc():
    """Load NVRTC shared library."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        libs = glob.glob(f"{cuda_home}/**/libnvrtc.{_get_sys_extension()}*", recursive=True)
        libs = list(filter(lambda x: not ("stub" in x or "libnvrtc-builtins" in x), libs))
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    libs = subprocess.check_output("ldconfig -p | grep 'libnvrtc'", shell=True)
    libs = libs.decode("utf-8").split("\n")
    sos = []
    for lib in libs:
        if "stub" in lib or "libnvrtc-builtins" in lib:
            continue
        if "libnvrtc" in lib and "=>" in lib:
            sos.append(lib.split(">")[1].strip())
    if sos:
        return ctypes.CDLL(sos[0], mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(f"libnvrtc.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)

def _load_userbuffers(dll_path):
    """Load shared library with userbuffers"""
    extension = _get_sys_extension()
    lib_name = "libtransformer_engine_userbuffers." + extension
    dll_path = os.path.join(dll_path, lib_name)

    if os.path.exists(dll_path):
        return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    return None

def _load_transformer_engine_torch():
    """Load shared library with Transformer Engine C extensions"""
    te_path = get_te_path()
    if not te_path:
        return
    _load_cudnn()
    _load_nvrtc()
    _load_library(te_path)
    if _load_userbuffers(te_path):
        module_name = "transformer_engine_extensions"
    else:
        module_name = "transformer_engine_torch"
    extension = _get_sys_extension()
    try:
        so_dir = te_path / "transformer_engine" / "wheel_lib"
        so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
    except StopIteration:
        try:
            so_dir = te_path / "transformer_engine"
            so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
        except StopIteration:
            so_dir = None
            so_path = next(te_path.glob(f"{module_name}.*.{extension}"))

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    solib = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = solib
    spec.loader.exec_module(solib)
    return so_dir