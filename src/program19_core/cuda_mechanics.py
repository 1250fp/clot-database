"""GPU integrator for one planar clot layer.

The kernel keeps adaptive substeps and cohesion search on the device. Accepted
links follow the Python valence rules and are uploaded at each outer step. Thermal noise is a
deterministic Philox stream seeded by ``config.seed``; it is not the NumPy
sequence stored in checkpoints. The Philox substep counter is saved as
``cuda_noise_substep`` when a checkpoint is written.
"""

from __future__ import annotations

import ctypes
import importlib.metadata as metadata
import math
import os
from pathlib import Path

import numpy as np

from .config import Program19Config
from .state import (
    ForceDiagnostics,
    SimulationState,
    StepDiagnostics,
    cohesion_degrees,
    kbt_nN_um,
    mobility_and_diffusion,
)


_RUNTIME: "CudaRuntime | None" = None
_BLOCK = 256
_CHUNK = 48


class CudaMechanicsError(RuntimeError):
    pass


def _as_c_char_p_array(values: list[bytes]):
    buffers = [ctypes.create_string_buffer(value) for value in values]
    array = (ctypes.c_char_p * len(buffers))()
    for index, buffer in enumerate(buffers):
        array[index] = ctypes.cast(buffer, ctypes.c_char_p)
    return array, buffers


def _check_driver(library, code: int, action: str) -> None:
    if code == 0:
        return
    message = ctypes.c_char_p()
    library.cuGetErrorString(code, ctypes.byref(message))
    detail = message.value.decode() if message.value else f"code {code}"
    raise CudaMechanicsError(f"{action} failed: {detail}")


def _check_nvrtc(library, code: int, action: str, program=None) -> None:
    if code == 0:
        return
    detail = library.nvrtcGetErrorString(code)
    text = detail.decode() if isinstance(detail, bytes) else str(detail)
    log = ""
    if program is not None:
        size = ctypes.c_size_t()
        if library.nvrtcGetProgramLogSize(program, ctypes.byref(size)) == 0 and size.value:
            raw = ctypes.create_string_buffer(size.value)
            library.nvrtcGetProgramLog(program, raw)
            log = "\n" + raw.value.decode(errors="replace")
    raise CudaMechanicsError(f"{action} failed: {text}{log}")


class CudaRuntime:
    """Driver API plus an NVRTC-compiled mechanics module."""

    def __init__(self) -> None:
        self._dll_dir = None
        self.nvrtc = self._load_nvrtc()
        self.driver = ctypes.WinDLL("nvcuda.dll")
        self._bind()
        _check_driver(self.driver, self.driver.cuInit(0), "cuInit")
        device = ctypes.c_int()
        _check_driver(self.driver, self.driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self.device = device
        context = ctypes.c_void_p()
        _check_driver(
            self.driver,
            self.driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
            "cuDevicePrimaryCtxRetain",
        )
        self.context = context
        _check_driver(self.driver, self.driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        _check_driver(
            self.driver,
            self.driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device),
            "compute capability major",
        )
        _check_driver(
            self.driver,
            self.driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device),
            "compute capability minor",
        )
        self.arch = int(major.value) * 10 + int(minor.value)
        image = self._compile()
        self._image_buffer = ctypes.create_string_buffer(image)
        self.module = ctypes.c_void_p()
        _check_driver(
            self.driver,
            self.driver.cuModuleLoadData(ctypes.byref(self.module), self._image_buffer),
            "cuModuleLoadData",
        )
        self.function = ctypes.c_void_p()
        _check_driver(
            self.driver,
            self.driver.cuModuleGetFunction(
                ctypes.byref(self.function), self.module, b"integrate_chunk"
            ),
            "cuModuleGetFunction",
        )
        self.filopodia = ctypes.c_void_p()
        _check_driver(
            self.driver,
            self.driver.cuModuleGetFunction(
                ctypes.byref(self.filopodia), self.module, b"update_filopodia"
            ),
            "update_filopodia",
        )
        self.cohesion_gather = self._function(b"gather_cohesion_pairs")
        self.cohesion_clear_bits = self._function(b"clear_pair_bits")
        self.cohesion_set_bits = self._function(b"set_pair_bits")
        self.cohesion_mark = self._function(b"mark_cohesion_candidates")
        self.cohesion_count = self._function(b"count_cohesion_flags")
        self.cohesion_accept = self._function(b"accept_cohesion_links")
        per_sm = ctypes.c_int(1)
        sms = ctypes.c_int(1)
        self.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(per_sm), self.function, _BLOCK, 0
        )
        self.driver.cuDeviceGetAttribute(ctypes.byref(sms), 16, self.device)
        resident = max(1, int(per_sm.value) * int(sms.value))
        # Software barriers dominate once the force loop fits in a couple of blocks.
        self.blocks = 1 if resident < 1 else 2

    def _load_nvrtc(self):
        candidate_dirs: list[Path] = []
        for pkg_name in ("nvidia-cuda-nvrtc-cu12", "nvidia-cuda-nvrtc-cu13", "nvidia-cuda-nvrtc"):
            try:
                dist = metadata.distribution(pkg_name)
                candidate_dirs.append(Path(dist.locate_file("nvidia/cuda_nvrtc/bin")))
            except metadata.PackageNotFoundError:
                continue
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidate_dirs.append(Path(cuda_path) / "bin")
        toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit_root.exists():
            for vdir in sorted(toolkit_root.glob("v*"), reverse=True):
                candidate_dirs.append(vdir / "bin")
        for bin_dir in candidate_dirs:
            if not bin_dir.exists():
                continue
            libraries = sorted(
                path for path in bin_dir.glob("nvrtc64_*.dll")
                if "builtin" not in path.name and "alt" not in path.name
            )
            if libraries:
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                self._dll_dir = os.add_dll_directory(str(bin_dir))
                self._preloaded = [
                    ctypes.WinDLL(str(path)) for path in sorted(bin_dir.glob("nvrtc-builtins*.dll"))
                ]
                return ctypes.WinDLL(str(libraries[-1]))
        ptx_path = Path(__file__).with_name("mechanics_kernel.ptx")
        if ptx_path.exists():
            return None
        raise CudaMechanicsError(
            "CUDA mechanics needs the NVRTC package or mechanics_kernel.ptx. Install with: "
            "py -3 -m pip install nvidia-cuda-nvrtc-cu12"
        )

    def _bind(self) -> None:
        driver = self.driver
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        driver.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        driver.cuCtxSetCurrent.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        driver.cuModuleLoadData.restype = ctypes.c_int
        driver.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        driver.cuModuleGetFunction.restype = ctypes.c_int
        driver.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        driver.cuLaunchKernel.restype = ctypes.c_int
        driver.cuLaunchCooperativeKernel.argtypes = driver.cuLaunchKernel.argtypes
        driver.cuLaunchCooperativeKernel.restype = ctypes.c_int
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        ]
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
        driver.cuCtxSynchronize.argtypes = []
        driver.cuCtxSynchronize.restype = ctypes.c_int
        driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        driver.cuMemAlloc_v2.restype = ctypes.c_int
        driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        driver.cuMemFree_v2.restype = ctypes.c_int
        driver.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        driver.cuMemcpyHtoD_v2.restype = ctypes.c_int
        driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        driver.cuMemcpyDtoH_v2.restype = ctypes.c_int
        driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        driver.cuGetErrorString.restype = ctypes.c_int
        nvrtc = self.nvrtc
        if nvrtc is None:
            return
        nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        nvrtc.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        ]
        nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
        nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetCUBINSize.restype = ctypes.c_int
        nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetCUBIN.restype = ctypes.c_int
        nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetPTX.restype = ctypes.c_int
        nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int
        nvrtc.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        nvrtc.nvrtcGetNumSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int)]
        nvrtc.nvrtcGetNumSupportedArchs.restype = ctypes.c_int
        nvrtc.nvrtcGetSupportedArchs.argtypes = [ctypes.POINTER(ctypes.c_int)]
        nvrtc.nvrtcGetSupportedArchs.restype = ctypes.c_int

    def _supported_archs(self) -> list[int]:
        count = ctypes.c_int()
        _check_nvrtc(
            self.nvrtc, self.nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(count)), "arch count"
        )
        values = (ctypes.c_int * int(count.value))()
        _check_nvrtc(self.nvrtc, self.nvrtc.nvrtcGetSupportedArchs(values), "arch list")
        return [int(values[index]) for index in range(int(count.value))]

    def _compile(self) -> bytes:
        if self.nvrtc is None:
            cubin_path = Path(__file__).with_name(f"mechanics_kernel_sm{self.arch}.cubin")
            if cubin_path.exists():
                return cubin_path.read_bytes()
            ptx_path = Path(__file__).with_name("mechanics_kernel.ptx")
            if ptx_path.exists():
                return ptx_path.read_bytes() + b"\x00"
        source_path = Path(__file__).with_name("mechanics_kernel.cu")
        source = source_path.read_text(encoding="utf-8").encode("utf-8")
        supported = set(self._supported_archs())
        attempts: list[tuple[str, bool]] = []
        if self.arch in supported:
            attempts.append((f"sm_{self.arch}", True))
        fallback = max((arch for arch in supported if arch <= self.arch), default=None)
        if fallback is not None:
            attempts.append((f"compute_{fallback}", False))
        elif supported:
            attempts.append((f"compute_{max(supported)}", False))
        else:
            attempts.append(("compute_80", False))
        errors: list[str] = []
        for arch, want_cubin in attempts:
            try:
                return self._compile_one(source, arch, want_cubin)
            except CudaMechanicsError as exc:
                errors.append(f"{arch}: {exc}")
                if "nvrtc-builtins" in str(exc):
                    break
        raise CudaMechanicsError("NVRTC could not compile the mechanics kernel.\n" + "\n".join(errors))

    def _compile_one(self, source: bytes, arch: str, want_cubin: bool) -> bytes:
        program = ctypes.c_void_p()
        source_buffer = ctypes.create_string_buffer(source)
        _check_nvrtc(
            self.nvrtc,
            self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(program), source_buffer, b"mechanics_kernel.cu", 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        options, retained = _as_c_char_p_array([
            b"--std=c++17",
            f"--gpu-architecture={arch}".encode(),
            b"--fmad=false",
        ])
        code = self.nvrtc.nvrtcCompileProgram(program, len(retained), options)
        _check_nvrtc(self.nvrtc, code, f"nvrtcCompileProgram ({arch})", program)
        size = ctypes.c_size_t()
        getter_size = self.nvrtc.nvrtcGetCUBINSize if want_cubin else self.nvrtc.nvrtcGetPTXSize
        getter = self.nvrtc.nvrtcGetCUBIN if want_cubin else self.nvrtc.nvrtcGetPTX
        _check_nvrtc(self.nvrtc, getter_size(program, ctypes.byref(size)), "image size", program)
        image = ctypes.create_string_buffer(size.value)
        _check_nvrtc(self.nvrtc, getter(program, image), "image download", program)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        return image.raw

    def activate(self) -> None:
        _check_driver(self.driver, self.driver.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")

    def alloc(self, nbytes: int) -> int:
        self.activate()
        pointer = ctypes.c_uint64()
        _check_driver(
            self.driver,
            self.driver.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(max(int(nbytes), 4))),
            "cuMemAlloc",
        )
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if pointer:
            _check_driver(self.driver, self.driver.cuMemFree_v2(ctypes.c_uint64(pointer)), "cuMemFree")

    def upload(self, pointer: int, array: np.ndarray) -> None:
        host = np.ascontiguousarray(array)
        _check_driver(
            self.driver,
            self.driver.cuMemcpyHtoD_v2(
                ctypes.c_uint64(pointer),
                host.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(host.nbytes),
            ),
            "cuMemcpyHtoD",
        )

    def download(self, array: np.ndarray, pointer: int) -> None:
        host = np.ascontiguousarray(array)
        _check_driver(
            self.driver,
            self.driver.cuMemcpyDtoH_v2(
                host.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint64(pointer),
                ctypes.c_size_t(host.nbytes),
            ),
            "cuMemcpyDtoH",
        )
        if host is not array:
            array[...] = host

    def launch(self, arguments: list[ctypes._SimpleCData]) -> None:
        self.activate()
        holders = arguments
        pointers = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in holders]
        )
        code = self.driver.cuLaunchCooperativeKernel(
            self.function,
            self.blocks, 1, 1,
            _BLOCK, 1, 1,
            0,
            None,
            pointers,
            None,
        )
        if code != 0:
            code = self.driver.cuLaunchKernel(
                self.function,
                1, 1, 1,
                _BLOCK, 1, 1,
                0,
                None,
                pointers,
                None,
            )
        _check_driver(self.driver, code, "cuLaunchKernel")
        _check_driver(self.driver, self.driver.cuCtxSynchronize(), "cuCtxSynchronize")

    def launch_block(self, function, threads: int, arguments: list[ctypes._SimpleCData]) -> None:
        self.activate()
        pointers = (ctypes.c_void_p * len(arguments))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in arguments]
        )
        _check_driver(
            self.driver,
            self.driver.cuLaunchKernel(
                function, 1, 1, 1, int(threads), 1, 1, 0, None, pointers, None
            ),
            "filopodia kernel",
        )
        _check_driver(self.driver, self.driver.cuCtxSynchronize(), "filopodia sync")

    def _function(self, name: bytes) -> ctypes.c_void_p:
        function = ctypes.c_void_p()
        _check_driver(
            self.driver,
            self.driver.cuModuleGetFunction(ctypes.byref(function), self.module, name),
            name.decode(),
        )
        return function

    def launch_grid(
        self, function, threads: int, arguments: list[ctypes._SimpleCData]
    ) -> None:
        self.activate()
        width = max(int(threads), 1)
        block = 256 if width > 256 else width
        grid = (width + block - 1) // block
        pointers = (ctypes.c_void_p * len(arguments))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in arguments]
        )
        _check_driver(
            self.driver,
            self.driver.cuLaunchKernel(
                function, int(grid), 1, 1, int(block), 1, 1, 0, None, pointers, None
            ),
            "cohesion gather",
        )
        _check_driver(self.driver, self.driver.cuCtxSynchronize(), "cohesion gather sync")


def runtime() -> CudaRuntime:
    global _RUNTIME
    if os.name != "nt":
        raise CudaMechanicsError(
            "The CUDA backend loads nvcuda.dll and NVRTC wheels, so it runs on Windows only; "
            "use --backend python here."
        )
    if _RUNTIME is None:
        _RUNTIME = CudaRuntime()
    return _RUNTIME


def _require_supported(config: Program19Config) -> None:
    if config.thermal_noise_model != "gaussian_independent":
        raise CudaMechanicsError(
            "CUDA mechanics supports independent Gaussian thermal noise. "
            "codeocean_uniform_shared stays on the Python integrator."
        )
    if (
        config.filopodia_selection_rule != "nearest_free"
        or config.filopodia_handover_rule != "global_reacquire"
    ):
        raise CudaMechanicsError(
            "CUDA mechanics supports nearest-free filopodia with global reacquire. "
            "codeocean_random and graph_neighbor stay on the Python integrator."
        )


def _pairs(values, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32).reshape((-1, width))
    return np.ascontiguousarray(array)


class _Buffer:
    def __init__(self, runtime_obj: CudaRuntime, nbytes: int) -> None:
        self.runtime = runtime_obj
        self.pointer = runtime_obj.alloc(nbytes)
        self.nbytes = max(int(nbytes), 4)

    def ensure(self, nbytes: int) -> None:
        if nbytes <= self.nbytes:
            return
        self.runtime.free(self.pointer)
        self.pointer = self.runtime.alloc(nbytes)
        self.nbytes = max(int(nbytes), 4)


class CudaSession:
    def __init__(self, state: SimulationState, config: Program19Config) -> None:
        _require_supported(config)
        self.runtime = runtime()
        self.n_nodes = int(len(state.nodes_um))
        self.n_platelets = int(len(state.platelets_um))
        self.n_backbone = int(len(state.backbone_pairs))
        self.n_bending = int(len(state.bending_triples))
        self.max_arms = max((len(arms) for arms in state.filopodia), default=0)
        self.fiber_stride = max((len(item) for item in state.node_fiber_ids), default=1)
        self.noise_substeps = int(getattr(state, "cuda_noise_substep", 0))
        self._allocate()
        self._upload_static(state)
        self._upload_positions(state)
        self._upload_filopodia(state)
        self.cohesion_count = -1
        self._upload_cohesion(state)

    def _d(self, count: int) -> _Buffer:
        return _Buffer(self.runtime, max(count, 1) * 8)

    def _i(self, count: int) -> _Buffer:
        return _Buffer(self.runtime, max(count, 1) * 4)

    def _allocate(self) -> None:
        n = self.n_nodes
        p = self.n_platelets
        self.node_x = self._d(n)
        self.node_y = self._d(n)
        self.platelet_x = self._d(p)
        self.platelet_y = self._d(p)
        self.backbone_pairs = self._i(max(self.n_backbone, 1) * 2)
        self.backbone_l0 = self._d(self.n_backbone)
        self.cohesion_pairs = self._i(2)
        self.cohesion_l0 = self._d(1)
        self.bend_i = self._i(self.n_bending)
        self.bend_j = self._i(self.n_bending)
        self.bend_k = self._i(self.n_bending)
        self.bend_phi0 = self._d(self.n_bending)
        self.bend_dual = self._d(self.n_bending)
        self.incident_offsets = self._i(n + 1)
        self.incident_edges = self._i(max(self.n_backbone * 2, 1))
        self.fixed_nodes = self._i(n)
        self.filopodia = self._i(max(p * max(self.max_arms, 1), 1))
        self.arm_counts = self._i(p)
        self.occ_platelet = self._i(n)
        self.occ_arm = self._i(n)
        self.wlc_x = self._d(n)
        self.wlc_y = self._d(n)
        self.bend_x = self._d(n)
        self.bend_y = self._d(n)
        self.contact_nx = self._d(n)
        self.contact_ny = self._d(n)
        self.contact_px = self._d(p)
        self.contact_py = self._d(p)
        self.move_nx = self._d(n)
        self.move_ny = self._d(n)
        self.move_px = self._d(p)
        self.move_py = self._d(p)
        self.candidate_stride = 160
        self.candidates = self._i(max(n, p * self.candidate_stride, 1))
        self.fiber_ids = self._i(n * max(self.fiber_stride, 1))
        self.fiber_counts = self._i(n)
        self.pair_bit_words = (n * n + 31) // 32 if n else 1
        self.pair_bits = self._i(self.pair_bit_words)
        self.error_flag = self._i(1)
        self.barrier = self._i(3)
        self.partials = self._d(128)
        self.scratch = self._d(4)
        self.result = self._d(17)

    def _upload_static(self, state: SimulationState) -> None:
        pairs = _pairs(state.backbone_pairs, 2) if self.n_backbone else np.zeros((1, 2), np.int32)
        self.runtime.upload(self.backbone_pairs.pointer, pairs.reshape(-1))
        l0 = np.asarray(state.backbone_L0_um, dtype=np.float64)
        if self.n_backbone:
            self.runtime.upload(self.backbone_l0.pointer, l0)
        triples = (
            np.asarray(state.bending_triples, dtype=np.int32).reshape((-1, 3))
            if self.n_bending else np.zeros((1, 3), np.int32)
        )
        self.runtime.upload(self.bend_i.pointer, triples[:, 0])
        self.runtime.upload(self.bend_j.pointer, triples[:, 1])
        self.runtime.upload(self.bend_k.pointer, triples[:, 2])
        if self.n_bending:
            self.runtime.upload(self.bend_phi0.pointer, np.asarray(state.bending_phi0_rad, dtype=np.float64))
            self.runtime.upload(
                self.bend_dual.pointer, np.asarray(state.bending_dual_L0_um, dtype=np.float64)
            )
        offsets = np.zeros(self.n_nodes + 1, dtype=np.int32)
        edges: list[int] = []
        for node, incident in enumerate(state.incident_backbone):
            edges.extend(int(edge) for edge in incident)
            offsets[node + 1] = len(edges)
        self.runtime.upload(self.incident_offsets.pointer, offsets)
        self.runtime.upload(
            self.incident_edges.pointer,
            np.asarray(edges or [0], dtype=np.int32),
        )
        fixed = np.zeros(max(self.n_nodes, 1), dtype=np.int32)
        if len(state.fixed_nodes):
            fixed[: self.n_nodes] = np.asarray(state.fixed_nodes, dtype=bool).astype(np.int32)
        self.runtime.upload(self.fixed_nodes.pointer, fixed[: max(self.n_nodes, 1)])
        counts = np.zeros(max(self.n_platelets, 1), dtype=np.int32)
        for platelet, arms in enumerate(state.filopodia):
            counts[platelet] = len(arms)
        self.runtime.upload(self.arm_counts.pointer, counts[: max(self.n_platelets, 1)])
        fiber_ids = np.full(self.n_nodes * max(self.fiber_stride, 1), -1, dtype=np.int32)
        fiber_counts = np.zeros(max(self.n_nodes, 1), dtype=np.int32)
        for node, fibers in enumerate(state.node_fiber_ids):
            fiber_counts[node] = len(fibers)
            for slot, fiber in enumerate(fibers):
                fiber_ids[node * self.fiber_stride + slot] = int(fiber)
        if self.n_nodes:
            self.runtime.upload(self.fiber_ids.pointer, fiber_ids)
            self.runtime.upload(self.fiber_counts.pointer, fiber_counts[: self.n_nodes])

    def _upload_positions(self, state: SimulationState) -> None:
        nodes = np.asarray(state.nodes_um, dtype=np.float64)
        platelets = np.asarray(state.platelets_um, dtype=np.float64)
        if self.n_nodes:
            self.runtime.upload(self.node_x.pointer, nodes[:, 0])
            self.runtime.upload(self.node_y.pointer, nodes[:, 1])
        if self.n_platelets:
            self.runtime.upload(self.platelet_x.pointer, platelets[:, 0])
            self.runtime.upload(self.platelet_y.pointer, platelets[:, 1])

    def _upload_filopodia(self, state: SimulationState) -> None:
        width = max(self.max_arms, 1)
        table = np.full((max(self.n_platelets, 1), width), -1, dtype=np.int32)
        occupied_p = np.full(max(self.n_nodes, 1), -1, dtype=np.int32)
        occupied_a = np.full(max(self.n_nodes, 1), -1, dtype=np.int32)
        for platelet, arms in enumerate(state.filopodia):
            for arm, node in enumerate(arms):
                if node is not None:
                    table[platelet, arm] = int(node)
                    occupied_p[int(node)] = platelet
                    occupied_a[int(node)] = arm
        self.runtime.upload(self.filopodia.pointer, table.reshape(-1))
        self.runtime.upload(self.occ_platelet.pointer, occupied_p)
        self.runtime.upload(self.occ_arm.pointer, occupied_a)

    def _upload_cohesion(self, state: SimulationState) -> None:
        count = len(state.cohesion_pairs)
        self.cohesion_pairs.ensure(max(count, 1) * 2 * 4)
        self.cohesion_l0.ensure(max(count, 1) * 8)
        if count:
            pairs = np.asarray(state.cohesion_pairs, dtype=np.int32).reshape((-1, 2))
            self.runtime.upload(self.cohesion_pairs.pointer, pairs.reshape(-1))
            self.runtime.upload(
                self.cohesion_l0.pointer, np.asarray(state.cohesion_L0_um, dtype=np.float64)
            )
        self.cohesion_count = count

    def add_cohesion_links(self, state: SimulationState, config: Program19Config) -> int:
        """Find and accept cohesion links on the GPU, in the Python candidate order."""
        if config.cohesion_max_total_neighbors == 0 or self.n_nodes == 0:
            return 0
        radius = float(config.cohesion_distance_um)
        if radius <= 0.0:
            return 0
        capacity = max(self.n_nodes * 64, 1024)
        found = self._gather_cohesion_count(radius * radius, capacity)
        if found == 0:
            return 0
        feasible = self._count_feasible_candidates(state, config, found)
        if feasible == 0:
            return 0
        left, right = self._download_cohesion_candidates(found)
        flags = np.empty(found, dtype=np.int32)
        self.runtime.download(flags, self._coh_flags.pointer)
        keep = flags != 0
        left = left[keep]
        right = right[keep]
        order = np.lexsort((right, left))
        return self._accept_cohesion_candidates(state, config, left[order], right[order])

    def _ensure_buffer(self, name: str, nbytes: int, integer: bool) -> _Buffer:
        buffer = getattr(self, name, None)
        if buffer is None:
            buffer = self._i(1) if integer else self._d(1)
            setattr(self, name, buffer)
        buffer.ensure(max(int(nbytes), 4))
        return buffer

    def _gather_cohesion_count(self, radius2: float, capacity: int) -> int | None:
        out_a = self._ensure_buffer("_coh_out_a", capacity * 4, True)
        out_b = self._ensure_buffer("_coh_out_b", capacity * 4, True)
        out_count = self._ensure_buffer("_coh_count", 4, True)
        self.runtime.upload(out_count.pointer, np.zeros(1, dtype=np.int32))
        self.runtime.launch_grid(self.runtime.cohesion_gather, self.n_nodes, [
            ctypes.c_uint64(self.node_x.pointer), ctypes.c_uint64(self.node_y.pointer),
            ctypes.c_int(self.n_nodes), ctypes.c_double(radius2), ctypes.c_int(capacity),
            ctypes.c_uint64(out_a.pointer), ctypes.c_uint64(out_b.pointer),
            ctypes.c_uint64(out_count.pointer),
        ])
        count = np.zeros(1, dtype=np.int32)
        self.runtime.download(count, out_count.pointer)
        found = int(count[0])
        if found > capacity:
            return self._gather_cohesion_count(radius2, found)
        return found

    def _download_cohesion_candidates(self, found: int) -> tuple[np.ndarray, np.ndarray]:
        left = np.empty(found, dtype=np.int32)
        right = np.empty(found, dtype=np.int32)
        self.runtime.download(left, self._coh_out_a.pointer)
        self.runtime.download(right, self._coh_out_b.pointer)
        return left, right

    def _count_feasible_candidates(
        self, state: SimulationState, config: Program19Config, found: int
    ) -> int:
        self._upload_cohesion(state)
        self.runtime.launch_grid(self.runtime.cohesion_clear_bits, self.pair_bit_words, [
            ctypes.c_uint64(self.pair_bits.pointer), ctypes.c_int(self.pair_bit_words),
        ])
        if self.n_backbone:
            self.runtime.launch_grid(self.runtime.cohesion_set_bits, self.n_backbone, [
                ctypes.c_uint64(self.backbone_pairs.pointer), ctypes.c_int(self.n_backbone),
                ctypes.c_int(self.n_nodes), ctypes.c_uint64(self.pair_bits.pointer),
            ])
        if self.cohesion_count:
            self.runtime.launch_grid(self.runtime.cohesion_set_bits, self.cohesion_count, [
                ctypes.c_uint64(self.cohesion_pairs.pointer), ctypes.c_int(self.cohesion_count),
                ctypes.c_int(self.n_nodes), ctypes.c_uint64(self.pair_bits.pointer),
            ])
        degree = cohesion_degrees(state).astype(np.int32)
        if config.cohesion_degree_includes_backbone:
            degree = degree + np.asarray(
                [len(neighbors) for neighbors in state.original_adjacency], dtype=np.int32
            )
        degree_buf = self._ensure_buffer("_coh_degree", self.n_nodes * 4, True)
        flags = self._ensure_buffer("_coh_flags", found * 4, True)
        self.runtime.upload(degree_buf.pointer, degree)
        self.runtime.launch_grid(self.runtime.cohesion_mark, found, [
            ctypes.c_uint64(self._coh_out_a.pointer), ctypes.c_uint64(self._coh_out_b.pointer),
            ctypes.c_int(found), ctypes.c_uint64(self.pair_bits.pointer), ctypes.c_int(self.n_nodes),
            ctypes.c_uint64(self.fiber_ids.pointer), ctypes.c_uint64(self.fiber_counts.pointer),
            ctypes.c_int(self.fiber_stride), ctypes.c_uint64(degree_buf.pointer),
            ctypes.c_int(int(config.cohesion_max_total_neighbors)),
            ctypes.c_int(int(config.cohesion_exclude_same_fiber)),
            ctypes.c_uint64(flags.pointer),
        ])
        counted = self._ensure_buffer("_coh_flag_count", 4, True)
        self.runtime.launch_block(self.runtime.cohesion_count, 256, [
            ctypes.c_uint64(flags.pointer), ctypes.c_int(found), ctypes.c_uint64(counted.pointer),
        ])
        total = np.zeros(1, dtype=np.int32)
        self.runtime.download(total, counted.pointer)
        return int(total[0])

    def _accept_cohesion_candidates(
        self,
        state: SimulationState,
        config: Program19Config,
        left: np.ndarray,
        right: np.ndarray,
    ) -> int:
        count = int(len(left))
        cand_a = self._ensure_buffer("_coh_cand_a", count * 4, True)
        cand_b = self._ensure_buffer("_coh_cand_b", count * 4, True)
        self.runtime.upload(cand_a.pointer, np.ascontiguousarray(left))
        self.runtime.upload(cand_b.pointer, np.ascontiguousarray(right))
        degree = cohesion_degrees(state).astype(np.int32)
        if config.cohesion_degree_includes_backbone:
            degree = degree + np.asarray(
                [len(neighbors) for neighbors in state.original_adjacency], dtype=np.int32
            )
        degree_buf = self._ensure_buffer("_coh_degree", self.n_nodes * 4, True)
        fresh = self._ensure_buffer("_coh_fresh", self.n_nodes * 4, True)
        self.runtime.upload(degree_buf.pointer, degree)
        self.runtime.upload(fresh.pointer, np.zeros(self.n_nodes, dtype=np.int32))
        room = self.n_nodes * max(int(config.cohesion_max_new_per_node_per_search), 1)
        accepted_a = self._ensure_buffer("_coh_acc_a", room * 4, True)
        accepted_b = self._ensure_buffer("_coh_acc_b", room * 4, True)
        accepted_l = self._ensure_buffer("_coh_acc_l", room * 8, False)
        accepted_n = self._ensure_buffer("_coh_acc_n", 4, True)
        self.runtime.launch_block(self.runtime.cohesion_accept, 1, [
            ctypes.c_uint64(self.node_x.pointer), ctypes.c_uint64(self.node_y.pointer),
            ctypes.c_uint64(cand_a.pointer), ctypes.c_uint64(cand_b.pointer),
            ctypes.c_int(count),
            ctypes.c_uint64(degree_buf.pointer), ctypes.c_uint64(fresh.pointer),
            ctypes.c_int(int(config.cohesion_max_total_neighbors)),
            ctypes.c_int(int(config.cohesion_max_new_per_node_per_search)),
            ctypes.c_uint64(accepted_a.pointer), ctypes.c_uint64(accepted_b.pointer),
            ctypes.c_uint64(accepted_l.pointer), ctypes.c_uint64(accepted_n.pointer),
        ])
        created = np.zeros(1, dtype=np.int32)
        self.runtime.download(created, accepted_n.pointer)
        new_links = int(created[0])
        if new_links > room:
            raise CudaMechanicsError("Cohesion acceptance exceeded the output buffer")
        if new_links:
            chosen_a = np.empty(new_links, dtype=np.int32)
            chosen_b = np.empty(new_links, dtype=np.int32)
            lengths = np.empty(new_links, dtype=np.float64)
            self.runtime.download(chosen_a, accepted_a.pointer)
            self.runtime.download(chosen_b, accepted_b.pointer)
            self.runtime.download(lengths, accepted_l.pointer)
            birth = float(state.time_s)
            for first, second, length in zip(chosen_a, chosen_b, lengths):
                key = (int(first), int(second))
                state.cohesion_pairs.append(key)
                state.cohesion_L0_um.append(float(length))
                state.cohesion_birth_time_s.append(birth)
                state.cohesion_pair_set.add(key)
        return new_links

    def download(self, state: SimulationState) -> None:
        if self.n_nodes:
            nodes = np.asarray(state.nodes_um, dtype=np.float64)
            self.runtime.download(nodes[:, 0], self.node_x.pointer)
            self.runtime.download(nodes[:, 1], self.node_y.pointer)
            state.nodes_um = nodes
        if self.n_platelets:
            platelets = np.asarray(state.platelets_um, dtype=np.float64)
            self.runtime.download(platelets[:, 0], self.platelet_x.pointer)
            self.runtime.download(platelets[:, 1], self.platelet_y.pointer)
            state.platelets_um = platelets
        if self.n_platelets and self.max_arms:
            table = np.empty(self.n_platelets * self.max_arms, dtype=np.int32)
            self.runtime.download(table, self.filopodia.pointer)
            table = table.reshape((self.n_platelets, self.max_arms))
            filopodia: list[list[int | None]] = []
            occupied: dict[int, tuple[int, int]] = {}
            for platelet, arms in enumerate(state.filopodia):
                row: list[int | None] = []
                for arm in range(len(arms)):
                    node = int(table[platelet, arm])
                    if node < 0:
                        row.append(None)
                    else:
                        row.append(node)
                        occupied[node] = (platelet, arm)
                filopodia.append(row)
            state.filopodia = filopodia
            state.node_occupied_by = occupied
        state.cuda_noise_substep = self.noise_substeps

    def integrate_chunk(
        self,
        config: Program19Config,
        *,
        interval_s: float,
        remaining: float,
        substep_origin: int,
    ) -> np.ndarray:
        node_drag, platelet_drag, node_d, platelet_d = mobility_and_diffusion(config)
        prefactor = (
            config.monomers_in_parallel_N * kbt_nN_um(config) / config.persistence_length_um
        )
        core_diameter = 2.0 * config.platelet_radius_um * config.platelet_core_fraction
        codeocean = config.equation_profile == "codeocean_literal"
        sigma = core_diameter if codeocean else core_diameter / 2.0 ** (1.0 / 6.0)
        seed = (np.uint64(config.seed).item() * np.uint64(0x9E3779B97F4A7C15).item()) & 0xFFFFFFFFFFFFFFFF
        arguments = [
            ctypes.c_uint64(self.node_x.pointer), ctypes.c_uint64(self.node_y.pointer),
            ctypes.c_uint64(self.platelet_x.pointer), ctypes.c_uint64(self.platelet_y.pointer),
            ctypes.c_uint64(self.backbone_pairs.pointer), ctypes.c_uint64(self.backbone_l0.pointer),
            ctypes.c_uint64(self.cohesion_pairs.pointer), ctypes.c_uint64(self.cohesion_l0.pointer),
            ctypes.c_uint64(self.bend_i.pointer), ctypes.c_uint64(self.bend_j.pointer),
            ctypes.c_uint64(self.bend_k.pointer), ctypes.c_uint64(self.bend_phi0.pointer),
            ctypes.c_uint64(self.bend_dual.pointer),
            ctypes.c_uint64(self.incident_offsets.pointer), ctypes.c_uint64(self.incident_edges.pointer),
            ctypes.c_uint64(self.fixed_nodes.pointer),
            ctypes.c_uint64(self.filopodia.pointer), ctypes.c_uint64(self.arm_counts.pointer),
            ctypes.c_uint64(self.occ_platelet.pointer), ctypes.c_uint64(self.occ_arm.pointer),
            ctypes.c_uint64(self.wlc_x.pointer), ctypes.c_uint64(self.wlc_y.pointer),
            ctypes.c_uint64(self.bend_x.pointer), ctypes.c_uint64(self.bend_y.pointer),
            ctypes.c_uint64(self.contact_nx.pointer), ctypes.c_uint64(self.contact_ny.pointer),
            ctypes.c_uint64(self.contact_px.pointer), ctypes.c_uint64(self.contact_py.pointer),
            ctypes.c_uint64(self.move_nx.pointer), ctypes.c_uint64(self.move_ny.pointer),
            ctypes.c_uint64(self.move_px.pointer), ctypes.c_uint64(self.move_py.pointer),
            ctypes.c_uint64(self.candidates.pointer), ctypes.c_uint64(self.error_flag.pointer),
            ctypes.c_int(self.n_nodes), ctypes.c_int(self.n_platelets),
            ctypes.c_int(self.n_backbone), ctypes.c_int(self.cohesion_count),
            ctypes.c_int(self.n_bending), ctypes.c_int(self.max_arms),
            ctypes.c_int(int(codeocean)),
            ctypes.c_int(int(config.filopodia_global_node_exclusivity)),
            ctypes.c_int(int(config.filopodia_force_distribution == "divide_by_attached")),
            ctypes.c_int(int(config.brownian_motion)),
            ctypes.c_int(int(config.adaptive_substeps)),
            ctypes.c_int(int(config.remove_global_translation)),
            ctypes.c_double(prefactor), ctypes.c_double(config.contour_multiplier_C),
            ctypes.c_double(config.gamma_max), ctypes.c_double(config.bending_EI_nN_um2),
            ctypes.c_double(config.archive_bending_force_nN),
            ctypes.c_double(config.platelet_core_fraction * config.platelet_radius_um + config.fiber_radius_um),
            ctypes.c_double(config.platelet_radius_um + config.fiber_radius_um),
            ctypes.c_double(config.platelet_fibrin_force_nN),
            ctypes.c_double(config.platelet_radius_um + config.fiber_radius_um),
            ctypes.c_double(config.platelet_force_radius_um),
            ctypes.c_double(config.filopodia_f0_nN), ctypes.c_double(config.filopodia_a),
            ctypes.c_double(config.filopodia_b), ctypes.c_double(config.filopodia_c),
            ctypes.c_double(sigma), ctypes.c_double(2.0 * config.platelet_radius_um),
            ctypes.c_double(config.platelet_platelet_force_parameter),
            ctypes.c_int(int(codeocean)),
            ctypes.c_double(node_drag), ctypes.c_double(platelet_drag),
            ctypes.c_double(node_d), ctypes.c_double(platelet_d),
            ctypes.c_double(config.adaptive_target_displacement_um),
            ctypes.c_double(config.abort_displacement_um),
            ctypes.c_double(interval_s), ctypes.c_double(remaining),
            ctypes.c_int(_CHUNK), ctypes.c_int(int(substep_origin)),
            ctypes.c_int(int(config.max_internal_substeps)),
            ctypes.c_uint64(seed),
            ctypes.c_uint64(self.barrier.pointer),
            ctypes.c_uint64(self.barrier.pointer + 4),
            ctypes.c_uint64(self.barrier.pointer + 8),
            ctypes.c_uint64(self.partials.pointer),
            ctypes.c_uint64(self.scratch.pointer),
            ctypes.c_uint64(self.result.pointer),
        ]
        self.runtime.upload(self.barrier.pointer, np.zeros(3, dtype=np.int32))
        self.runtime.upload(self.scratch.pointer, np.zeros(4, dtype=np.float64))
        filo_inner = config.platelet_radius_um + config.fiber_radius_um
        self.runtime.launch_block(self.runtime.filopodia, 128, [
            ctypes.c_uint64(self.node_x.pointer), ctypes.c_uint64(self.node_y.pointer),
            ctypes.c_uint64(self.platelet_x.pointer), ctypes.c_uint64(self.platelet_y.pointer),
            ctypes.c_uint64(self.filopodia.pointer), ctypes.c_uint64(self.arm_counts.pointer),
            ctypes.c_uint64(self.occ_platelet.pointer), ctypes.c_uint64(self.occ_arm.pointer),
            ctypes.c_uint64(self.candidates.pointer),
            ctypes.c_int(self.n_nodes), ctypes.c_int(self.n_platelets), ctypes.c_int(self.max_arms),
            ctypes.c_int(int(config.filopodia_global_node_exclusivity)),
            ctypes.c_double(filo_inner), ctypes.c_double(config.platelet_force_radius_um),
            ctypes.c_int(self.candidate_stride),
        ])
        self.runtime.launch(arguments)
        out = np.empty(17, dtype=np.float64)
        self.runtime.download(out, self.result.pointer)
        return out


def _session(state: SimulationState, config: Program19Config) -> CudaSession:
    session = getattr(state, "_cuda_session", None)
    if session is None or session.n_nodes != len(state.nodes_um):
        session = CudaSession(state, config)
        state._cuda_session = session
    return session


def integrate_interval_cuda(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator,
    interval_s: float,
) -> StepDiagnostics:
    """Integrate one outer interval on the GPU. ``rng`` is unused by this backend."""
    del rng
    _require_supported(config)
    session = _session(state, config)
    new_links = 0
    if state.step % config.cohesion_search_stride == 0:
        new_links = session.add_cohesion_links(state, config)
    session._upload_cohesion(state)
    node_drag, platelet_drag, node_d, platelet_d = mobility_and_diffusion(config)
    remaining = float(interval_s)
    epsilon = max(1e-15, interval_s * 1e-12)
    substeps = 0
    maximum_node_move = 0.0
    maximum_platelet_move = 0.0
    last = None
    try:
        while remaining > epsilon:
            if substeps >= config.max_internal_substeps:
                raise RuntimeError(
                    f"Adaptive integrator exceeded {config.max_internal_substeps} substeps "
                    f"during an outer interval of {interval_s:g} s. No displacement was clipped."
                )
            last = session.integrate_chunk(
                config,
                interval_s=interval_s,
                remaining=remaining,
                substep_origin=session.noise_substeps,
            )
            chunk_steps = int(last[0])
            status = int(last[1])
            session.noise_substeps += chunk_steps
            substeps += chunk_steps
            remaining = float(last[2])
            maximum_node_move = max(maximum_node_move, float(last[3]))
            maximum_platelet_move = max(maximum_platelet_move, float(last[4]))
            if status == 2:
                raise RuntimeError(
                    f"Adaptive integrator exceeded {config.max_internal_substeps} substeps "
                    f"during an outer interval of {interval_s:g} s. No displacement was clipped."
                )
            if status == 3:
                raise FloatingPointError("Adaptive integration produced an invalid substep")
            if status == 4:
                raise RuntimeError(
                    f"Safety stop: a substep requested {float(last[16]):.6g} um, above "
                    f"abort_displacement_um={config.abort_displacement_um:g}. No clip was applied."
                )
            if status == 5:
                raise RuntimeError(
                    "A platelet contact is singular; the CUDA integrator did not move that substep."
                )
            if status != 0:
                raise CudaMechanicsError(f"Unexpected CUDA integrator status {status}")
            if chunk_steps <= 0:
                raise CudaMechanicsError("CUDA integrator made no progress on a remaining interval")
    finally:
        session.download(state)
    if last is None or substeps == 0:
        current = ForceDiagnostics()
    else:
        minimum = float(last[15])
        current = ForceDiagnostics(
            wlc_rms_nN=float(last[5]),
            bending_rms_nN=float(last[6]),
            contact_rms_nN=float(last[7]),
            filopodia_rms_nN=0.0,
            max_backbone_strain=float(last[8]),
            max_total_node_force_nN=float(last[9]),
            max_total_platelet_force_nN=float(last[10]),
            net_internal_force_nN=float(math.hypot(float(last[11]), float(last[12]))),
            attached_filopodia=int(last[13]),
            platelet_overlap_count=int(last[14]),
            minimum_platelet_center_distance_um=(
                float("inf") if minimum >= 1e299 else minimum
            ),
        )
    state.time_s += float(interval_s)
    state.step += 1
    degree = cohesion_degrees(state)
    return StepDiagnostics(
        **current.__dict__,
        new_cohesion_links=new_links,
        total_cohesion_links=len(state.cohesion_pairs),
        maximum_cohesion_degree=int(np.max(degree)) if len(degree) else 0,
        max_node_displacement_um=maximum_node_move,
        max_platelet_displacement_um=maximum_platelet_move,
        internal_substeps=substeps,
        node_diffusion_um2_s=node_d,
        platelet_diffusion_um2_s=platelet_d,
    )
