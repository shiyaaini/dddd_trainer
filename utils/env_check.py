"""Environment diagnostics and install command helpers."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# (显示名, import 名, 是否训练必需, pip 包名提示)
PACKAGE_CHECKS: List[Tuple[str, str, bool, str]] = [
    ("Python", "sys", True, ""),
    ("PyTorch", "torch", True, "torch"),
    ("PyYAML", "yaml", True, "pyyaml"),
    ("NumPy", "numpy", True, "numpy<2"),
    ("Pillow", "PIL", True, "pillow==9.5.0"),
    ("tqdm", "tqdm", True, "tqdm"),
    ("loguru", "loguru", True, "loguru"),
    ("Fire", "fire", True, "fire"),
    ("PyQt6", "PyQt6", True, "PyQt6"),
    ("OpenCV", "cv2", False, "opencv-python<4.10"),
    ("ONNX", "onnx", False, "onnx"),
    ("onnxruntime", "onnxruntime", False, "onnxruntime"),
    ("ddddocr", "ddddocr", False, "ddddocr"),
]


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str
    required: bool = True
    fix_hint: str = ""


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mem_mb: Optional[int] = None


@dataclass
class EnvReport:
    python_version: str
    python_executable: str
    platform: str
    items: List[CheckItem] = field(default_factory=list)
    gpus: List[GpuInfo] = field(default_factory=list)
    nvidia_smi_ok: bool = False
    driver_version: str = ""
    cuda_from_driver: str = ""  # e.g. "12.6"
    torch_version: str = ""
    torch_cuda_built: str = ""  # e.g. "12.4" or ""
    torch_cuda_available: bool = False
    recommended_torch_tag: str = "cpu"  # cu124 / cu121 / cu118 / cpu
    recommended_commands: List[str] = field(default_factory=list)
    summary: str = ""


def _run_cmd(args: List[str], timeout: int = 12) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
        )
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output or str(e)
    except Exception as e:
        return 1, str(e)


def _parse_cuda_major_minor(text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d+)\.(\d+)", text or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def recommend_torch_cuda_tag(driver_cuda: str) -> str:
    """
    Map nvidia-smi reported CUDA version to a PyTorch wheel tag.
    Conservative: pick a widely available wheel that the driver can run.
    """
    parsed = _parse_cuda_major_minor(driver_cuda)
    if not parsed:
        return "cpu"
    major, minor = parsed
    # Driver advertises max supported CUDA toolkit for apps / UMD
    if major >= 13 or (major == 12 and minor >= 4):
        return "cu124"
    if major == 12 and minor >= 1:
        return "cu121"
    if major == 12 or (major == 11 and minor >= 8):
        return "cu118"
    if major == 11:
        return "cu118"
    return "cpu"


def torch_index_url(tag: str) -> str:
    if tag == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    return f"https://download.pytorch.org/whl/{tag}"


def detect_nvidia():
    code, out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if code == 0:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            mem = None
            if len(parts) >= 3:
                try:
                    mem = int(float(parts[2]))
                except ValueError:
                    mem = None
            gpus.append(GpuInfo(index=idx, name=parts[1], total_mem_mb=mem))

    code2, out2 = _run_cmd(["nvidia-smi"])
    driver = ""
    cuda_ver = ""
    if code2 == 0:
        # Classic: "Driver Version: 551.23    CUDA Version: 12.4"
        m_drv = re.search(r"Driver Version:\s*([0-9.]+)", out2)
        m_cuda = re.search(r"CUDA Version:\s*([0-9.]+)", out2)
        # Newer Windows KMD header: "NVIDIA-SMI 610.88 ... CUDA UMD Version: 13.3"
        m_kmd = re.search(r"KMD Version:\s*([0-9.]+)", out2)
        m_umd = re.search(r"CUDA UMD Version:\s*([0-9.]+)", out2)
        m_smi = re.search(r"NVIDIA-SMI\s+([0-9.]+)", out2)
        if m_drv:
            driver = m_drv.group(1)
        elif m_kmd:
            driver = m_kmd.group(1)
        elif m_smi:
            driver = m_smi.group(1)
        if m_cuda:
            cuda_ver = m_cuda.group(1)
        elif m_umd:
            cuda_ver = m_umd.group(1)

    ok = bool(gpus) or (code2 == 0)
    return ok, driver, cuda_ver, gpus


def _check_python() -> CheckItem:
    ver = platform.python_version()
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 8) and (major, minor) < (3, 13)
    detail = f"{ver} ({sys.executable})"
    hint = ""
    if not ok:
        hint = "建议使用 Python 3.10 或 3.11（与本仓库启动脚本一致）"
    return CheckItem("Python", ok, detail, True, hint)


def _check_import(display: str, module: str, required: bool, pip_name: str) -> CheckItem:
    if module == "sys":
        return _check_python()
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", None)
        if ver is None and module == "PIL":
            ver = getattr(mod, "PILLOW_VERSION", None) or getattr(getattr(mod, "Image", None), "__version__", None)
        detail = f"已安装" + (f"  v{ver}" if ver else "")
        return CheckItem(display, True, detail, required, "")
    except Exception as e:
        hint = f"pip install {pip_name}" if pip_name else str(e)
        return CheckItem(display, False, f"未安装 / 导入失败: {e}", required, hint)


def _check_torch_detail(item: CheckItem) -> Tuple[CheckItem, str, str, bool]:
    torch_ver = ""
    cuda_built = ""
    cuda_ok = False
    if not item.ok:
        return item, torch_ver, cuda_built, cuda_ok
    try:
        import torch
        torch_ver = torch.__version__
        cuda_built = getattr(torch.version, "cuda", None) or ""
        cuda_ok = bool(torch.cuda.is_available())
        parts = [f"v{torch_ver}"]
        if cuda_built:
            parts.append(f"编译 CUDA {cuda_built}")
        else:
            parts.append("CPU 版")
        if cuda_ok:
            parts.append(f"可用 GPU x{torch.cuda.device_count()}")
            try:
                parts.append(torch.cuda.get_device_name(0))
            except Exception:
                pass
        else:
            parts.append("当前进程无法使用 CUDA")
        item.detail = " | ".join(parts)
        if not cuda_built and cuda_ok is False:
            item.fix_hint = "检测到可能是 CPU 版 PyTorch，N 卡训练请安装 CUDA 版"
        elif cuda_built and not cuda_ok:
            item.ok = False
            item.fix_hint = "已装 CUDA 版 torch 但 cuda.is_available()=False，请检查驱动 / CUDA"
            item.required = True
    except Exception as e:
        item.ok = False
        item.detail = str(e)
    return item, torch_ver, cuda_built, cuda_ok


def build_pip_commands(
    tag: str,
    *,
    install_reqs: bool = True,
    reinstall_torch: bool = True,
    mirror: str = "https://pypi.tuna.tsinghua.edu.cn/simple",
) -> List[str]:
    """Return shell commands (one logical step per string) for the current interpreter."""
    py = sys.executable
    cmds: List[str] = []
    cmds.append(f'"{py}" -m pip install --upgrade pip')
    if install_reqs:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        req = os.path.join(root, "requirements.txt")
        if os.path.isfile(req):
            cmds.append(f'"{py}" -m pip install -r "{req}" -i {mirror}')
    if reinstall_torch:
        index = torch_index_url(tag)
        # numpy must exist before torch; prefer offline Windows wheel (not .tar.gz)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        numpy_whl = os.path.join(
            root, "dist", "numpy-1.24.4-cp310-cp310-win_amd64.whl"
        )
        if os.path.isfile(numpy_whl):
            cmds.append(f'"{py}" -m pip install "{numpy_whl}" --no-cache-dir')
        else:
            cmds.append(f'"{py}" -m pip install "numpy==1.24.4" -i {mirror}')
        # uninstall first to avoid leftover cpu wheels
        cmds.append(f'"{py}" -m pip uninstall -y torch torchvision torchaudio')
        cmds.append(
            f'"{py}" -m pip install torch torchvision torchaudio --index-url {index}'
        )
    return cmds


def collect_env_report() -> EnvReport:
    report = EnvReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )

    for display, module, required, pip_name in PACKAGE_CHECKS:
        item = _check_import(display, module, required, pip_name)
        if display == "PyTorch":
            item, tv, tb, ta = _check_torch_detail(item)
            report.torch_version = tv
            report.torch_cuda_built = tb or ""
            report.torch_cuda_available = ta
        report.items.append(item)

    smi_ok, driver, cuda_ver, gpus = detect_nvidia()
    report.nvidia_smi_ok = smi_ok
    report.driver_version = driver
    report.cuda_from_driver = cuda_ver
    report.gpus = gpus

    if smi_ok and cuda_ver:
        report.recommended_torch_tag = recommend_torch_cuda_tag(cuda_ver)
    elif report.torch_cuda_built and report.torch_cuda_available:
        # nvidia-smi 格式异常时，用当前可用的 torch CUDA 反推
        built = report.torch_cuda_built
        report.recommended_torch_tag = recommend_torch_cuda_tag(built)
        if report.recommended_torch_tag == "cpu":
            report.recommended_torch_tag = "cu124"
    elif smi_ok:
        report.recommended_torch_tag = "cu124"
    else:
        report.recommended_torch_tag = "cpu"

    # If torch already has matching CUDA, still expose reinstall cmds for repair
    report.recommended_commands = build_pip_commands(report.recommended_torch_tag)

    missing_required = [i for i in report.items if i.required and not i.ok]
    torch_item = next((i for i in report.items if i.name == "PyTorch"), None)

    lines = []
    if missing_required:
        lines.append(f"缺少必需依赖 {len(missing_required)} 项")
    else:
        lines.append("必需依赖已齐全")

    if report.gpus:
        names = ", ".join(f"[{g.index}] {g.name}" for g in report.gpus)
        lines.append(f"检测到 GPU: {names}")
        if report.driver_version or report.cuda_from_driver:
            lines.append(
                f"驱动 {report.driver_version or '?'} / 驱动声明 CUDA {report.cuda_from_driver or '?'}"
            )
        lines.append(f"建议安装 PyTorch 轮子: {report.recommended_torch_tag}")
        if torch_item and torch_item.ok and not report.torch_cuda_available:
            lines.append("警告: 有 N 卡但当前 torch 无法使用 CUDA，请点「安装/修复 PyTorch」")
        elif report.torch_cuda_available:
            lines.append("PyTorch CUDA 可用，可开始训练")
    else:
        lines.append("未检测到 NVIDIA GPU（nvidia-smi 不可用），建议安装 CPU 版 torch 或检查驱动")

    report.summary = "；".join(lines)
    return report


def format_commands_for_display(cmds: List[str]) -> str:
    if os.name == "nt":
        return "\r\n".join(cmds)
    return "\n".join(cmds)
