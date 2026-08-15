import hashlib
import math
import os
import random
import re
import uuid
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QThread, pyqtSignal


DEFAULT_TEMPLATE = '''# =============================================================================
# 批量生成脚本说明（给用户 / AI）
# -----------------------------------------------------------------------------
# 1) 必须定义: generate(index: int) -> (PIL.Image.Image, label: str)
# 2) 框架会循环 index=0..N-1 调用，并把图片保存到所选项目:
#      projects/{项目名}/datasets/{label}_{8位hash}.{png|jpg|bmp}
# 3) label 不能含空格或 \\ / : * ? " < > | ；保存后可直接用于「文件名标注」缓存
# 4) 预注入可用: Image, ImageDraw, ImageFont, random, math, os, hashlib
# 5) 禁止危险 import（subprocess/socket/sys 等会被拦截）
# 6) 生成结果视为「已标注完成样本」，进入 datasets（完成队列），不是 inbox
# =============================================================================

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def generate(index: int):
    """生成一张验证码图；返回 (图片, 标签文本)。index 为当前序号。"""
    width, height = 128, 48
    length = 4
    label = "".join(random.choice(CHARS) for _ in range(length))

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    # 干扰线
    for _ in range(3):
        draw.line(
            (
                random.randint(0, width), random.randint(0, height),
                random.randint(0, width), random.randint(0, height),
            ),
            fill=(random.randint(100, 200),) * 3,
            width=1,
        )

    # 文字
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(2, (width - tw) // 2)
    y = max(2, (height - th) // 2)
    draw.text((x, y), label, fill=(0, 0, 0), font=font)

    # 噪点
    for _ in range(80):
        draw.point(
            (random.randint(0, width - 1), random.randint(0, height - 1)),
            fill=(random.randint(0, 255),) * 3,
        )

    return img, label
'''

_BLOCKED_MODULES = {
    "subprocess", "socket", "ctypes", "multiprocessing", "shutil",
    "pathlib", "http", "urllib", "requests", "pickle", "importlib",
    "sys", "builtins", "code", "codeop", "pty", "signal",
}

_INVALID_LABEL_CHARS = re.compile(r'[\\/:*?"<>|\s]')


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _BLOCKED_MODULES:
        raise ImportError(f"不允许导入模块: {name}")
    return __import__(name, globals, locals, fromlist, level)


def build_namespace():
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "print": print, "range": range,
        "repr": repr, "round": round, "set": set, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "zip": zip, "True": True, "False": False,
        "None": None, "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "RuntimeError": RuntimeError, "OSError": OSError,
        "__import__": _safe_import,
    }
    return {
        "__builtins__": safe_builtins,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "random": random,
        "math": math,
        "os": os,
        "hashlib": hashlib,
    }


def compile_generate(source: str):
    namespace = build_namespace()
    code = compile(source, "<user_generate>", "exec")
    exec(code, namespace, namespace)
    generate_fn = namespace.get("generate")
    if not callable(generate_fn):
        raise ValueError("代码中必须定义可调用的 generate(index) 函数")
    return generate_fn


def sanitize_label(label: str) -> str:
    label = str(label).replace(" ", "")
    if not label:
        return ""
    if _INVALID_LABEL_CHARS.search(label):
        return ""
    return label


def make_filename(label: str, ext: str) -> str:
    digest = hashlib.md5(uuid.uuid4().bytes).hexdigest()[:8]
    return f"{label}_{digest}.{ext}"


def run_once(generate_fn, index: int) -> Tuple[Image.Image, str]:
    result = generate_fn(index)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("generate(index) 必须返回 (Image, label)")
    image, label = result
    if not isinstance(image, Image.Image):
        raise TypeError("generate 返回的第一项必须是 PIL.Image.Image")
    label = sanitize_label(label)
    if not label:
        raise ValueError("label 为空或包含非法字符/空格")
    return image, label


class GenerateWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    log = pyqtSignal(str)
    preview = pyqtSignal(object, str)  # Image, label
    finished_ok = pyqtSignal(int, int)  # success, failed
    failed = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        output_dir: str,
        count: int,
        ext: str = "png",
        preview_first: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.output_dir = output_dir
        self.count = count
        self.ext = ext.lstrip(".").lower()
        self.preview_first = preview_first
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        success = 0
        failed = 0
        try:
            generate_fn = compile_generate(self.source)
        except Exception as e:
            self.failed.emit(f"代码编译/执行失败: {e}")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        if self.ext not in ("png", "jpg", "jpeg", "bmp"):
            self.failed.emit(f"不支持的图片格式: {self.ext}")
            return

        save_kwargs = {}
        if self.ext in ("jpg", "jpeg"):
            save_kwargs["quality"] = 95

        for i in range(self.count):
            if self._stop:
                self.log.emit("已停止生成。")
                break
            try:
                image, label = run_once(generate_fn, i)
                filename = make_filename(label, "jpg" if self.ext == "jpeg" else self.ext)
                path = os.path.join(self.output_dir, filename)
                to_save = image
                if self.ext in ("jpg", "jpeg") and image.mode not in ("RGB", "L"):
                    to_save = image.convert("RGB")
                to_save.save(path, **save_kwargs)
                success += 1
                if self.preview_first and i == 0:
                    self.preview.emit(image.copy(), label)
                if (i + 1) % 50 == 0 or i == 0:
                    self.log.emit(f"已保存 [{i + 1}/{self.count}] {filename}")
            except Exception as e:
                failed += 1
                self.log.emit(f"第 {i} 张失败: {e}")
            self.progress.emit(i + 1, self.count)

        self.finished_ok.emit(success, failed)


def preview_one(source: str, index: int = 0) -> Tuple[Image.Image, str]:
    generate_fn = compile_generate(source)
    return run_once(generate_fn, index)
