import json
import os
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_cache_key = None
_cache_sess = None
_cache_meta = None


def _ctc_decode_indices(indices, charset) -> str:
    """CTC: collapse repeats, drop blank(0)."""
    decoded = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx != prev and idx != 0:
            if 0 <= idx < len(charset):
                decoded.append(str(charset[idx]))
        prev = idx
    return "".join(decoded)


def _preprocess_image(image_path: str, resize, channel: int) -> np.ndarray:
    """Match training: resize → ToTensor(/255) → Normalize."""
    mode = "L" if int(channel) == 1 else "RGB"
    image = Image.open(image_path).convert(mode)
    width_cfg, height_cfg = int(resize[0]), int(resize[1])
    iw, ih = image.size
    if width_cfg == -1:
        image = image.resize((int(iw * (height_cfg / ih)), height_cfg))
    else:
        image = image.resize((width_cfg, height_cfg))

    arr = np.asarray(image).astype(np.float32) / 255.0
    if int(channel) == 1:
        # training: Normalize(mean=[0.456], std=[0.224])
        arr = (arr - 0.456) / 0.224
        arr = arr[None, None, :, :]
    else:
        # training: ImageNet mean/std, CHW
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = arr.transpose(2, 0, 1)
        arr = (arr - mean) / std
        arr = arr[None, :, :, :]
    return arr


def _decode_output(output: np.ndarray, charset) -> str:
    """
    Support:
    - logits: (T, N, C) or (N, T, C)
    - already argmax indices: (N, T) or (T,)  [current trainer export]
    """
    if output.ndim == 3:
        # prefer (T, 1, C) from CRNN
        if output.shape[1] == 1:
            indices = np.argmax(output[:, 0, :], axis=1)
        elif output.shape[0] == 1:
            indices = np.argmax(output[0, :, :], axis=1)
        else:
            indices = np.argmax(output[:, 0, :], axis=1)
        return _ctc_decode_indices(indices, charset)

    if output.ndim == 2:
        # (N, T) class indices from Net.forward export
        return _ctc_decode_indices(output[0], charset)

    if output.ndim == 1:
        return _ctc_decode_indices(output, charset)

    raise ValueError(f"不支持的模型输出形状: {output.shape}")


def load_project_session(onnx_path: str, charsets_path: str):
    """Load onnxruntime session + charsets meta (cached)."""
    global _cache_key, _cache_sess, _cache_meta
    key = (os.path.abspath(onnx_path), os.path.abspath(charsets_path))
    if _cache_sess is not None and _cache_key == key:
        return _cache_sess, _cache_meta

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"模型不存在: {onnx_path}")
    if not os.path.isfile(charsets_path):
        raise FileNotFoundError(
            f"缺少 charsets.json: {charsets_path}\n"
            "请确认训练已导出 onnx（与 models/charsets.json 同目录）。"
        )

    with open(charsets_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for required in ("charset", "image", "channel"):
        if required not in meta:
            raise ValueError(f"charsets.json 缺少字段: {required}")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    _cache_key = key
    _cache_sess = sess
    _cache_meta = meta
    return sess, meta


def clear_ocr_cache():
    global _cache_key, _cache_sess, _cache_meta
    _cache_key = None
    _cache_sess = None
    _cache_meta = None


def predict_image(onnx_path: str, charsets_path: str, image_path: str) -> str:
    sess, meta = load_project_session(onnx_path, charsets_path)
    arr = _preprocess_image(image_path, meta["image"], meta["channel"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: arr})[0]
    return _decode_output(output, meta["charset"]).strip()


# backward-compatible alias used by older call sites
def load_project_ocr(onnx_path: str, charsets_path: str):
    load_project_session(onnx_path, charsets_path)
    return None


def parse_expected_from_name(filename: str) -> str:
    """Parse label from label_hash.ext; empty if not matched."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in stem:
        return ""
    parts = stem.split("_")
    maybe_hash = parts[-1]
    # short hashes like 6e325c17 (8) or longer md5
    if len(maybe_hash) >= 6 and re.fullmatch(r"[0-9a-fA-F]+", maybe_hash):
        return "_".join(parts[:-1])
    return ""


def compare_result(pred: str, expected: str) -> Tuple[bool, str]:
    expected = (expected or "").replace(" ", "")
    pred = (pred or "").replace(" ", "")
    if not expected:
        return False, "无期望标签"
    ok = pred == expected
    return ok, "正确" if ok else f"错误 (期望={expected}, 预测={pred})"
