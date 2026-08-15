import threading
from typing import Optional, Union

_lock = threading.Lock()
_ocr = None
_import_error: Optional[str] = None


def get_ocr():
    """Lazy-load a shared ddddocr.DdddOcr instance."""
    global _ocr, _import_error
    if _ocr is not None:
        return _ocr
    with _lock:
        if _ocr is not None:
            return _ocr
        try:
            import ddddocr
            # show_ad=False avoids banner noise in GUI logs
            _ocr = ddddocr.DdddOcr(show_ad=False)
        except Exception as e:
            _import_error = str(e)
            raise ImportError(
                f"无法加载 ddddocr: {e}\n请执行: pip install ddddocr"
            ) from e
        return _ocr


def recognize(data: Union[bytes, str]) -> str:
    """
    Recognize captcha text from image bytes or file path.
    Returns stripped prediction string (may be empty on failure).
    """
    ocr = get_ocr()
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("recognize 需要 bytes 或文件路径")
    result = ocr.classification(bytes(data))
    return (result or "").strip()


def is_available() -> bool:
    try:
        get_ocr()
        return True
    except Exception:
        return False
