import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QDialog, QApplication, QTabWidget,
    QSizePolicy,
)

from gui.model_tester import (
    predict_image, parse_expected_from_name, compare_result, clear_ocr_cache,
)
from utils.project_manager import ProjectManager

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 预览区上限，避免宽图/高图把窗口撑开导致宽度抖动
PREVIEW_MAX_W = 480
PREVIEW_MAX_H = 200


def _norm_path(path: str) -> str:
    if not path:
        return ""
    return os.path.abspath(path).replace("\\", "/")


def build_python_examples(onnx_path: str, charsets_path: str, image_path: str = "test.png") -> dict:
    """Generate copy-ready Python snippets for the selected model."""
    onnx = _norm_path(onnx_path) or "path/to/model.onnx"
    charset = _norm_path(charsets_path) or "path/to/charsets.json"
    image = _norm_path(image_path) if image_path and os.path.isfile(image_path) else "test.png"

    ddddocr_code = f'''# pip install ddddocr
import ddddocr

ocr = ddddocr.DdddOcr(
    det=False,
    ocr=False,
    show_ad=False,
    import_onnx_path=r"{onnx}",
    charsets_path=r"{charset}",
)

with open(r"{image}", "rb") as f:
    image_bytes = f.read()

result = ocr.classification(image_bytes)
print(result)
'''

    onnxruntime_code = f'''# pip install onnxruntime pillow numpy
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

ONNX_PATH = r"{onnx}"
CHARSETS_PATH = r"{charset}"
IMAGE_PATH = r"{image}"

with open(CHARSETS_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

charset = meta["charset"]
width_cfg, height_cfg = int(meta["image"][0]), int(meta["image"][1])
channel = int(meta["channel"])

mode = "L" if channel == 1 else "RGB"
image = Image.open(IMAGE_PATH).convert(mode)
iw, ih = image.size
if width_cfg == -1:
    image = image.resize((int(iw * (height_cfg / ih)), height_cfg))
else:
    image = image.resize((width_cfg, height_cfg))

arr = np.asarray(image).astype(np.float32) / 255.0
if channel == 1:
    arr = (arr - 0.456) / 0.224
    arr = arr[None, None, :, :]
else:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = arr.transpose(2, 0, 1)
    arr = (arr - mean) / std
    arr = arr[None, :, :, :]

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
output = sess.run(None, {{sess.get_inputs()[0].name: arr}})[0]

# CTC decode: collapse repeats, drop blank(0)
if output.ndim == 3:
    indices = np.argmax(output[:, 0, :], axis=1) if output.shape[1] == 1 else np.argmax(output[0], axis=1)
elif output.ndim == 2:
    indices = output[0]
else:
    indices = output

decoded, prev = [], None
for idx in indices:
    idx = int(idx)
    if idx != prev and idx != 0 and 0 <= idx < len(charset):
        decoded.append(str(charset[idx]))
    prev = idx
print("".join(decoded))
'''

    project_helper_code = f'''# 在本仓库根目录下运行（复用 GUI 同款推理）
# pip install onnxruntime pillow numpy
from gui.model_tester import predict_image

text = predict_image(
    r"{onnx}",
    r"{charset}",
    r"{image}",
)
print(text)
'''

    return {
        "ddddocr": ddddocr_code.strip() + "\n",
        "onnxruntime": onnxruntime_code.strip() + "\n",
        "project": project_helper_code.strip() + "\n",
    }


class PythonExampleDialog(QDialog):
    def __init__(self, onnx_path: str, charsets_path: str, image_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Python 调用训练模型示例")
        self.resize(720, 560)
        self.setMinimumSize(560, 420)

        layout = QVBoxLayout(self)
        tip = QLabel(
            "下方代码已填入当前选中的 onnx / charsets 路径，可直接复制运行。"
            "推荐优先使用 ddddocr 方式部署。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        path_row = QFormLayout()
        self.onnx_label = QLabel(_norm_path(onnx_path) or "（未选择）")
        self.onnx_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.onnx_label.setWordWrap(True)
        path_row.addRow("ONNX:", self.onnx_label)
        self.charset_label = QLabel(_norm_path(charsets_path) or "（未选择）")
        self.charset_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.charset_label.setWordWrap(True)
        path_row.addRow("字符集:", self.charset_label)
        layout.addLayout(path_row)

        examples = build_python_examples(onnx_path, charsets_path, image_path)
        self.tabs = QTabWidget()
        self._editors = {}
        for key, title in (
            ("ddddocr", "ddddocr（推荐）"),
            ("onnxruntime", "onnxruntime 原生"),
            ("project", "本项目 helper"),
        ):
            editor = QTextEdit()
            editor.setReadOnly(True)
            editor.setFont(QFont("Consolas", 10))
            editor.setPlainText(examples[key])
            self._editors[key] = editor
            self.tabs.addTab(editor, title)
        layout.addWidget(self.tabs, stretch=1)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("复制当前示例")
        copy_btn.clicked.connect(self._copy_current)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _copy_current(self):
        editor = self.tabs.currentWidget()
        if not isinstance(editor, QTextEdit):
            return
        QApplication.clipboard().setText(editor.toPlainText())
        QMessageBox.information(self, "已复制", "当前示例代码已复制到剪贴板")


class DropZone(QLabel):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)
        self.setStyleSheet(
            "QLabel { background: #1e1e1e; border: 2px dashed #666; color: #bbb; }"
        )
        self.setText("将图片拖到此处测试\n或点击下方「选择图片」")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "QLabel { background: #1a3050; border: 2px dashed #4a90d9; color: #ddd; }"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            "QLabel { background: #1e1e1e; border: 2px dashed #666; color: #bbb; }"
        )

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(
            "QLabel { background: #1e1e1e; border: 2px dashed #666; color: #bbb; }"
        )
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    full = os.path.join(path, name)
                    if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                        paths.append(full)
            elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS:
                paths.append(path)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class PredictWorker(QThread):
    progress = pyqtSignal(int, int)
    one_done = pyqtSignal(str, str, str, str)  # path, pred, expected, status
    finished_ok = pyqtSignal(int, int)  # correct, total_with_expected
    failed = pyqtSignal(str)

    def __init__(self, onnx_path, charsets_path, paths, parent=None):
        super().__init__(parent)
        self.onnx_path = onnx_path
        self.charsets_path = charsets_path
        self.paths = paths

    def run(self):
        correct = 0
        with_expected = 0
        try:
            total = len(self.paths)
            for i, path in enumerate(self.paths):
                pred = predict_image(self.onnx_path, self.charsets_path, path)
                expected = parse_expected_from_name(path)
                if expected:
                    with_expected += 1
                    ok, status = compare_result(pred, expected)
                    if ok:
                        correct += 1
                else:
                    status = "无文件名标签（仅显示预测）"
                self.one_done.emit(path, pred, expected, status)
                self.progress.emit(i + 1, total)
            self.finished_ok.emit(correct, with_expected)
        except Exception as e:
            self.failed.emit(str(e))


class TestPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self.worker = None
        self._onnx_map = {}
        self._last_image_path = ""
        self._build_ui()
        self.refresh_projects()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        help_box = QGroupBox("测试说明")
        help_layout = QVBoxLayout(help_box)
        help = QLabel(
            "使用训练导出的 onnx + models/charsets.json 做识别测试（自动按训练预处理归一化）。\n"
            "支持：选择图片 / 拖拽图片或文件夹。若文件名为 label_hash.ext，会自动比对期望标签并统计正确率。\n"
            "点「Python 调用示例」可查看 ddddocr / onnxruntime 调用代码；表格列宽可拖动表头分隔线调整。"
        )
        help.setWordWrap(True)
        help_layout.addWidget(help)
        layout.addWidget(help_box)

        top = QGroupBox("模型选择")
        form = QFormLayout(top)

        proj_row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        proj_row.addWidget(self.project_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_projects)
        proj_row.addWidget(refresh_btn)
        proj_row.addStretch()
        form.addRow("项目:", proj_row)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(320)
        model_row.addWidget(self.model_combo)
        browse_model = QPushButton("浏览 onnx…")
        browse_model.clicked.connect(self.browse_onnx)
        model_row.addWidget(browse_model)
        example_btn = QPushButton("Python 调用示例…")
        example_btn.setToolTip("查看如何用 Python 加载当前训练好的 onnx 模型")
        example_btn.clicked.connect(self.show_python_example)
        model_row.addWidget(example_btn)
        form.addRow("模型:", model_row)

        self.charset_label = QLabel("-")
        self.charset_label.setWordWrap(True)
        self.charset_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("字符集:", self.charset_label)
        layout.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.run_on_files)
        left_layout.addWidget(self.drop_zone)

        self.preview_label = QLabel("预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedHeight(PREVIEW_MAX_H)
        self.preview_label.setMaximumWidth(PREVIEW_MAX_W)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.preview_label.setStyleSheet(
            "QLabel { background: #1e1e1e; border: 1px solid #555; color: #bbb; }"
        )
        left_layout.addWidget(self.preview_label)

        self.result_label = QLabel("预测: -")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        left_layout.addWidget(self.result_label)

        self.expect_edit = QLineEdit()
        self.expect_edit.setPlaceholderText("可选：手动填写期望标签，用于单张核对")
        left_layout.addWidget(self.expect_edit)

        btn_row = QHBoxLayout()
        pick_btn = QPushButton("选择图片…")
        pick_btn.clicked.connect(self.pick_images)
        btn_row.addWidget(pick_btn)
        folder_btn = QPushButton("选择文件夹批量测…")
        folder_btn.clicked.connect(self.pick_folder)
        btn_row.addWidget(folder_btn)
        clear_btn = QPushButton("清空结果")
        clear_btn.clicked.connect(self.clear_results)
        btn_row.addWidget(clear_btn)
        left_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)
        self.summary_label = QLabel("")
        left_layout.addWidget(self.summary_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("批量结果"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件", "预测", "期望", "结果"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(48)
        self.table.setColumnWidth(0, 160)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 160)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setStyleSheet(
            "QTableWidget { background: #1e1e1e; gridline-color: #444; color: #ddd; }"
            "QHeaderView::section { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 4px; }"
        )
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        right_layout.addWidget(self.table)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        right_layout.addWidget(self.log_edit)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def refresh_projects(self):
        current = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if current and current in projects:
            self.project_combo.setCurrentText(current)
        self.project_combo.blockSignals(False)
        self._on_project_changed(self.project_combo.currentText())

    def _on_project_changed(self, name: str):
        self.model_combo.clear()
        self._onnx_map = {}
        if not name:
            self.charset_label.setText("-")
            return
        charset = self.pm.get_charsets_path(name)
        self.charset_label.setText(charset if os.path.isfile(charset) else f"{charset} （尚未导出，请先完成训练）")
        for path in self.pm.list_onnx_models(name):
            display = os.path.basename(path)
            self._onnx_map[display] = path
            self.model_combo.addItem(display)
        if self.model_combo.count() == 0:
            self.model_combo.addItem("（无 onnx，请先训练导出）")

    def browse_onnx(self):
        start = self.pm.base_path
        path, _ = QFileDialog.getOpenFileName(self, "选择 onnx 模型", start, "ONNX (*.onnx)")
        if not path:
            return
        display = os.path.basename(path)
        self._onnx_map[display] = path
        if self.model_combo.findText(display) < 0:
            self.model_combo.addItem(display)
        self.model_combo.setCurrentText(display)
        # try sibling charsets.json
        sibling = os.path.join(os.path.dirname(path), "charsets.json")
        if os.path.isfile(sibling):
            self.charset_label.setText(sibling)

    def show_python_example(self):
        onnx = self._current_onnx()
        charset = self._current_charset()
        if not onnx or not os.path.isfile(onnx):
            QMessageBox.warning(
                self, "提示",
                "请先选择有效的 onnx 模型后再查看调用示例。\n"
                "训练完成后模型位于 projects/项目名/models/",
            )
            return
        if not charset or not os.path.isfile(charset):
            QMessageBox.warning(self, "提示", "缺少 charsets.json，示例路径可能不完整")
        image_path = self._last_image_path if self._last_image_path and os.path.isfile(self._last_image_path) else ""
        dlg = PythonExampleDialog(onnx, charset or "", image_path, parent=self)
        dlg.exec()

    def _current_onnx(self):
        text = self.model_combo.currentText()
        return self._onnx_map.get(text)

    def _current_charset(self):
        text = self.charset_label.text().strip()
        # may contain hint after path
        path = text.split(" ")[0]
        project = self.project_combo.currentText().strip()
        if project:
            default = self.pm.get_charsets_path(project)
            if os.path.isfile(default):
                return default
        if os.path.isfile(path):
            return path
        return ""

    def pick_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片",
            self.pm.get_datasets_path(self.project_combo.currentText()) if self.project_combo.currentText() else "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if paths:
            self.run_on_files(paths)

    def pick_folder(self):
        start = ""
        if self.project_combo.currentText():
            start = self.pm.get_datasets_path(self.project_combo.currentText())
        folder = QFileDialog.getExistingDirectory(self, "选择测试文件夹", start)
        if not folder:
            return
        paths = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in IMAGE_EXTS
        ]
        if not paths:
            QMessageBox.warning(self, "提示", "文件夹中没有图片")
            return
        self.run_on_files(paths)

    def clear_results(self):
        self.table.setRowCount(0)
        self.log_edit.clear()
        self.summary_label.setText("")
        self.result_label.setText("预测: -")
        self.progress.setValue(0)
        self.preview_label.setText("预览")
        self.preview_label.setPixmap(QPixmap())

    def run_on_files(self, paths: list):
        onnx = self._current_onnx()
        charset = self._current_charset()
        if not onnx:
            QMessageBox.warning(self, "提示", "请先选择有效的 onnx 模型（训练完成后位于 projects/项目/models/）")
            return
        if not charset or not os.path.isfile(charset):
            QMessageBox.warning(self, "提示", "缺少 charsets.json，无法加载自定义模型")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "正在测试中")
            return

        clear_ocr_cache()
        self.clear_results()
        self.progress.setMaximum(len(paths))
        self.progress.setValue(0)
        self.log_edit.append(f"开始测试 {len(paths)} 张 | 模型={os.path.basename(onnx)}")

        # single-file fast path with optional manual expected
        if len(paths) == 1 and self.expect_edit.text().strip():
            try:
                pred = predict_image(onnx, charset, paths[0])
                expected = self.expect_edit.text().strip()
                ok, status = compare_result(pred, expected)
                self._show_preview(paths[0])
                self.result_label.setText(f"预测: {pred}\n期望: {expected}\n{status}")
                self._add_row(paths[0], pred, expected, status)
                self.summary_label.setText("正确 1/1" if ok else "正确 0/1")
                self.progress.setValue(1)
            except Exception as e:
                QMessageBox.critical(self, "识别失败", str(e))
            return

        self.worker = PredictWorker(onnx, charset, paths)
        self.worker.progress.connect(lambda c, t: self.progress.setValue(c))
        self.worker.one_done.connect(self._on_one_done)
        self.worker.finished_ok.connect(self._on_batch_finished)
        self.worker.failed.connect(lambda m: QMessageBox.critical(self, "识别失败", m))
        self.worker.start()

    def _on_one_done(self, path, pred, expected, status):
        self._add_row(path, pred, expected, status)
        # update preview to latest
        self._show_preview(path)
        self.result_label.setText(f"预测: {pred}" + (f"\n期望: {expected}" if expected else ""))

    def _on_batch_finished(self, correct, with_expected):
        if with_expected:
            acc = correct / with_expected * 100
            self.summary_label.setText(f"有标签样本正确率: {correct}/{with_expected} = {acc:.2f}%")
            self.log_edit.append(self.summary_label.text())
        else:
            self.summary_label.setText("完成（文件名无 label_hash 时无法自动算准确率，可看预测结果）")
            self.log_edit.append(self.summary_label.text())

    def _add_row(self, path, pred, expected, status):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
        self.table.setItem(row, 1, QTableWidgetItem(pred))
        self.table.setItem(row, 2, QTableWidgetItem(expected))
        self.table.setItem(row, 3, QTableWidgetItem(status))
        self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, path)

    def _on_row_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        path = self.table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        pred = self.table.item(items[0].row(), 1).text()
        expected = self.table.item(items[0].row(), 2).text()
        if path:
            self._show_preview(path)
            self.result_label.setText(f"预测: {pred}" + (f"\n期望: {expected}" if expected else ""))

    def _show_preview(self, path: str):
        self._last_image_path = path
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.clear()
            self.preview_label.setText("无法预览")
            return
        # 按固定上限缩放，不依赖当前控件尺寸，避免布局反馈导致窗口宽度变化
        box_w = min(PREVIEW_MAX_W, max(1, self.preview_label.width()))
        box_h = min(PREVIEW_MAX_H, max(1, self.preview_label.height() or PREVIEW_MAX_H))
        scaled = pixmap.scaled(
            QSize(box_w, box_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)
        self.drop_zone.setText(os.path.basename(path))
