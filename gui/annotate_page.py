import hashlib
import os
import re
import shutil
import uuid

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QSplitter, QAbstractItemView, QSizePolicy,
    QScrollArea, QFrame,
)

from utils.project_manager import ProjectManager

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_INVALID = re.compile(r'[\\/:*?"<>|\s]')

# 预览区上限，避免宽验证码把整个窗口撑开
PREVIEW_MAX_W = 640
PREVIEW_MAX_H = 400


def parse_label_from_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in stem:
        return ""
    parts = stem.split("_")
    if len(parts) < 2:
        return ""
    maybe_hash = parts[-1]
    if len(maybe_hash) >= 6 and re.fullmatch(r"[0-9a-fA-F]+", maybe_hash):
        return "_".join(parts[:-1])
    return ""


def sanitize_label(label: str) -> str:
    label = str(label).replace(" ", "")
    if not label or _INVALID.search(label):
        return ""
    return label


def list_images(directory: str):
    if not directory or not os.path.isdir(directory):
        return []
    files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(path)
    return files


class OcrWorker(QThread):
    finished_ok = pyqtSignal(str, str)
    failed = pyqtSignal(str, str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            from gui.ocr_helper import recognize
            text = recognize(self.path)
            self.finished_ok.emit(self.path, text)
        except Exception as e:
            self.failed.emit(self.path, str(e))


class AnnotatePage(QWidget):
    """
    待标注队列 (inbox) → 保存后进入完成队列 (datasets)。
    ddddocr 仅手动预识别，避免覆盖已有标注。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self.inbox_dir = ""
        self.done_dir = ""
        self.pending = []
        self.done = []
        self.index = 0
        self.queue_mode = "pending"
        self.ocr_worker = None
        self._build_ui()
        self.refresh_projects()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QGroupBox("标注设置")
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
        form.addRow("目标项目:", proj_row)

        self.inbox_label = QLabel("-")
        self.inbox_label.setWordWrap(True)
        self.inbox_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("待标注目录:", self.inbox_label)

        self.done_path_label = QLabel("-")
        self.done_path_label.setWordWrap(True)
        self.done_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("完成队列目录:", self.done_path_label)

        dir_row = QHBoxLayout()
        browse_inbox = QPushButton("导入外部目录到待标注")
        browse_inbox.clicked.connect(self.import_external_dir)
        dir_row.addWidget(browse_inbox)
        reload_btn = QPushButton("重新扫描队列")
        reload_btn.clicked.connect(self.reload_queues)
        dir_row.addWidget(reload_btn)
        dir_row.addStretch()
        form.addRow("", dir_row)
        layout.addWidget(top)

        self.stats_label = QLabel("待标注 0 | 已完成 0")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stats_label)

        mode_row = QHBoxLayout()
        self.pending_mode_btn = QPushButton("待标注队列")
        self.pending_mode_btn.setCheckable(True)
        self.pending_mode_btn.setChecked(True)
        self.pending_mode_btn.clicked.connect(lambda: self.set_queue_mode("pending"))
        mode_row.addWidget(self.pending_mode_btn)
        self.done_mode_btn = QPushButton("完成队列")
        self.done_mode_btn.setCheckable(True)
        self.done_mode_btn.clicked.connect(lambda: self.set_queue_mode("done"))
        mode_row.addWidget(self.done_mode_btn)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_title = QLabel("待标注")
        left_layout.addWidget(self.queue_title)
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.queue_list.currentRowChanged.connect(self._on_list_row_changed)
        self.queue_list.setMaximumWidth(320)
        left_layout.addWidget(self.queue_list)
        left.setMaximumWidth(340)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_label = QLabel("0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.progress_label)

        preview_frame = QFrame()
        preview_frame.setStyleSheet(
            "QFrame { background: #1e1e1e; border: 1px solid #555; }"
        )
        preview_frame.setMinimumHeight(PREVIEW_MAX_H + 16)
        preview_frame.setMaximumHeight(PREVIEW_MAX_H + 48)
        preview_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")
        self.image_scroll.setMaximumHeight(PREVIEW_MAX_H + 24)

        self.image_label = QLabel("请选择项目")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: #1e1e1e; color: #bbb; border: none; }")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.image_label.setMaximumSize(PREVIEW_MAX_W, PREVIEW_MAX_H)
        self.image_scroll.setWidget(self.image_label)
        preview_layout.addWidget(self.image_scroll)
        right_layout.addWidget(preview_frame)

        self.file_label = QLabel("")
        self.file_label.setWordWrap(True)
        self.file_label.setMaximumWidth(720)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.file_label)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("标签:"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("可手填；需要时点右侧按钮预识别，不会自动覆盖")
        self.label_edit.returnPressed.connect(self.save_to_done)
        label_row.addWidget(self.label_edit)
        self.ocr_btn = QPushButton("ddddocr 预识别")
        self.ocr_btn.setToolTip("手动调用 ddddocr 填入标签（Ctrl+R）。不会自动执行，避免覆盖已有标注。")
        self.ocr_btn.clicked.connect(self.recognize_current)
        label_row.addWidget(self.ocr_btn)
        right_layout.addLayout(label_row)

        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        btn_row.addWidget(self.prev_btn)
        self.next_btn = QPushButton("下一张 / 跳过")
        self.next_btn.clicked.connect(self.next_image)
        btn_row.addWidget(self.next_btn)
        self.save_btn = QPushButton("确认标注 → 完成队列 (Enter)")
        self.save_btn.clicked.connect(self.save_to_done)
        btn_row.addWidget(self.save_btn)
        self.reopen_btn = QPushButton("移回待标注")
        self.reopen_btn.clicked.connect(self.move_back_to_pending)
        self.reopen_btn.setEnabled(False)
        btn_row.addWidget(self.reopen_btn)
        self.delete_btn = QPushButton("删除当前")
        self.delete_btn.clicked.connect(self.delete_current)
        btn_row.addWidget(self.delete_btn)
        right_layout.addLayout(btn_row)

        tip = QLabel(
            "说明：未标完可随时关闭。待标注在 inbox/，确认后进入 datasets/ 完成队列。"
            "ddddocr 仅手动预识别（按钮 / Ctrl+R），切换图片不会自动覆盖已有标签。"
        )
        tip.setWordWrap(True)
        right_layout.addWidget(tip)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([240, 720])
        layout.addWidget(splitter, stretch=1)

        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.recognize_current)

    def _fit_preview(self, pixmap: QPixmap) -> QPixmap:
        """Scale into fixed box; never larger than PREVIEW_MAX_*."""
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(
            QSize(PREVIEW_MAX_W, PREVIEW_MAX_H),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

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
        if not name:
            return
        self.inbox_dir = self.pm.ensure_inbox_dir(name)
        self.done_dir = self.pm.ensure_datasets_dir(name)
        self.inbox_label.setText(self.inbox_dir)
        self.done_path_label.setText(self.done_dir)
        self.set_queue_mode("pending", reload=True)

    def import_external_dir(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        start = self.inbox_dir or self.pm.base_path
        path = QFileDialog.getExistingDirectory(self, "选择要导入的图片目录", start)
        if not path:
            return
        inbox = self.pm.ensure_inbox_dir(project)
        copied = 0
        for src in list_images(path):
            name = os.path.basename(src)
            dst = os.path.join(inbox, name)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(name)
                dst = os.path.join(inbox, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError:
                continue
        self.reload_queues()
        self.set_queue_mode("pending")
        QMessageBox.information(self, "导入完成", f"已复制 {copied} 张到待标注队列:\n{inbox}")

    def reload_queues(self):
        self.pending = list_images(self.inbox_dir)
        self.done = list_images(self.done_dir)
        self._update_stats()
        self._refresh_list()
        if self.index >= len(self._active_files()):
            self.index = max(0, len(self._active_files()) - 1)
        self.show_current()

    def _update_stats(self):
        self.stats_label.setText(f"待标注 {len(self.pending)} | 已完成 {len(self.done)}")

    def _active_files(self):
        return self.pending if self.queue_mode == "pending" else self.done

    def set_queue_mode(self, mode: str, reload: bool = False):
        self.queue_mode = mode
        self.pending_mode_btn.setChecked(mode == "pending")
        self.done_mode_btn.setChecked(mode == "done")
        self.queue_title.setText("待标注队列" if mode == "pending" else "完成队列")
        self.reopen_btn.setEnabled(mode == "done")
        self.ocr_btn.setEnabled(mode == "pending")
        self.save_btn.setText(
            "确认标注 → 完成队列 (Enter)" if mode == "pending" else "更新标签并保存 (Enter)"
        )
        if reload:
            self.reload_queues()
        else:
            self.index = 0
            self._refresh_list()
            self.show_current()

    def _refresh_list(self):
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        files = self._active_files()
        for path in files:
            label = parse_label_from_name(path)
            text = os.path.basename(path)
            if label:
                text = f"[{label}] {text}"
            self.queue_list.addItem(QListWidgetItem(text))
        if files and 0 <= self.index < len(files):
            self.queue_list.setCurrentRow(self.index)
        self.queue_list.blockSignals(False)

    def _on_list_row_changed(self, row: int):
        if row < 0:
            return
        files = self._active_files()
        if row >= len(files):
            return
        if row == self.index:
            return
        self.index = row
        self.show_current()

    def current_path(self):
        files = self._active_files()
        if not files or self.index < 0 or self.index >= len(files):
            return None
        return files[self.index]

    def show_current(self):
        path = self.current_path()
        files = self._active_files()
        total = len(files)
        self.progress_label.setText(f"{(self.index + 1) if path else 0} / {total}")
        self._update_stats()

        if not path:
            empty = "待标注队列为空（可导入图片或稍后再标）" if self.queue_mode == "pending" else "完成队列为空"
            self.image_label.setText(empty)
            self.image_label.setPixmap(QPixmap())
            self.image_label.adjustSize()
            self.file_label.setText("")
            self.label_edit.clear()
            self.status_label.setText("")
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载: {os.path.basename(path)}")
            self.image_label.setPixmap(QPixmap())
            self.image_label.adjustSize()
        else:
            scaled = self._fit_preview(pixmap)
            self.image_label.setText("")
            self.image_label.setPixmap(scaled)
            self.image_label.resize(scaled.size())
            # 滚动区视口不随原图像素变宽
            self.image_scroll.setMaximumWidth(PREVIEW_MAX_W + 40)

        self.file_label.setText(path)
        existing = parse_label_from_name(path)
        # 只保留文件名中的已有标签，绝不自动 OCR 覆盖
        self.label_edit.setText(existing)

        self.queue_list.blockSignals(True)
        self.queue_list.setCurrentRow(self.index)
        self.queue_list.blockSignals(False)

        if self.queue_mode == "done":
            self.status_label.setText("完成队列：可改标签后保存，或移回待标注")
            return

        if existing:
            self.status_label.setText("已保留文件名标签；需要时再点「ddddocr 预识别」")
        else:
            self.status_label.setText("无已有标签，可手填或点「ddddocr 预识别」")

    def recognize_current(self):
        if self.queue_mode != "pending":
            QMessageBox.information(self, "提示", "完成队列请直接改标签；预识别仅用于待标注。")
            return
        path = self.current_path()
        if not path:
            return
        if self.ocr_worker and self.ocr_worker.isRunning():
            return

        current_text = self.label_edit.text().strip()
        if current_text:
            reply = QMessageBox.question(
                self, "确认预识别",
                f"当前标签为「{current_text}」，预识别将覆盖输入框内容。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.status_label.setText("ddddocr 识别中…")
        self.ocr_btn.setEnabled(False)
        self.ocr_worker = OcrWorker(path)
        self.ocr_worker.finished_ok.connect(self._on_ocr_ok)
        self.ocr_worker.failed.connect(self._on_ocr_fail)
        self.ocr_worker.finished.connect(lambda: self.ocr_btn.setEnabled(self.queue_mode == "pending"))
        self.ocr_worker.start()

    def _on_ocr_ok(self, path: str, text: str):
        if path != self.current_path():
            return
        self.label_edit.setText(text)
        self.label_edit.selectAll()
        self.label_edit.setFocus()
        self.status_label.setText(f"预识别结果: {text or '(空)'}")

    def _on_ocr_fail(self, path: str, error: str):
        if path != self.current_path():
            return
        self.status_label.setText(f"预识别失败: {error}")
        QMessageBox.warning(self, "识别失败", error)

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def next_image(self):
        files = self._active_files()
        if self.index < len(files) - 1:
            self.index += 1
            self.show_current()
        else:
            self.status_label.setText("已经是当前队列最后一张")

    def save_to_done(self):
        path = self.current_path()
        if not path:
            return
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return

        label = sanitize_label(self.label_edit.text())
        if not label:
            QMessageBox.warning(self, "提示", "标签为空或包含非法字符/空格")
            return

        out_dir = self.pm.ensure_datasets_dir(project)
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        if ext == "jpeg":
            ext = "jpg"
        digest = hashlib.md5(uuid.uuid4().bytes).hexdigest()[:8]
        new_name = f"{label}_{digest}.{ext}"
        new_path = os.path.join(out_dir, new_name)

        try:
            if os.path.abspath(path) == os.path.abspath(new_path):
                pass
            elif self.queue_mode == "done" and os.path.dirname(os.path.abspath(path)) == os.path.abspath(out_dir):
                os.rename(path, new_path)
                self.done[self.index] = new_path
                self.status_label.setText(f"已更新: {new_name}")
                self._refresh_list()
                self.show_current()
                return
            else:
                shutil.move(path, new_path)
        except OSError as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return

        if self.queue_mode == "pending":
            self.pending.pop(self.index)
            self.done.append(new_path)
            self._update_stats()
            self.status_label.setText(f"已进入完成队列: {new_name}")
            if self.index >= len(self.pending):
                self.index = max(0, len(self.pending) - 1)
            self._refresh_list()
            self.show_current()
            if not self.pending:
                self.status_label.setText(f"已进入完成队列: {new_name}。待标注已空，可稍后继续导入。")
        else:
            self.reload_queues()

    def move_back_to_pending(self):
        if self.queue_mode != "done":
            return
        path = self.current_path()
        if not path:
            return
        project = self.project_combo.currentText().strip()
        if not project:
            return
        inbox = self.pm.ensure_inbox_dir(project)
        name = os.path.basename(path)
        dst = os.path.join(inbox, name)
        if os.path.exists(dst):
            stem, ext = os.path.splitext(name)
            dst = os.path.join(inbox, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
        try:
            shutil.move(path, dst)
        except OSError as e:
            QMessageBox.critical(self, "移回失败", str(e))
            return
        self.done.pop(self.index)
        self.pending.append(dst)
        self._update_stats()
        if self.index >= len(self.done):
            self.index = max(0, len(self.done) - 1)
        self._refresh_list()
        self.show_current()
        self.status_label.setText(f"已移回待标注: {os.path.basename(dst)}")

    def delete_current(self):
        path = self.current_path()
        if not path:
            return
        reply = QMessageBox.question(
            self, "确认删除", f"删除文件?\n{path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(path)
        except OSError as e:
            QMessageBox.critical(self, "删除失败", str(e))
            return
        files = self._active_files()
        files.pop(self.index)
        if self.index >= len(files) and self.index > 0:
            self.index -= 1
        self._update_stats()
        self._refresh_list()
        self.show_current()
