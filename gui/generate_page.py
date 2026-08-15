import io

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QProgressBar, QTextEdit, QGroupBox,
    QMessageBox, QSplitter, QFormLayout,
)

from gui.code_runner import DEFAULT_TEMPLATE, GenerateWorker, preview_one
from gui.widgets.code_editor import CodeEditor
from utils.project_manager import ProjectManager


def pil_to_qpixmap(image):
    buf = io.BytesIO()
    img = image
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    fmt = "PNG"
    img.save(buf, format=fmt)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    return pixmap


class GeneratePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self.worker = None
        self._build_ui()
        self.refresh_projects()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        help_box = QGroupBox("使用说明（生成路径 / 命名 / 代码约定）")
        help_layout = QVBoxLayout(help_box)
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        self.help_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.help_label.setStyleSheet("QLabel { color: #ffffff; line-height: 1.4; }")
        help_layout.addWidget(self.help_label)
        layout.addWidget(help_box)

        top = QGroupBox("生成设置")
        form = QFormLayout(top)

        proj_row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._update_output_path)
        proj_row.addWidget(self.project_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_projects)
        proj_row.addWidget(refresh_btn)
        proj_row.addStretch()
        form.addRow("目标项目:", proj_row)

        self.output_label = QLabel("-")
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("输出目录 (datasets):", self.output_label)

        self.naming_label = QLabel("文件命名: {label}_{8位hash}.{格式}  （与手动标注完成队列相同）")
        self.naming_label.setWordWrap(True)
        form.addRow("命名规则:", self.naming_label)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1_000_000)
        self.count_spin.setValue(100)
        form.addRow("生成数量:", self.count_spin)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "bmp"])
        self.format_combo.currentTextChanged.connect(
            lambda _t: self._update_output_path(self.project_combo.currentText())
        )
        form.addRow("图片格式:", self.format_combo)

        layout.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel(
            "生成代码：必须定义 generate(index) -> (Image, label)；详见上方说明与模板注释"
        ))
        self.editor = CodeEditor()
        self.editor.setPlainText(DEFAULT_TEMPLATE)
        left_layout.addWidget(self.editor)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("预览"))
        self.preview_label = QLabel("尚未预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(200, 120)
        self.preview_label.setStyleSheet("QLabel { background: #f0f0f0; border: 1px solid #ccc; }")
        right_layout.addWidget(self.preview_label)
        self.preview_info = QLabel("")
        self.preview_info.setWordWrap(True)
        right_layout.addWidget(self.preview_info)
        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("预览一张")
        self.preview_btn.clicked.connect(self.preview_sample)
        btn_row.addWidget(self.preview_btn)
        self.start_btn = QPushButton("开始生成")
        self.start_btn.clicked.connect(self.start_generate)
        btn_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generate)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(140)
        layout.addWidget(self.log_edit)

    def refresh_projects(self):
        current = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if current and current in projects:
            self.project_combo.setCurrentText(current)
        self.project_combo.blockSignals(False)
        self._update_output_path(self.project_combo.currentText())

    def _update_output_path(self, name: str):
        if not name:
            self.output_label.setText("-")
            self.help_label.setText(self._help_text(None, None))
            return
        path = self.pm.get_datasets_path(name)
        rel = f"projects/{name}/datasets"
        self.output_label.setText(f"{path}\n相对路径: {rel}")
        self.help_label.setText(self._help_text(name, path))

    def _help_text(self, project: str | None, abs_datasets: str | None) -> str:
        if not project:
            return (
                "请先选择项目。生成图片会写入该项目的 datasets 目录（完成队列），"
                "不是 inbox（待标注）。\n"
                "代码约定: def generate(index: int) -> (PIL.Image, label: str)\n"
                "保存名: {label}_{8位md5}.{png|jpg|bmp}，可直接 python app.py cache 项目名 datasets路径"
            )
        inbox = self.pm.get_inbox_path(project)
        return (
            f"【项目】{project}\n"
            f"【生成输出 / 完成队列】{abs_datasets}\n"
            f"【相对路径】projects/{project}/datasets/\n"
            f"【待标注 inbox】{inbox}  ← 本页不会写入此处；未标注图请放到 inbox 用「手动标注」\n"
            f"【文件名】{{label}}_{{8位hash}}.{self.format_combo.currentText() if hasattr(self, 'format_combo') else 'png'}\n"
            "【代码】必须实现 generate(index)->(Image, label)；预注入 Image/ImageDraw/ImageFont/random/math/os/hashlib\n"
            f"【后续】缓存: python app.py cache {project} projects/{project}/datasets\n"
            f"【后续】训练: python app.py train {project}  （或使用「训练」页）"
        )

    def _append_log(self, text: str):
        self.log_edit.append(text)

    def preview_sample(self):
        if not self.project_combo.currentText():
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        try:
            image, label = preview_one(self.editor.toPlainText(), 0)
            self._show_preview(image, label)
            self._append_log(f"预览成功, label={label}")
        except Exception as e:
            QMessageBox.critical(self, "预览失败", str(e))
            self._append_log(f"预览失败: {e}")

    def _show_preview(self, image, label: str):
        pixmap = pil_to_qpixmap(image)
        scaled = pixmap.scaled(
            self.preview_label.width() or 280,
            self.preview_label.height() or 160,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_info.setText(f"label: {label}\nsize: {image.size} mode: {image.mode}")

    def start_generate(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "正在生成中")
            return

        output_dir = self.pm.ensure_datasets_dir(project)
        count = self.count_spin.value()
        ext = self.format_combo.currentText()
        source = self.editor.toPlainText()

        self.progress.setValue(0)
        self.progress.setMaximum(count)
        self._append_log(f"开始生成 → {output_dir}，数量={count}，格式={ext}")

        self.worker = GenerateWorker(
            source=source,
            output_dir=output_dir,
            count=count,
            ext=ext,
            preview_first=True,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.preview.connect(self._show_preview)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop_generate(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._append_log("正在停止...")

    def _on_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_finished(self, success: int, failed: int):
        project = self.project_combo.currentText().strip()
        self._append_log(f"完成: 成功 {success}，失败 {failed}")
        if project:
            rel = f"projects/{project}/datasets"
            self._append_log(f"缓存命令: python app.py cache {project} {rel}")
        self._reset_buttons()

    def _on_failed(self, message: str):
        self._append_log(message)
        QMessageBox.critical(self, "生成失败", message)
        self._reset_buttons()

    def _reset_buttons(self):
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
