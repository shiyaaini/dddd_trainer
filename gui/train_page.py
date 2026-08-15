import os
import sys
import time

from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTextEdit, QGroupBox, QFormLayout, QMessageBox,
)

from utils.project_manager import ProjectManager


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self.process = None
        self._mode = None  # "cache" | "train"
        self._start_ts = None
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.python = sys.executable
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._build_ui()
        self.refresh_projects()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QGroupBox("训练设置")
        form = QFormLayout(top)

        proj_row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._update_paths)
        proj_row.addWidget(self.project_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_projects)
        proj_row.addWidget(refresh_btn)
        proj_row.addStretch()
        form.addRow("目标项目:", proj_row)

        self.datasets_label = QLabel("-")
        self.datasets_label.setWordWrap(True)
        self.datasets_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("数据集:", self.datasets_label)

        self.status_label = QLabel("空闲")
        form.addRow("状态:", self.status_label)

        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; font-family: Consolas, 'Courier New', monospace;"
        )
        form.addRow("已用时长:", self.elapsed_label)

        layout.addWidget(top)

        btn_row = QHBoxLayout()
        self.cache_btn = QPushButton("缓存数据")
        self.cache_btn.clicked.connect(self.start_cache)
        btn_row.addWidget(self.cache_btn)
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_train)
        btn_row.addWidget(self.train_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_process)
        btn_row.addWidget(self.stop_btn)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        btn_row.addWidget(self.clear_btn)
        self.reset_btn = QPushButton("重置项目")
        self.reset_btn.setToolTip("清空 cache / checkpoints / models，保留 datasets 与 inbox 图片")
        self.reset_btn.clicked.connect(self.reset_project)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        tip = QLabel(
            "建议流程：标注/生成样本 → 缓存数据 → 开始训练。\n"
            "训练需要已安装 PyTorch（含 CUDA 版本请按 README）。停止会终止子进程。\n"
            "「已用时长」在缓存/训练过程中每秒刷新，结束后保留本次总耗时。\n"
            "「重置项目」会清空缓存、断点与导出模型，不删除 datasets/inbox 图片。\n\n"
            "日志参数说明：\n"
            "· Epoch：第几轮完整扫过训练集\n"
            "· Step：累计训练步数（每处理一批数据 +1）\n"
            "· LastLoss：最近一批的损失，越小越好\n"
            "· AvgLoss：到目前为止的平均损失，整体应缓慢下降\n"
            "· Lr：学习率，控制参数更新幅度\n"
            "· Acc：验证准确率（仅部分 Step 打印）；OCR 通常要求整串全对才算对，早期为 0 很常见"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, stretch=1)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _tick_elapsed(self):
        if self._start_ts is None:
            return
        self.elapsed_label.setText(self._format_elapsed(time.time() - self._start_ts))

    def _start_elapsed(self):
        self._start_ts = time.time()
        self.elapsed_label.setText("00:00:00")
        self._elapsed_timer.start()

    def _stop_elapsed(self) -> str:
        self._elapsed_timer.stop()
        if self._start_ts is None:
            text = self.elapsed_label.text() or "00:00:00"
        else:
            text = self._format_elapsed(time.time() - self._start_ts)
            self.elapsed_label.setText(text)
        return text

    def refresh_projects(self):
        current = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if current and current in projects:
            self.project_combo.setCurrentText(current)
        self.project_combo.blockSignals(False)
        self._update_paths(self.project_combo.currentText())

    def _update_paths(self, name: str):
        if not name:
            self.datasets_label.setText("-")
            return
        self.datasets_label.setText(self.pm.get_datasets_path(name))

    def _append_log(self, text: str):
        if not text:
            return
        self.log_edit.moveCursor(self.log_edit.textCursor().MoveOperation.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(self.log_edit.textCursor().MoveOperation.End)

    def _busy(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def _set_running(self, running: bool, mode: str = None):
        self.cache_btn.setEnabled(not running)
        self.train_btn.setEnabled(not running)
        self.reset_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self._mode = mode
            self.status_label.setText("缓存中…" if mode == "cache" else "训练中…")
            self._start_elapsed()
        else:
            self._mode = None
            self.status_label.setText("空闲")

    def reset_project(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        if self._busy():
            QMessageBox.warning(self, "提示", "请先停止当前任务再重置")
            return

        reply = QMessageBox.question(
            self,
            "确认重置项目",
            (
                f"将重置项目「{project}」的训练状态：\n\n"
                f"会删除：\n"
                f"  · cache（训练/验证缓存）\n"
                f"  · checkpoints（断点权重）\n"
                f"  · models（onnx / charsets.json）\n\n"
                f"会保留：\n"
                f"  · datasets 图片\n"
                f"  · inbox 图片\n"
                f"  · config.yaml 配置\n\n"
                f"重置后需重新「缓存数据」再训练。是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        ok, message, deleted = self.pm.reset_training(project)
        self._append_log(f"\n[reset] {message}\n")
        if ok:
            self.elapsed_label.setText("00:00:00")
            self._start_ts = None
            QMessageBox.information(self, "重置完成", message)
        else:
            QMessageBox.warning(self, "重置未完全成功", message)

    def _start_process(self, args, mode: str):
        if self._busy():
            QMessageBox.warning(self, "提示", "已有任务在运行")
            return

        self.process = QProcess(self)
        self.process.setWorkingDirectory(self.root)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)

        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

        self._append_log(f"\n$ {' '.join(args)}\n")
        self._set_running(True, mode)
        self.process.start(self.python, args)

    def _on_ready_read(self):
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(data)

    def _on_finished(self, exit_code: int, _status):
        mode = self._mode or "任务"
        elapsed = self._stop_elapsed()
        self._append_log(f"\n[{mode}] 结束，exit={exit_code}，耗时 {elapsed}\n")
        self._set_running(False)
        self.process = None

    def _on_error(self, error):
        self._append_log(f"\n进程错误: {error}\n")

    def stop_process(self):
        if not self.process or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self._append_log("\n正在停止进程…\n")
        self.process.kill()

    def start_cache(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        datasets = self.pm.ensure_datasets_dir(project)
        if not any(
            f.lower().endswith(ext)
            for f in os.listdir(datasets)
            for ext in (".jpg", ".jpeg", ".png", ".bmp")
        ):
            QMessageBox.warning(self, "提示", f"数据集目录没有图片:\n{datasets}")
            return
        args = ["app.py", "cache", project, datasets]
        self._start_process(args, "cache")

    def start_train(self):
        project = self.project_combo.currentText().strip()
        if not project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "缺少 PyTorch",
                "未检测到 torch。\n请按 README 安装对应 CUDA 版本的 PyTorch 后再训练。",
            )
            return

        cache_dir = os.path.join(self.pm.get_project_path(project), "cache")
        train_cache = os.path.join(cache_dir, "cache.train.tmp")
        if not os.path.isfile(train_cache):
            reply = QMessageBox.question(
                self, "尚未缓存",
                "未找到 cache.train.tmp，是否先缓存再训练？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.start_cache()
                return
            return

        args = ["app.py", "train", project]
        self._start_process(args, "train")
