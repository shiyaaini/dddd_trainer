from __future__ import annotations

import os
import sys
from typing import List, Optional

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt6.QtGui import QFont, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFormLayout, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QComboBox, QApplication,
)

from utils.env_check import (
    collect_env_report,
    build_pip_commands,
    format_commands_for_display,
    EnvReport,
)


class EnvPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.python = sys.executable
        self.process = None  # type: Optional[QProcess]
        self._report = None  # type: Optional[EnvReport]
        self._pending_cmds = []  # type: List[str]
        self._cmd_index = 0
        self._build_ui()
        self.refresh_check()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        tip = QLabel(
            "检查本机 Python / 依赖 / NVIDIA 显卡，并按驱动 CUDA 版本给出 PyTorch 安装建议。"
            "不同电脑的 N 卡与驱动不同，建议安装命令也会不同。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        summary_box = QGroupBox("概览")
        summary_form = QFormLayout(summary_box)
        self.summary_label = QLabel("-")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_form.addRow("结论:", self.summary_label)
        self.py_label = QLabel("-")
        self.py_label.setWordWrap(True)
        self.py_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_form.addRow("Python:", self.py_label)
        self.plat_label = QLabel("-")
        summary_form.addRow("系统:", self.plat_label)
        self.gpu_label = QLabel("-")
        self.gpu_label.setWordWrap(True)
        self.gpu_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_form.addRow("显卡:", self.gpu_label)
        self.torch_rec_label = QLabel("-")
        summary_form.addRow("建议 torch:", self.torch_rec_label)
        layout.addWidget(summary_box)

        table_box = QGroupBox("依赖检查")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["组件", "状态", "详情", "修复提示"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(260)
        self.table.setStyleSheet(
            "QTableWidget { background: #1e1e1e; gridline-color: #444; color: #ddd; }"
            "QHeaderView::section { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 4px; }"
        )
        table_layout.addWidget(self.table)
        layout.addWidget(table_box)

        cmd_box = QGroupBox("安装指令")
        cmd_layout = QVBoxLayout(cmd_box)
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("PyTorch 通道:"))
        self.tag_combo = QComboBox()
        self.tag_combo.addItems(["自动推荐", "cu124", "cu121", "cu118", "cpu"])
        self.tag_combo.currentTextChanged.connect(self._rebuild_commands)
        opt_row.addWidget(self.tag_combo)
        opt_row.addWidget(QLabel("PyPI 镜像:"))
        self.mirror_combo = QComboBox()
        self.mirror_combo.addItem("清华", "https://pypi.tuna.tsinghua.edu.cn/simple")
        self.mirror_combo.addItem("官方 PyPI", "https://pypi.org/simple")
        self.mirror_combo.addItem("阿里云", "https://mirrors.aliyun.com/pypi/simple")
        self.mirror_combo.currentIndexChanged.connect(self._rebuild_commands)
        opt_row.addWidget(self.mirror_combo)
        opt_row.addStretch()
        cmd_layout.addLayout(opt_row)

        self.cmd_edit = QTextEdit()
        self.cmd_edit.setReadOnly(True)
        self.cmd_edit.setFont(QFont("Consolas", 10))
        self.cmd_edit.setMinimumHeight(120)
        cmd_layout.addWidget(self.cmd_edit)
        layout.addWidget(cmd_box)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("重新检查")
        self.refresh_btn.clicked.connect(self.refresh_check)
        btn_row.addWidget(self.refresh_btn)
        self.copy_btn = QPushButton("复制安装指令")
        self.copy_btn.clicked.connect(self.copy_commands)
        btn_row.addWidget(self.copy_btn)
        self.install_reqs_btn = QPushButton("安装 requirements.txt")
        self.install_reqs_btn.setToolTip("安装除 torch CUDA 轮子外的项目依赖（使用上方镜像）")
        self.install_reqs_btn.clicked.connect(self.install_requirements)
        btn_row.addWidget(self.install_reqs_btn)
        self.install_torch_btn = QPushButton("安装/修复 PyTorch（CUDA）")
        self.install_torch_btn.setToolTip("按所选通道卸载旧 torch 并安装官方 CUDA/CPU 轮子")
        self.install_torch_btn.clicked.connect(self.install_torch)
        btn_row.addWidget(self.install_torch_btn)
        self.install_all_btn = QPushButton("一键安装（依赖 + PyTorch）")
        self.install_all_btn.clicked.connect(self.install_all)
        btn_row.addWidget(self.install_all_btn)
        self.stop_btn = QPushButton("停止安装")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_install)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setPlaceholderText("安装日志…")
        layout.addWidget(self.log_edit, stretch=1)

    def refresh_check(self):
        self._report = collect_env_report()
        r = self._report
        self.summary_label.setText(r.summary)
        self.py_label.setText(f"{r.python_version}\n{r.python_executable}")
        self.plat_label.setText(r.platform)
        if r.gpus:
            lines = []
            for g in r.gpus:
                mem = f" / {g.total_mem_mb} MB" if g.total_mem_mb else ""
                lines.append(f"GPU {g.index}: {g.name}{mem}")
            if r.driver_version or r.cuda_from_driver:
                lines.append(
                    f"驱动 {r.driver_version or '-'}，nvidia-smi CUDA {r.cuda_from_driver or '-'}"
                )
            self.gpu_label.setText("\n".join(lines))
        else:
            self.gpu_label.setText("未检测到 NVIDIA GPU（请安装驱动并确认 nvidia-smi 可用）")

        self.torch_rec_label.setText(
            f"{r.recommended_torch_tag}  →  {self._index_hint(r.recommended_torch_tag)}"
        )

        self.table.setRowCount(0)
        for item in r.items:
            row = self.table.rowCount()
            self.table.insertRow(row)
            status = "通过" if item.ok else ("缺失*" if item.required else "缺失")
            vals = [item.name, status, item.detail, item.fix_hint]
            for col, text in enumerate(vals):
                cell = QTableWidgetItem(text)
                if col == 1:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if item.ok:
                        cell.setForeground(QColor("#6bcf7f"))
                    else:
                        cell.setForeground(QColor("#ff6b6b" if item.required else "#f0c674"))
                self.table.setItem(row, col, cell)

        # sync auto tag
        if self.tag_combo.currentText() == "自动推荐" or self.tag_combo.currentIndex() == 0:
            self._rebuild_commands()
        else:
            self._rebuild_commands()

    def _selected_tag(self) -> str:
        text = self.tag_combo.currentText()
        if text == "自动推荐":
            if self._report:
                return self._report.recommended_torch_tag
            return "cpu"
        return text

    def _selected_mirror(self) -> str:
        return self.mirror_combo.currentData() or "https://pypi.tuna.tsinghua.edu.cn/simple"

    def _index_hint(self, tag: str) -> str:
        if tag == "cpu":
            return "https://download.pytorch.org/whl/cpu"
        return f"https://download.pytorch.org/whl/{tag}"

    def _rebuild_commands(self):
        tag = self._selected_tag()
        mirror = self._selected_mirror()
        cmds = build_pip_commands(tag, install_reqs=True, reinstall_torch=True, mirror=mirror)
        self.cmd_edit.setPlainText(format_commands_for_display(cmds))
        if self._report:
            self.torch_rec_label.setText(f"{tag}  →  {self._index_hint(tag)}")

    def copy_commands(self):
        text = self.cmd_edit.toPlainText().strip()
        if not text:
            return
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "已复制", "安装指令已复制到剪贴板")

    def _set_busy(self, busy: bool):
        for w in (
            self.refresh_btn, self.copy_btn, self.install_reqs_btn,
            self.install_torch_btn, self.install_all_btn, self.tag_combo, self.mirror_combo,
        ):
            w.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def install_requirements(self):
        mirror = self._selected_mirror()
        req = os.path.join(self.root, "requirements.txt")
        if not os.path.isfile(req):
            QMessageBox.warning(self, "提示", f"未找到 {req}")
            return
        cmds = [
            f'"{self.python}" -m pip install --upgrade pip',
            f'"{self.python}" -m pip install -r "{req}" -i {mirror}',
        ]
        self._start_commands(cmds, "安装 requirements.txt")

    def install_torch(self):
        tag = self._selected_tag()
        reply = QMessageBox.question(
            self,
            "安装 PyTorch",
            f"将卸载旧的 torch/torchvision/torchaudio，并安装通道 【{tag}】。\n"
            f"索引: {self._index_hint(tag)}\n\n"
            "是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmds = build_pip_commands(
            tag, install_reqs=False, reinstall_torch=True, mirror=self._selected_mirror()
        )
        self._start_commands(cmds, f"安装 PyTorch ({tag})")

    def install_all(self):
        tag = self._selected_tag()
        reply = QMessageBox.question(
            self,
            "一键安装",
            f"将依次：升级 pip → 安装 requirements.txt → 安装 PyTorch【{tag}】。\n"
            "可能需要几分钟，期间请保持网络畅通。\n\n是否开始？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmds = build_pip_commands(
            tag, install_reqs=True, reinstall_torch=True, mirror=self._selected_mirror()
        )
        self._start_commands(cmds, "一键安装")

    def _start_commands(self, cmds, title: str):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "提示", "已有安装任务在进行")
            return
        self._pending_cmds = list(cmds)
        self._cmd_index = 0
        self.log_edit.append(f"\n===== {title} =====")
        self._set_busy(True)
        self._run_next_command()

    def _run_next_command(self):
        if self._cmd_index >= len(self._pending_cmds):
            self.log_edit.append("\n全部命令执行完毕，正在重新检查环境…")
            self._set_busy(False)
            self.refresh_check()
            QMessageBox.information(self, "完成", "安装流程结束，请查看上方检查结果。")
            return

        raw = self._pending_cmds[self._cmd_index]
        self._cmd_index += 1
        self.log_edit.append(f"\n$ {raw}")

        args = self._split_cmd(raw)
        if not args:
            self.log_edit.append("无法解析命令，已跳过")
            self._run_next_command()
            return

        self.process = QProcess(self)
        self.process.setWorkingDirectory(self.root)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_proc_output)
        self.process.finished.connect(self._on_proc_finished)
        program, program_args = args[0], args[1:]
        self.process.start(program, program_args)

    def _split_cmd(self, raw: str):
        import shlex
        try:
            if os.name == "nt":
                return shlex.split(raw, posix=False)
            return shlex.split(raw)
        except Exception:
            return []

    def _on_proc_output(self):
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.log_edit.insertPlainText(data)
            self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _on_proc_finished(self, code: int, _status):
        if code != 0:
            self.log_edit.append(f"\n命令失败，退出码 {code}。后续步骤已中止。")
            self._set_busy(False)
            self._pending_cmds = []
            QMessageBox.warning(self, "安装失败", f"命令退出码 {code}，请查看日志。可复制指令手动执行。")
            return
        self._run_next_command()

    def stop_install(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
        self._pending_cmds = []
        self._set_busy(False)
        self.log_edit.append("\n已停止安装。")
