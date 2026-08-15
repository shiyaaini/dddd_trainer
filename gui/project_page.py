from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QComboBox, QMessageBox, QGroupBox,
)

from utils.project_manager import ProjectManager


class ProjectPage(QWidget):
    projects_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self._build_ui()
        self.refresh_projects()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        help_box = QGroupBox("说明")
        help_layout = QVBoxLayout(help_box)
        help = QLabel(
            "项目管理用于创建 / 选择训练项目。推荐流程：\n"
            "创建项目 →（可选）批量生成 / 手动标注 → 训练页缓存数据 → 训练 → 模型测试。\n\n"
            "目录约定：\n"
            "  · datasets/  已标注完成的训练样本（文件名标签 或 labels.txt）\n"
            "  · inbox/     待标注图片（在「手动标注」中处理后再进入 datasets）\n"
            "  · models/    导出的 onnx 与 charsets.json\n\n"
            "模式选择：\n"
            "  · CRNN（默认）多字符验证码 / 序列 OCR，绝大多数场景选这个\n"
            "  · CNN（勾选下方）整图单标签分类或单字识别，多字符图不要用\n\n"
            "数据集规模建议（验证码）：\n"
            "  · 简单 4 位、干扰少：约 3,000～8,000 张\n"
            "  · 一般噪声 / 变形：约 1～3 万张\n"
            "  · 多字体 / 扭曲 / 中文：3 万以上，且每个字符尽量出现数百次\n"
            "  · 少于 1,000 张容易过拟合；实盘不准时优先补真实图与难例"
        )
        help.setWordWrap(True)
        help.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        help_layout.addWidget(help)
        layout.addWidget(help_box)

        create_box = QGroupBox("创建项目")
        create_layout = QVBoxLayout(create_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("项目名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: captcha_v1（勿含 \\ / : * ? \" < > |）")
        row.addWidget(self.name_edit)
        create_layout.addLayout(row)

        self.cnn_check = QCheckBox("CNN 模式 (--single，单字/分类场景；多字符验证码请勿勾选)")
        create_layout.addWidget(self.cnn_check)

        btn_row = QHBoxLayout()
        self.create_btn = QPushButton("创建项目")
        self.create_btn.clicked.connect(self.create_project)
        btn_row.addWidget(self.create_btn)
        btn_row.addStretch()
        create_layout.addLayout(btn_row)

        tip = QLabel(
            "创建后会在 projects/{名称}/ 下生成 config.yaml、datasets、inbox、cache、checkpoints、models。"
        )
        tip.setWordWrap(True)
        create_layout.addWidget(tip)
        layout.addWidget(create_box)

        list_box = QGroupBox("已有项目")
        list_layout = QVBoxLayout(list_box)
        list_row = QHBoxLayout()
        list_row.addWidget(QLabel("项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(240)
        list_row.addWidget(self.project_combo)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_projects)
        list_row.addWidget(self.refresh_btn)
        list_row.addStretch()
        list_layout.addLayout(list_row)

        self.path_label = QLabel("路径: -")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        list_layout.addWidget(self.path_label)
        self.datasets_label = QLabel("数据集: -")
        self.datasets_label.setWordWrap(True)
        self.datasets_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        list_layout.addWidget(self.datasets_label)
        select_tip = QLabel("下方各页签会尽量同步当前选中的项目；也可在各页自行切换。")
        select_tip.setWordWrap(True)
        list_layout.addWidget(select_tip)
        self.project_combo.currentTextChanged.connect(self._on_project_selected)
        layout.addWidget(list_box)

        layout.addStretch()

    def refresh_projects(self):
        current = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if current and current in projects:
            self.project_combo.setCurrentText(current)
        self.project_combo.blockSignals(False)
        self._on_project_selected(self.project_combo.currentText())
        self.projects_changed.emit()

    def _on_project_selected(self, name: str):
        if not name:
            self.path_label.setText("路径: -")
            self.datasets_label.setText("数据集: -")
            return
        self.path_label.setText(f"路径: {self.pm.get_project_path(name)}")
        self.datasets_label.setText(f"数据集: {self.pm.get_datasets_path(name)}")

    def create_project(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        if any(ch in name for ch in '\\/:*?"<>|'):
            QMessageBox.warning(self, "提示", "项目名称包含非法字符")
            return
        ok = self.pm.create_project(name, single=self.cnn_check.isChecked())
        if ok:
            mode = "CNN / Word" if self.cnn_check.isChecked() else "CRNN"
            QMessageBox.information(
                self,
                "成功",
                f"项目已创建: {name}\n模式: {mode}\n"
                f"数据集目录: {self.pm.get_datasets_path(name)}\n"
                f"待标注目录: {self.pm.get_inbox_path(name)}\n\n"
                "建议样本量：简单验证码约 3千～8千张，一般 1～3 万张。",
            )
            self.name_edit.clear()
            self.refresh_projects()
            self.project_combo.setCurrentText(name)
        else:
            QMessageBox.critical(self, "失败", f"创建失败，项目可能已存在: {name}")

    def current_project(self) -> str:
        return self.project_combo.currentText().strip()
