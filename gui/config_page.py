import os
import copy
import subprocess

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)

from configs import Config
from utils.project_manager import ProjectManager


def list_gpus():
    """Return list of dicts: id, name, total_mem_mb, free_mem_mb, source."""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = int(props.total_memory / (1024 * 1024))
                free_mb = None
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    free_mb = int(free / (1024 * 1024))
                except Exception:
                    pass
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "total_mem_mb": total_mb,
                    "free_mem_mb": free_mb,
                    "source": "torch",
                })
            return gpus
    except Exception:
        pass

    # fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append({
                "id": int(parts[0]),
                "name": parts[1],
                "total_mem_mb": int(float(parts[2])),
                "free_mem_mb": int(float(parts[3])),
                "source": "nvidia-smi",
            })
    except Exception:
        pass
    return gpus


class ConfigPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = ProjectManager()
        self._build_ui()
        self.refresh_projects()
        self.refresh_gpu_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QGroupBox("选择项目")
        top_form = QFormLayout(top)
        proj_row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self.load_config)
        proj_row.addWidget(self.project_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_projects)
        proj_row.addWidget(refresh_btn)
        reload_btn = QPushButton("重新加载")
        reload_btn.clicked.connect(lambda: self.load_config(self.project_combo.currentText()))
        proj_row.addWidget(reload_btn)
        proj_row.addStretch()
        top_form.addRow("目标项目:", proj_row)
        self.config_path_label = QLabel("-")
        self.config_path_label.setWordWrap(True)
        self.config_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top_form.addRow("配置文件:", self.config_path_label)
        layout.addWidget(top)

        tip = QLabel(
            "修改后点「保存配置」写入项目的 config.yaml。训练目标需同时满足 Acc / Epoch / Cost 才会自动结束并导出 onnx。\n"
            "字符集 CharSet 由「缓存数据」生成，此处只读；改完后若已在训练请先停止再重新开始。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        # —— System ——
        sys_box = QGroupBox("System（系统）")
        sys_form = QFormLayout(sys_box)
        self.gpu_check = QCheckBox("使用 GPU")
        sys_form.addRow("GPU:", self.gpu_check)
        self.gpu_id_spin = QSpinBox()
        self.gpu_id_spin.setRange(0, 15)
        self.gpu_id_spin.setToolTip("对应下表「GPU ID」列；双击表格行可快速填入")
        sys_form.addRow("GPU_ID:", self.gpu_id_spin)

        gpu_box = QGroupBox("本机 GPU 列表（对照 GPU_ID）")
        gpu_layout = QVBoxLayout(gpu_box)
        gpu_tip = QLabel("查看物理显卡与配置项 GPU_ID 的对应关系。双击一行可将该卡 ID 写入上方 GPU_ID。")
        gpu_tip.setWordWrap(True)
        gpu_layout.addWidget(gpu_tip)
        self.gpu_table = QTableWidget(0, 5)
        self.gpu_table.setHorizontalHeaderLabels(["GPU ID", "显卡名称", "显存总量", "显存空闲", "来源"])
        header = self.gpu_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.gpu_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.gpu_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gpu_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gpu_table.setMaximumHeight(160)
        self.gpu_table.setStyleSheet(
            "QTableWidget { background: #1e1e1e; gridline-color: #444; color: #ddd; }"
            "QHeaderView::section { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 4px; }"
        )
        self.gpu_table.cellDoubleClicked.connect(self._on_gpu_row_double_clicked)
        gpu_layout.addWidget(self.gpu_table)
        gpu_btn_row = QHBoxLayout()
        refresh_gpu_btn = QPushButton("刷新 GPU 列表")
        refresh_gpu_btn.clicked.connect(self.refresh_gpu_table)
        gpu_btn_row.addWidget(refresh_gpu_btn)
        use_sel_btn = QPushButton("使用选中卡")
        use_sel_btn.clicked.connect(self._apply_selected_gpu)
        gpu_btn_row.addWidget(use_sel_btn)
        gpu_btn_row.addStretch()
        self.gpu_status_label = QLabel("")
        self.gpu_status_label.setWordWrap(True)
        gpu_btn_row.addWidget(self.gpu_status_label)
        gpu_layout.addLayout(gpu_btn_row)
        sys_form.addRow(gpu_box)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.01, 0.5)
        self.val_spin.setSingleStep(0.01)
        self.val_spin.setDecimals(3)
        self.val_spin.setToolTip("验证集比例，缓存数据时生效")
        sys_form.addRow("Val（验证集比例）:", self.val_spin)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("数据集目录（一般自动维护）")
        sys_form.addRow("Path:", self.path_edit)
        self.ext_edit = QLineEdit()
        self.ext_edit.setPlaceholderText("jpg, jpeg, png, bmp")
        self.ext_edit.setToolTip("允许的图片扩展名，逗号分隔")
        sys_form.addRow("Allow_Ext:", self.ext_edit)
        body_layout.addWidget(sys_box)

        # —— Model ——
        model_box = QGroupBox("Model（模型）")
        model_form = QFormLayout(model_box)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(-1, 4096)
        self.width_spin.setSpecialValueText("自适应 (-1)")
        self.width_spin.setToolTip("-1 表示按高度等比缩放宽度（CRNN 常用）")
        model_form.addRow("ImageWidth:", self.width_spin)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 512)
        model_form.addRow("ImageHeight:", self.height_spin)
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("1（灰度）", 1)
        self.channel_combo.addItem("3（彩色）", 3)
        model_form.addRow("ImageChannel:", self.channel_combo)
        self.word_check = QCheckBox("CNN / Word 模式（单字或整图分类）")
        model_form.addRow("Word:", self.word_check)
        self.charset_label = QLabel("-")
        self.charset_label.setWordWrap(True)
        self.charset_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_form.addRow("CharSet（只读）:", self.charset_label)
        body_layout.addWidget(model_box)

        # —— Train ——
        train_box = QGroupBox("Train（训练）")
        train_form = QFormLayout(train_box)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        train_form.addRow("BATCH_SIZE:", self.batch_spin)
        self.test_batch_spin = QSpinBox()
        self.test_batch_spin.setRange(1, 512)
        train_form.addRow("TEST_BATCH_SIZE:", self.test_batch_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.001)
        train_form.addRow("LR（学习率）:", self.lr_spin)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        train_form.addRow("DROPOUT:", self.dropout_spin)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam"])
        train_form.addRow("OPTIMIZER:", self.optimizer_combo)
        self.cnn_edit = QLineEdit()
        self.cnn_edit.setPlaceholderText("ddddocr")
        train_form.addRow("CNN.NAME:", self.cnn_edit)
        self.test_step_spin = QSpinBox()
        self.test_step_spin.setRange(100, 100000)
        self.test_step_spin.setSingleStep(100)
        self.test_step_spin.setToolTip("每隔多少 Step 在验证集上算 Acc")
        train_form.addRow("TEST_STEP:", self.test_step_spin)
        self.save_step_spin = QSpinBox()
        self.save_step_spin.setRange(100, 100000)
        self.save_step_spin.setSingleStep(100)
        self.save_step_spin.setToolTip("每隔多少 Step 保存 checkpoint")
        train_form.addRow("SAVE_CHECKPOINTS_STEP:", self.save_step_spin)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setToolTip("DataLoader 子进程数；Windows 下建议 2~4")
        train_form.addRow("NUM_WORKERS:", self.num_workers_spin)
        self.amp_check = QCheckBox("启用混合精度 AMP（需 NVIDIA GPU）")
        train_form.addRow("AMP:", self.amp_check)
        self.pin_memory_check = QCheckBox("pin_memory（加速 CPU→GPU 拷贝）")
        train_form.addRow("PIN_MEMORY:", self.pin_memory_check)
        self.cache_mem_check = QCheckBox("内存缓存已解码图片（小数据集推荐）")
        train_form.addRow("CACHE_IN_MEMORY:", self.cache_mem_check)
        body_layout.addWidget(train_box)

        # —— TARGET ——
        target_box = QGroupBox("Train.TARGET（结束条件）")
        target_form = QFormLayout(target_box)
        self.target_acc_spin = QDoubleSpinBox()
        self.target_acc_spin.setRange(0.0, 1.0)
        self.target_acc_spin.setSingleStep(0.01)
        self.target_acc_spin.setDecimals(3)
        self.target_acc_spin.setToolTip("验证准确率需高于此值")
        target_form.addRow("Accuracy:", self.target_acc_spin)
        self.target_epoch_spin = QSpinBox()
        self.target_epoch_spin.setRange(0, 100000)
        self.target_epoch_spin.setToolTip("至少训练超过该 Epoch 后才允许结束")
        target_form.addRow("Epoch（最少轮数）:", self.target_epoch_spin)
        self.target_cost_spin = QDoubleSpinBox()
        self.target_cost_spin.setRange(0.0, 10.0)
        self.target_cost_spin.setSingleStep(0.01)
        self.target_cost_spin.setDecimals(4)
        self.target_cost_spin.setToolTip("平均损失需低于此值")
        target_form.addRow("Cost（最大损失）:", self.target_cost_spin)
        target_tip = QLabel(
            "同时满足：Acc > Accuracy、Epoch > 最少轮数、AvgLoss < Cost → 导出 onnx 并结束训练。\n"
            "想更快结束可适当降低 Accuracy（如 0.90）或减少最少 Epoch。"
        )
        target_tip.setWordWrap(True)
        target_form.addRow(target_tip)
        body_layout.addWidget(target_box)

        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        btn_row.addWidget(self.save_btn)
        self.reset_btn = QPushButton("恢复默认配置")
        self.reset_btn.setToolTip(
            "将训练相关参数恢复为程序默认值，并立即写入 config.yaml。\n"
            "会保留：项目名、数据集 Path、CharSet、Word 模式。"
        )
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def refresh_projects(self):
        current = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if current and current in projects:
            self.project_combo.setCurrentText(current)
        self.project_combo.blockSignals(False)
        self.load_config(self.project_combo.currentText())

    def refresh_gpu_table(self):
        gpus = list_gpus()
        self.gpu_table.setRowCount(0)
        if not gpus:
            self.gpu_status_label.setText("未检测到 NVIDIA GPU（需安装 CUDA 版 PyTorch，或本机有 nvidia-smi）")
            self.gpu_id_spin.setRange(0, 15)
            return

        max_id = 0
        for info in gpus:
            row = self.gpu_table.rowCount()
            self.gpu_table.insertRow(row)
            gid = int(info["id"])
            max_id = max(max_id, gid)
            total = f"{info['total_mem_mb']} MB"
            free = f"{info['free_mem_mb']} MB" if info.get("free_mem_mb") is not None else "-"
            values = [str(gid), info["name"], total, free, info.get("source", "-")]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setData(Qt.ItemDataRole.UserRole, gid)
                self.gpu_table.setItem(row, col, item)

        self.gpu_id_spin.setRange(0, max(max_id, 15))
        self.gpu_status_label.setText(f"共检测到 {len(gpus)} 张 GPU")
        self._highlight_current_gpu()

    def _selected_gpu_id(self):
        rows = self.gpu_table.selectionModel().selectedRows() if self.gpu_table.selectionModel() else []
        if not rows:
            return None
        item = self.gpu_table.item(rows[0].row(), 0)
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return int(data) if data is not None else int(item.text())

    def _apply_selected_gpu(self):
        gid = self._selected_gpu_id()
        if gid is None:
            QMessageBox.information(self, "提示", "请先在表格中选中一张 GPU")
            return
        self.gpu_id_spin.setValue(gid)
        self.gpu_check.setChecked(True)
        self._highlight_current_gpu()
        self.gpu_status_label.setText(f"已选择 GPU_ID = {gid}（记得点「保存配置」）")

    def _on_gpu_row_double_clicked(self, row: int, _column: int):
        item = self.gpu_table.item(row, 0)
        if item is None:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        gid = int(data) if data is not None else int(item.text())
        self.gpu_id_spin.setValue(gid)
        self.gpu_check.setChecked(True)
        self._highlight_current_gpu()
        self.gpu_status_label.setText(f"已选择 GPU_ID = {gid}（记得点「保存配置」）")

    def _highlight_current_gpu(self):
        current = self.gpu_id_spin.value()
        for row in range(self.gpu_table.rowCount()):
            item = self.gpu_table.item(row, 0)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            gid = int(data) if data is not None else int(item.text())
            if gid == current:
                self.gpu_table.selectRow(row)
                return

    def _config_path(self, name: str) -> str:
        return os.path.join(self.pm.get_project_path(name), "config.yaml")

    def load_config(self, name: str):
        if not name:
            self.config_path_label.setText("-")
            self._set_enabled(False)
            return
        path = self._config_path(name)
        self.config_path_label.setText(path)
        if not os.path.isfile(path):
            self._set_enabled(False)
            QMessageBox.warning(self, "提示", f"未找到配置文件:\n{path}")
            return

        try:
            conf = Config(name).load_config()
        except Exception as e:
            self._set_enabled(False)
            QMessageBox.critical(self, "读取失败", str(e))
            return

        self._set_enabled(True)
        system = conf.get("System", {})
        model = conf.get("Model", {})
        train = conf.get("Train", {})
        target = train.get("TARGET", {})

        self.gpu_check.setChecked(bool(system.get("GPU", True)))
        self.gpu_id_spin.setValue(int(system.get("GPU_ID", 0)))
        self._highlight_current_gpu()
        self.val_spin.setValue(float(system.get("Val", 0.03)))
        self.path_edit.setText(str(system.get("Path", "") or ""))
        exts = system.get("Allow_Ext", ["jpg", "jpeg", "png", "bmp"])
        self.ext_edit.setText(", ".join(str(x) for x in exts))

        self.width_spin.setValue(int(model.get("ImageWidth", -1)))
        self.height_spin.setValue(int(model.get("ImageHeight", 64)))
        ch = int(model.get("ImageChannel", 1))
        idx = self.channel_combo.findData(ch)
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.word_check.setChecked(bool(model.get("Word", False)))
        charset = model.get("CharSet", [])
        if charset:
            shown = ", ".join(repr(c) if c == " " else str(c) for c in charset)
            self.charset_label.setText(f"{len(charset)} 个: [{shown}]")
        else:
            self.charset_label.setText("（空，缓存数据后会自动写入）")

        self.batch_spin.setValue(int(train.get("BATCH_SIZE", 32)))
        self.test_batch_spin.setValue(int(train.get("TEST_BATCH_SIZE", 32)))
        self.lr_spin.setValue(float(train.get("LR", 0.01)))
        self.dropout_spin.setValue(float(train.get("DROPOUT", 0.3)))
        opt = str(train.get("OPTIMIZER", "SGD"))
        oi = self.optimizer_combo.findText(opt)
        self.optimizer_combo.setCurrentIndex(oi if oi >= 0 else 0)
        cnn = train.get("CNN", {}) or {}
        self.cnn_edit.setText(str(cnn.get("NAME", "ddddocr")))
        self.test_step_spin.setValue(int(train.get("TEST_STEP", 1000)))
        self.save_step_spin.setValue(int(train.get("SAVE_CHECKPOINTS_STEP", 2000)))
        self.num_workers_spin.setValue(int(train.get("NUM_WORKERS", 2)))
        self.amp_check.setChecked(bool(train.get("AMP", True)))
        self.pin_memory_check.setChecked(bool(train.get("PIN_MEMORY", True)))
        self.cache_mem_check.setChecked(bool(train.get("CACHE_IN_MEMORY", True)))

        self.target_acc_spin.setValue(float(target.get("Accuracy", 0.97)))
        self.target_epoch_spin.setValue(int(target.get("Epoch", 20)))
        self.target_cost_spin.setValue(float(target.get("Cost", 0.05)))

    def _set_enabled(self, enabled: bool):
        for w in (
            self.gpu_check, self.gpu_id_spin, self.val_spin, self.path_edit, self.ext_edit,
            self.width_spin, self.height_spin, self.channel_combo, self.word_check,
            self.batch_spin, self.test_batch_spin, self.lr_spin, self.dropout_spin,
            self.optimizer_combo, self.cnn_edit, self.test_step_spin, self.save_step_spin,
            self.num_workers_spin, self.amp_check, self.pin_memory_check, self.cache_mem_check,
            self.target_acc_spin, self.target_epoch_spin, self.target_cost_spin,
            self.save_btn, self.reset_btn,
        ):
            w.setEnabled(enabled)

    def reset_to_defaults(self):
        name = self.project_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        path = self._config_path(name)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "提示", f"未找到配置文件:\n{path}")
            return

        reply = QMessageBox.question(
            self,
            "恢复默认配置",
            "确定将当前项目的可训练参数恢复为默认值并写入 config.yaml？\n\n"
            "会保留：项目名、数据集 Path、CharSet、Word 模式。\n"
            "会重置：GPU / 图像尺寸 / 训练超参 / TARGET 等。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            current = Config(name).load_config()
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        defaults = copy.deepcopy(Config(name).config_dict)
        defaults["System"]["Project"] = name
        # 保留项目相关字段，避免误清空数据路径与字符集
        defaults["System"]["Path"] = current.get("System", {}).get("Path", "") or ""
        defaults["Model"]["CharSet"] = current.get("Model", {}).get("CharSet", []) or []
        defaults["Model"]["Word"] = bool(current.get("Model", {}).get("Word", False))

        try:
            Config(name).make_config(config_dict=defaults)
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))
            return

        self.load_config(name)
        QMessageBox.information(self, "完成", f"已恢复默认配置并写入:\n{path}")

    def save_config(self):
        name = self.project_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        path = self._config_path(name)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "提示", f"未找到配置文件:\n{path}")
            return

        try:
            conf = Config(name).load_config()
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        exts = [e.strip().lstrip(".") for e in self.ext_edit.text().split(",") if e.strip()]
        if not exts:
            QMessageBox.warning(self, "提示", "Allow_Ext 不能为空")
            return

        conf.setdefault("System", {})
        conf.setdefault("Model", {})
        conf.setdefault("Train", {})
        conf["Train"].setdefault("CNN", {})
        conf["Train"].setdefault("TARGET", {})

        conf["System"]["Project"] = name
        conf["System"]["GPU"] = self.gpu_check.isChecked()
        conf["System"]["GPU_ID"] = self.gpu_id_spin.value()
        conf["System"]["Val"] = self.val_spin.value()
        conf["System"]["Path"] = self.path_edit.text().strip()
        conf["System"]["Allow_Ext"] = exts

        conf["Model"]["ImageWidth"] = self.width_spin.value()
        conf["Model"]["ImageHeight"] = self.height_spin.value()
        conf["Model"]["ImageChannel"] = int(self.channel_combo.currentData())
        conf["Model"]["Word"] = self.word_check.isChecked()
        # CharSet 保持原值，不在此覆盖

        conf["Train"]["BATCH_SIZE"] = self.batch_spin.value()
        conf["Train"]["TEST_BATCH_SIZE"] = self.test_batch_spin.value()
        conf["Train"]["LR"] = self.lr_spin.value()
        conf["Train"]["DROPOUT"] = self.dropout_spin.value()
        conf["Train"]["OPTIMIZER"] = self.optimizer_combo.currentText()
        conf["Train"]["CNN"]["NAME"] = self.cnn_edit.text().strip() or "ddddocr"
        conf["Train"]["TEST_STEP"] = self.test_step_spin.value()
        conf["Train"]["SAVE_CHECKPOINTS_STEP"] = self.save_step_spin.value()
        conf["Train"]["NUM_WORKERS"] = self.num_workers_spin.value()
        conf["Train"]["AMP"] = self.amp_check.isChecked()
        conf["Train"]["PIN_MEMORY"] = self.pin_memory_check.isChecked()
        conf["Train"]["CACHE_IN_MEMORY"] = self.cache_mem_check.isChecked()
        conf["Train"]["TARGET"]["Accuracy"] = self.target_acc_spin.value()
        conf["Train"]["TARGET"]["Epoch"] = self.target_epoch_spin.value()
        conf["Train"]["TARGET"]["Cost"] = self.target_cost_spin.value()

        try:
            Config(name).make_config(config_dict=conf)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return

        QMessageBox.information(self, "成功", f"已保存:\n{path}")
        self.load_config(name)
