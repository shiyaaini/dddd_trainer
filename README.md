# dddd_trainer

[中文](#中文) | [English](#english)

基于 [sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer) 改进，配套 [ddddocr](https://github.com/sml2h3/ddddocr) 的验证码 / 文本 OCR 训练工具。

Fork / enhancement of [sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer) — training toolkit for [ddddocr](https://github.com/sml2h3/ddddocr) captcha / text OCR models.

---

## 中文

### 简介

基于 **PyTorch**，支持 **CRNN（序列识别，默认）** 与 **CNN / Word（整图分类）**，提供断点续训、自动导出 ONNX、**PyQt 图形界面**，以及与 ddddocr 自定义模型加载对接。

| 能力 | 说明 |
|------|------|
| 训练模式 | CRNN + CTC（多字符）；`--single` 开启 CNN + CrossEntropy（单标签） |
| 骨架网络 | `ddddocr`（默认）、MobileNetV2/V3、EfficientNetV2-S/M/L/XL |
| 数据标注 | 文件名标签，或 `labels.txt` + `images/` |
| 导出部署 | `*.onnx` + `charsets.json`，可用 ddddocr / onnxruntime |
| 图形界面 | 环境检查、项目、生成、标注、配置、训练、测试 |

**硬件建议：** NVIDIA GPU + CUDA 版 PyTorch。CPU / macOS 可跑，但会慢很多。

### Windows 一键环境（推荐）

1. 确保 `dist\` 中有：
   - `python-3.10.11-embed-amd64.zip`（绿色 Python）
   - `numpy-1.24.4-cp310-cp310-win_amd64.whl`（离线 numpy，推荐随仓库上传）
2. 双击 **`setup_env.bat`**  
   - 解压绿色 Python 到根目录 `runtime\`  
   - 安装 `requirements.txt` 基础依赖（含 PyQt6 等）  
   - **不安装 torch**
3. 双击 **`run_gui.bat`** 启动界面  
4. 在 GUI「环境检查」页按显卡一键安装 / 修复 PyTorch

整份工程目录（含 `runtime\`）可直接拷贝给他人使用。`runtime\` 本身不进 Git，由 `setup_env.bat` 生成。

### 常规安装（已有系统 Python）

```bash
# 建议 Python 3.10
git clone <this-repo>
cd dddd_trainer
pip install -r requirements.txt
# 再到 https://pytorch.org 按显卡安装 CUDA 版 torch
# 或启动 GUI 后在「环境检查」页安装
python gui_app.py
```

### 图形界面

| 页签 | 作用 |
|------|------|
| 环境检查 | Python / torch / 依赖 / N 卡；一键安装 PyTorch |
| 项目管理 | 创建 / 选择项目 |
| 批量生成 | 合成训练样本（可选） |
| 手动标注 | 校对标签，可选手动 ddddocr 预识别 |
| 训练配置 | 编辑 `config.yaml`、GPU 对照、恢复默认 |
| 训练 | 缓存、启停训练、日志 |
| 模型测试 | 拖拽测图、批量准确率、Python 调用示例 |

### 命令行

```bash
python app.py create my_project
python app.py create my_project --single   # CNN / Word，勿用于多字符验证码
python app.py cache my_project /path/to/images_set
python app.py cache my_project /path/to/images_set file
python app.py train my_project
python app.py export my_project
```

### 快速流程

```text
创建项目 → 准备数据 → 缓存 →（可选）改配置 → 训练 → 导出 onnx → 测试 / 部署
```

### 数据格式

大小写敏感：需要区分大小写时，标注本身就要正确。

**方式 A：文件名标签**（默认）`标签_随机hash.扩展名`

```text
/path/to/images_set/
  ├── mkGu_000001d00f140741741ed9916240d8d5.jpg
  └── abcd_a1b2c3d4.jpg
```

**方式 B：`labels.txt` + `images/`**  
每行：`相对 images 的路径` + 制表符 `\t` + `标签`

```text
aaaa/xxx.jpg	abcd
yyy.jpg	酱闷肘子
```

示例数据集（上游提供）：[数据集一](https://wwm.lanzoum.com/iUyYb0b5z3lg) · [数据集二](https://wwm.lanzoum.com/itczd0b5z3yj)

### 项目目录

```text
projects/{project_name}/
  ├── config.yaml
  ├── datasets/
  ├── inbox/
  ├── cache/
  ├── checkpoints/
  └── models/
        ├── *.onnx
        └── charsets.json
```

### 配置要点

路径：`projects/{name}/config.yaml`（也可在 GUI 修改）。

- `ImageChannel`: 1 灰度 / 3 彩色  
- `ImageHeight`: 建议 16 的倍数；`ImageWidth: -1` 表示按高度自适应（CRNN 常用）  
- `TARGET`: 同时满足 Acc / Epoch / Cost 后自动导出 ONNX  
- 验证码优先骨架 `ddddocr`  
- Windows 多进程异常时可将 `NUM_WORKERS` 设为 `0`

### 部署调用

```python
import ddddocr

ocr = ddddocr.DdddOcr(
    det=False, ocr=False, show_ad=False,
    import_onnx_path=r"projects/my_project/models/xxx.onnx",
    charsets_path=r"projects/my_project/models/charsets.json",
)
with open(r"test.png", "rb") as f:
    print(ocr.classification(f.read()))
```

### 常见问题

| 问题 | 处理 |
|------|------|
| 训练慢 / GPU 利用率低 | 确认 CUDA 版 torch；检查 `GPU`/`GPU_ID`；增大 `BATCH_SIZE` |
| 误装 CPU 版 torch | 「环境检查」选择 cu124/cu121/cu118 后一键安装 |
| 改了 Val / 数据集 | 重新 `cache` |
| 准确率上不去 | 检查标注；优先 `ddddocr`；补真实难例 |
| CNN / CRNN 选错 | 多字符用 CRNN；整图单类/单字才用 `--single` |

### 致谢

- 源仓库：[sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer)
- [ddddocr](https://github.com/sml2h3/ddddocr)
- CRNN 结构参考：[crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

### License

以仓库内声明为准（上游为 Apache-2.0）。用于自有业务的验证码 / OCR 训练与部署。

---

## English

### Overview

PyTorch trainer for captcha / text OCR models used with **ddddocr**. Supports **CRNN** (default, sequence) and **CNN / Word** (whole-image classification), checkpoint resume, automatic **ONNX** export, and a **PyQt GUI**.

| Feature | Notes |
|---------|--------|
| Modes | CRNN + CTC; `--single` for CNN + CrossEntropy |
| Backbones | `ddddocr` (default), MobileNetV2/V3, EfficientNetV2-S/M/L/XL |
| Labels | Filename tags, or `labels.txt` + `images/` |
| Deploy | `*.onnx` + `charsets.json` via ddddocr / onnxruntime |
| GUI | Env check, projects, generate, annotate, config, train, test |

**Hardware:** NVIDIA GPU + CUDA PyTorch recommended. CPU / macOS works but is slow.

### Windows one-click setup (recommended)

1. Make sure `dist\` contains:
   - `python-3.10.11-embed-amd64.zip` (embed Python)
   - `numpy-1.24.4-cp310-cp310-win_amd64.whl` (offline NumPy; recommended in the repo)
2. Run **`setup_env.bat`**  
   - Extracts embed Python to `runtime\`  
   - Installs base deps from `requirements.txt` (PyQt6, etc.)  
   - **Does not install torch**
3. Run **`run_gui.bat`** to open the GUI  
4. Install / repair PyTorch in the GUI **Env Check** tab

You can copy the whole project folder (including `runtime\`) to other machines. `runtime\` is gitignored and created by `setup_env.bat`.

### Standard install

```bash
# Python 3.10 recommended
git clone <this-repo>
cd dddd_trainer
pip install -r requirements.txt
# Install CUDA torch from https://pytorch.org (or via GUI Env Check)
python gui_app.py
```

### GUI tabs

| Tab | Purpose |
|-----|---------|
| Env Check | Python / torch / deps / NVIDIA; one-click torch install |
| Projects | Create / select projects |
| Generate | Optional synthetic samples |
| Annotate | Label review; optional ddddocr assist |
| Config | Edit `config.yaml`, GPU list, reset defaults |
| Train | Cache, start/stop, logs |
| Test | Drag-drop / batch accuracy, Python snippet |

### CLI

```bash
python app.py create my_project
python app.py create my_project --single   # CNN/Word only — not for multi-char captchas
python app.py cache my_project /path/to/images_set
python app.py cache my_project /path/to/images_set file
python app.py train my_project
python app.py export my_project
```

### Pipeline

```text
create → prepare data → cache → (optional) config → train → export onnx → test / deploy
```

### Data formats

Case-sensitive: labels must already use the correct case.

**A. Filename labels (default):** `label_randomhash.ext`

**B. `labels.txt` + `images/`:** each line `relpath_under_images\tlabel`

Upstream sample sets: [set 1](https://wwm.lanzoum.com/iUyYb0b5z3lg) · [set 2](https://wwm.lanzoum.com/itczd0b5z3yj)

### Project layout

```text
projects/{project_name}/
  ├── config.yaml
  ├── datasets/
  ├── inbox/
  ├── cache/
  ├── checkpoints/
  └── models/   # *.onnx + charsets.json
```

### Config tips

- `ImageChannel`: 1 gray / 3 color  
- `ImageHeight` multiple of 16; `ImageWidth: -1` for CRNN adaptive width  
- `TARGET` Acc / Epoch / Cost → auto ONNX export  
- Prefer backbone `ddddocr` for captchas  
- Set `NUM_WORKERS: 0` if Windows DataLoader multiprocessing fails

### Inference

```python
import ddddocr

ocr = ddddocr.DdddOcr(
    det=False, ocr=False, show_ad=False,
    import_onnx_path=r"projects/my_project/models/xxx.onnx",
    charsets_path=r"projects/my_project/models/charsets.json",
)
with open(r"test.png", "rb") as f:
    print(ocr.classification(f.read()))
```

### FAQ

| Issue | Fix |
|-------|-----|
| Slow / low GPU use | CUDA torch; check `GPU`/`GPU_ID`; larger `BATCH_SIZE` |
| CPU torch by mistake | Env Check → cu124/cu121/cu118 → install |
| Changed Val / data | Re-run `cache` |
| Low accuracy | Fix labels; use `ddddocr`; add hard real samples |
| Wrong mode | Multi-char → CRNN; single-label image → `--single` |

### Credits

- Upstream: [sml2h3/dddd_trainer](https://github.com/sml2h3/dddd_trainer)
- [ddddocr](https://github.com/sml2h3/ddddocr)
- CRNN reference: [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

### License

See repository license file (upstream Apache-2.0). Intended for training/deploying your own captcha / OCR models.
