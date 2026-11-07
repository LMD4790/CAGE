# CAGE

CAGE 是一种多视图代码克隆分类模型，融合显式比较特征、AST 结构关系和 CodeBERT 语义表示，支持 Java 和 C 代码。

## 方法概述

CAGE 从三个视图提取代码对的信息：

- **显式特征视图（Explicit feature view）**：通过 LightGBM 将词法比较特征转换为类别支持向量。
- **AST 结构视图（AST structural view）**：根据 AST 统计量构建代码对的结构关系表示。
- **上下文语义视图（Contextual semantic view）**：通过 CodeBERT 联合编码两个代码片段，提取上下文语义表示。

模型将语义表示与类别支持向量融合，通过交叉注意力聚合 AST 结构信息，完成代码克隆分类。

## 安装

在仓库根目录执行：

```bash
python -m pip install -e ".[dev]"
```

## 数据准备

将数据整理为 UTF-8 编码的 JSONL 文件，分别保存为：

```text
data/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

每行包含一个代码对：

```json
{"id":"pair-001","code1":"int inc(int x) { return x + 1; }","code2":"int next(int y) { return y + 1; }","label":1}
```

| 字段 | 说明 |
| --- | --- |
| `code1`、`code2` | 两个代码片段 |
| `label` | 类别标签，取值为 `0` 至 `K-1`，其中 `K` 为类别数 |
| `id` | 可选的样本标识 |

训练集、验证集和测试集使用相同的类别映射。分组和预处理选项见[训练与评估](docs/REPRODUCIBILITY.md)。

## 训练与预测

### 1. 生成类别支持向量

```bash
python scripts/prepare_support.py --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl --test-jsonl data/test.jsonl --num-classes 5 --output-dir prepared
```

将 `--num-classes` 设置为任务的类别数。训练集使用折外预测生成支持向量，验证集和测试集使用训练集拟合的 LightGBM 模型生成支持向量。

输出保存在 `prepared/`，包括处理后的数据、支持模型和数据清单。后续训练直接使用该目录。

### 2. 训练 CAGE

```bash
python train.py --train-jsonl prepared/train.jsonl --val-jsonl prepared/val.jsonl --test-jsonl prepared/test.jsonl --model-name microsoft/codebert-base --language java --output-dir runs/cage
```

处理 C 代码时，将 `--language java` 改为 `--language c`。`--model-name` 也支持本地 CodeBERT 模型目录。

程序根据验证集 Macro-F1 保存最佳模型，并输出测试集结果。模型、训练记录和预测文件保存在 `runs/cage/`。

实验使用 NVIDIA RTX 3090。

### 3. 预测新的代码对

```bash
python predict.py --model-dir runs/cage --input-jsonl data/new_pairs.jsonl --support-model prepared/support_model.joblib --output-jsonl runs/predictions.jsonl
```

预测输入包含 `code1` 和 `code2`，可附带 `id`。输出包含样本标识、预测类别和各类别的概率。

## 测试

```bash
python -m pytest -q
```

运行包含合成数据的完整示例：

```bash
python examples/verify_installation.py --output-dir runs/example
```

## 文档

- [特征定义](docs/FEATURES.md)
- [训练与评估](docs/REPRODUCIBILITY.md)

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
