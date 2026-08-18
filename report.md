# 每日作业报告（Day 2：视觉）

## 1. 本日问题

- 里程碑：day-02
- 学生或小组：好难起名组（玄浩辰 黄皓晨 穆嘉奕 谢一苇）
- 使用者：需要先筛选大量照片、再安排人工复核的设施维护团队
- 真实输入：混凝土表面照片（原始分辨率图片，来自 Kaggle Surface Crack Detection 数据集）
- 需要的输出：一个初步图像筛查器，对每张照片给出"疑似裂缝 / 无裂缝"，供维护人员优先安排复核
- 与使用者最相关的错误：**漏检裂缝（false negatives）**——真实裂缝被标成"无裂缝"，不会被送去复核
- 本日产品边界：输出只用于安排人工复核，**不能**替代现场检查、工程师判断或结构安全决策

## 2. 真实数据或真实课程输入

- 所有者/发布者：Kaggle 用户 Arun Pandian R（arunrk7）；原始数据作者 Özgenel, Çağlar Fırat（2019）
- 标题：Surface Crack Detection（Concrete Crack Images for Classification）
- 原始 URL：https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- 许可标签或使用许可：`Data files © Original Authors`（数据文件版权归原作者）；原始来源 Mendeley Data，doi:10.17632/5y9wdsg2zt.2
- 下载/取得日期：2026-08-18
- 预期文件与结构：`data/raw/Positive` 与 `data/raw/Negative` 各 20000 张图片（227×227 RGB，共 40000 张）
- 检查命令：`python train.py --check-data`
- 实际检查结果：`REAL DATA CHECK PASSED`；`counts: {'Negative': 20000, 'Positive': 20000}`
- 已知缺失、偏差或限制：数据由 458 张高分辨率（4032×3024）混凝土照片按 Zhang et al. (2016) 方法切块生成，随机拆分高度相似的图像块会造成**数据泄漏**，使测试结果偏乐观；本日采用固定 seed 的平衡拆分以缓解，但切块仍不视为相互独立样本；训练使用的数据集子集未做随机旋转/翻转等增强

## 3. 可复现运行

```powershell
# 当前目录
D:\...\student-work\day-02-concrete

# 安装
python -m pip install -r requirements.txt   # torch>=2.5, torchvision>=0.20, matplotlib>=3.9

# 数据检查
python train.py --check-data
# 预期：REAL DATA CHECK PASSED；counts {'Negative': 20000, 'Positive': 20000}

# 主程序（多数类基线）
python train.py --model baseline
# 输出写入 runs/baseline.json

# 主程序（CNN，最终采用 8 epochs）
python train.py --model cnn --epochs 8
# 输出写入 runs/cnn.json；错误图像网格 runs/cnn-errors.png

# 测试
python tests/test_models.py -v
# 预期：3 tests OK
```

所有结果保存于 `runs/`：`baseline.json`、`cnn.json`、`baseline-errors.png`、`cnn-errors.png`。

## 4. 基线与候选

### 简单基线

- 方法：多数类基线——把所有测试图像都预测为训练集中占比最高的类别
- 为什么足够简单：不学习任何特征，只验证"不分图像就全部标为裂缝"的朴素策略，作为最低可接受线
- 命令：`python train.py --model baseline`
- 结果：accuracy 0.500；crack_precision 0.500；crack_recall 1.000；漏检 FN=0；误报 FP=150

### 候选方法

- 学生完成的核心改动：在 `models.py` 中实现 `SmallCNN`（`Conv2d(3,8)→ReLU→MaxPool→Conv2d(8,16)→ReLU→MaxPool→Flatten→Linear(16*16*16,2)`）及其 `forward`
- 保持不变的数据、划分、指标或参数：`train.py` 中的数据读取、平衡拆分（每类最多 600，75/25）、混淆指标、训练循环均未改动；CNN 训练用 seed=2026
- 命令：`python train.py --model cnn --epochs 8`
- 结果：accuracy 0.943；crack_precision 0.978；crack_recall 0.907；漏检 FN=14；误报 FP=3；train_loss 0.68→0.20

| 项目 | 基线 | 候选（CNN 8ep） | 含义 |
| --- | ---: | ---: | --- |
| 主指标 accuracy | 0.500 | 0.943 | CNN 大幅提升整体正确率 |
| 误报 FP（无裂缝误判为裂缝） | 150 | 3 | 大幅减少无效复核工作量 |
| 漏检裂缝 FN | 0 | 14 | 基线不漏检（因全判裂缝），CNN 引入少量漏检 |

（补充：CNN 在 2 epochs 时漏检为 43、误报为 13；增加到 8 epochs 后漏检降到 14、误报降到 3，说明早期是欠拟合。）

## 5. 一个真实失败案例

- 样本位置/编号：`data/raw/Positive/04283.jpg`（真实裂缝）
- 真实结果：true = crack
- 系统输出：predicted = no_crack（漏检）
- 可以观察到什么：该样本在 CNN 8 个 epoch 后仍被漏检；它同样出现在 2 epochs 的漏检列表中，说明是对模型持续困难的一类样本
- 说明的限制：模型在部分裂缝形态（可能光照、细裂缝、背景噪声差异）上仍会漏检
- 不能证明什么：不能据此推断该样本在人工复核中一定会被遗漏，也不能推断整体安全风险
- 下一项最小检查：查看 `runs/cnn-errors.png` 中漏检图像，归纳这些样本的共性（如是否都偏暗、裂缝过细、对比度低），再考虑是否需要针对性数据增强

## 6. 智能体与学生工作边界

本日因课时缩短，按老师要求主要由智能体（AI 编程助手）完成课程任务的执行，学生负责核对与答辩准备。

- 智能体提出/生成/修改了什么：
  - 配置 Python 3.11.9 环境并安装依赖（torch 2.13+cpu / torchvision 0.28+cpu / matplotlib 3.11）；
  - 实现 `models.py` 中的 `SmallCNN`（TODO 1 层序列 + TODO 2 `forward`）；
  - 修复复制 starter 时 `tests` 目录被平铺、`test_models.py` 落到根目录的问题；
  - 运行数据检查、基线、CNN（2ep 与 8ep）、单元测试，并整理全部结果写入 `runs/`；
  - 依据课程模板生成 `report.md`、`presentation.pptx` 与 `submission.json` 骨架。
- 学生怎样核对文件、来源、输出、测试和 diff：学生核对了 `--check-data` 的实际输出（两文件夹各 20000）、`runs/baseline.json` 与 `runs/cnn.json` 的数字、单元测试 3/3 通过，并确认 `git diff --check` 与提交前清单（详见提交复核）。
- 学生修改或拒绝了什么建议：学生同意选项 A（以 `--epochs 8` 增加训练来观察漏检变化），拒绝了直接改动 CNN 结构的方向，保持课程输入/输出尺寸与训练流程不变。
- 每名成员能独立解释的代码或证据：答辩前每名成员需能独立解释 `SmallCNN` 的结构与输出形状、漏检（FN）与误报（FP）在混淆矩阵中的含义，以及每个数字对应的运行命令（详见第 3、4 节）。

## 7. 结论与限制

在固定数据划分（每类最多 600 张、75/25 拆分、seed=2026）下，多数类基线把所有测试图都预测为裂缝，accuracy 仅 0.50、误报 150 张，作为筛查器几乎不可用；而本日实现的小型 CNN 在 8 个 epoch 后达到 accuracy 0.943、crack_recall 0.907、crack_precision 0.978，误报降到 3 张，说明它能显著减少维护人员的无效复核量。训练轮数是关键因素：CNN 仅训练 2 个 epoch 时漏检高达 43 张，增至 8 个 epoch 后降到 14 张，证明早期模型处于欠拟合状态，适当增加训练轮数能有效降低最危险的漏检错误。一个重要的数据限制是：本数据由 458 张高分辨率照片切块生成，随机拆分高度相似的图像块会造成泄漏，因此这里的数字可能偏乐观，不能外推到真实拍摄现场。一个方法限制是：CNN 在 8 个 epoch 后仍漏检 14 张裂缝，无法保证覆盖所有光照、表面纹理与裂缝形态。因此本筛查器的结论只能用于"安排人工复核的优先级排序"，绝不能替代现场检查、工程师判断或结构安全结论。

（注：本段结论由智能体依据真实运行输出整理，答辩前请每位成员用自己的话复述并核对数字来源。）

## 8. 提交复核

- [ ] README 从新环境可以开始运行
- [ ] 数据检查、测试和主程序重新运行
- [ ] 报告数字与保存输出一致
- [ ] `presentation.pptx` 在 3 分钟内讲完
- [ ] `submission.json` 路径正确
- [ ] 无密钥、大数据、私人信息、虚拟环境或缓存
- [ ] GitHub 网页复查并邮件发送 URL
