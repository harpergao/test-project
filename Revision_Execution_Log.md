# FACDNet Revision Execution Log

## 中文说明

### 文档用途

本文件作为 FACDNet 论文修改工作的统一记录入口，用于持续跟踪：

- 基于 `Reviewers.md` 提炼出的执行清单
- 审稿意见与论文修改动作之间的对应关系
- 后续计划调整
- 具体修改记录
- 当前进展状态

相关文件：
- 审稿意见：`/mnt/data1/gyf/code/newtest/ChangeFormer/Reviewers.md`
- 论文目录：`/mnt/data1/gyf/code/newtest/ChangeFormer/Feature-Aligned and Frequency-Aware Network for High-Resolution Remote Sensing Change Detection`
- 代码目录：`/mnt/data1/gyf/code/newtest/ChangeFormer`

### 当前状态

- 状态：审稿意见梳理中
- 最后更新时间：2026-04-24
- 协作方：Codex + 用户

### 中文执行清单

#### A. 审稿意见梳理

- [x] 对 `ChangeFormer/Reviewers.md` 中编辑和审稿人的意见完成编号
- [x] 建立映射表：
  审稿意见 -> 论文位置 -> 修改方式 -> 优先级
- [ ] 将每条意见进一步分类为：
  文字补充 / 图表修改 / 实验补充 / 代码支撑 / 参考文献 / 语言润色

#### B. 新颖性定位与相关工作

- [ ] 重写引言中的新颖性定位
- [ ] 明确 FACDNet 与 Transformer 类变化检测方法的差异
- [ ] 明确 FACDNet 与频域/频率感知方法的差异
- [ ] 解释 FACDNet 为什么不是简单模块拼接
- [ ] 补充一个与最相关方法的对比表

#### C. 方法细节与可复现性

- [ ] 明确 Siamese encoder/backbone 的具体配置
- [ ] 明确各 stage 的尺度和通道维度
- [ ] 明确 patch size、stride、embed dims、depths、attention heads
- [ ] 明确 FDAM 在每个编码阶段中的插入位置
- [ ] 明确 DWT/IDWT 在各尺度特征上的处理方式
- [ ] 明确 decoder 结构与最终预测流程
- [ ] 明确训练监督路径与各输出头的作用

#### D. 概念合理性与理论回应

- [ ] 解释 FDAM 对齐的是什么，不对齐的是什么
- [ ] 讨论 FDAM 是否可能错误抑制真实语义变化
- [ ] 补充为什么 FDAM 不会破坏真实变化语义的说明或证据
- [ ] 回应频域重建缺少一致性约束的问题
- [ ] 解释 FAFM 如何获得有效训练信号
- [ ] 判断是否需要增加轻量 consistency constraint 或 auxiliary supervision

#### E. 实验与证据补充

- [ ] 增加更强、更新的 baseline
- [ ] 增加效率分析：
  参数量 / FLOPs / 如可行则补充推理速度
- [ ] 将消融实验扩展到 LEVIR-CD 之外
- [ ] 至少增加一个额外数据集的消融，优先 SYSU-CD
- [ ] 增加 FAFM 局部窗口大小的消融
- [ ] 增加小波基选择分析或更充分的解释
- [ ] 如可行，补充更细粒度的模块/损失设计消融
- [ ] 强化定性可视化示例

#### F. 图表修改

- [ ] 重画 Figure 1，使整体流程清晰可见
- [ ] 在图中和图注中明确解释 `(a)`、`(b)`、`(c)`
- [ ] 在图中显示 FDAM 在各编码阶段中的位置
- [ ] 改进 Table III 的配置展示方式
- [ ] 明确替换或定义 `Base`

#### G. 数据集与结果分析

- [ ] 补充 WHU-CD 的样本数量和主要特点
- [ ] 平衡三个数据集的描述
- [ ] 修改 SYSU-CD 的结果分析，解释 Recall/F1 提升而 Precision 略降
- [ ] 强化定量与定性结果的解释

#### H. 语言、格式与参考文献

- [ ] 修改全文拼写和语法问题
- [ ] 将 `original dimension` 等模糊表达改为精确表述
- [ ] 统一公式标点为英文标点
- [ ] 统一全文术语与模块命名
- [ ] 规范参考文献格式
- [ ] 补全会议全称、卷号、期号、页码等缺失信息

#### I. 回复信准备

- [ ] 起草逐条回复框架
- [ ] 对每条意见记录：
  问题 -> 修改动作 -> 论文位置 -> 证据
- [ ] 标记未解决项和待补实验

### 中文编号说明

#### 编辑意见

- `AE-1` 方法新颖性不足，尚未清楚证明其超出已有思想组合。
- `AE-2` 方法描述和可复现性不足，包括实现细节、数学形式、设计理由和结构图示。
- `AE-3` 实验验证不充分，包括 baseline 偏旧、消融范围有限、缺少效率分析。
- `AE-4` 特征对齐与频域重建的概念合理性存在疑问，包括真实变化可能被误对齐，以及缺少显式一致性/直接监督。
- `AE-5` 表达质量仍需提升，包括图示、语言、格式和参考文献。

#### 审稿人 1

- `R1-1` 补充 WHU-CD 的样本规模或主要特征。
- `R1-2` 解释 SYSU-CD 上 Precision 略降但 Recall/F1 提升的原因。
- `R1-3` 明确 Table III 中 `Base` 的含义。
- `R1-4` 解释为什么选择 Haar wavelet。
- `R1-5` 解释或分析公式中局部邻域 `Omega`（如 `3x3`）的选取依据。
- `R1-6` 修正拼写、语法和公式标点问题。
- `R1-7` 修正参考文献格式不完整或不统一的问题。

#### 审稿人 2

- `R2-1` 更清楚地证明方法新颖性，并与已有 Transformer-based CD 方法区分开。
- `R2-2` 补充缺失的实现细节，增强可复现性。
- `R2-3` 更明确地说明与最相关工作的差异，以及现有方法尚未解决的问题。
- `R2-4` 补充更强的近期 baseline 和效率分析。
- `R2-5` 扩展消融实验，不仅限于模块组合和 `lambda`。
- `R2-6` 提升英文表达与排版质量。

#### 审稿人 3

- `R3-1` 重画 Figure 1，使整体结构和各子模块位置更清楚。
- `R3-2` 解释 semantic 与 non-semantic 成分的区分，并讨论 FDAM 是否可能误对齐真实变化。
- `R3-3` 回应频域重建缺少一致性约束的问题。
- `R3-4` 更清楚地给出 Wavelet Loss 的数学形式。
- `R3-5` 解释或改进 FAFM 缺少直接监督的问题。
- `R3-6` 将消融实验扩展到 LEVIR-CD 之外。
- `R3-7` 增加与更新方法的比较。
- `R3-8` 修正全文拼写和语法问题。

## Document Purpose

This file is the single source of truth for the revision workflow of the FACDNet manuscript.

It records:
- the execution checklist derived from `Reviewers.md`
- the mapping between reviewer comments and manuscript changes
- subsequent plan changes
- detailed modification records
- progress updates

Related files:
- Review comments: `/mnt/data1/gyf/code/newtest/ChangeFormer/Reviewers.md`
- Manuscript directory: `/mnt/data1/gyf/code/newtest/ChangeFormer/Feature-Aligned and Frequency-Aware Network for High-Resolution Remote Sensing Change Detection`
- Code directory: `/mnt/data1/gyf/code/newtest/ChangeFormer`

## Current Status

- Status: review triage in progress
- Last updated: 2026-04-24
- Owner: Codex + user

## Execution Checklist

### A. Review Comment Triage

- [x] Number every reviewer/editor comment from `ChangeFormer/Reviewers.md`
- [ ] Build a mapping table:
  reviewer comment -> manuscript section -> modification type -> priority
- [ ] Classify each item as:
  text-only / figure / experiment / code / reference / language

### B. Novelty Positioning and Related Work

- [ ] Rewrite the novelty positioning in the introduction
- [ ] Clarify the difference between FACDNet and Transformer-based CD methods
- [ ] Clarify the difference between FACDNet and frequency-aware CD methods
- [ ] Explain why FACDNet is not a simple composition of existing modules
- [ ] Add a focused comparison table for closely related methods

### C. Method Clarity and Reproducibility

- [ ] Explicitly describe the Siamese encoder/backbone configuration
- [ ] Explicitly list stage-wise feature scales and channel dimensions
- [ ] Clarify patch size, stride, embed dimensions, depths, and attention heads
- [ ] Clarify where FDAM is inserted in each encoding stage
- [ ] Clarify how DWT/IDWT are applied across feature scales
- [ ] Clarify the decoder structure and final prediction pipeline
- [ ] Clarify the training supervision path and output heads

### D. Conceptual Soundness and Theoretical Response

- [ ] Add a dedicated explanation of what FDAM aligns and what it does not align
- [ ] Discuss the risk of incorrectly suppressing real semantic changes
- [ ] Add explanation or evidence for why semantic changes are preserved after FDAM
- [ ] Respond to the missing consistency-constraint concern for frequency reconstruction
- [ ] Explain how FAFM receives effective training signal
- [ ] Decide whether to add a new lightweight consistency constraint or auxiliary supervision

### E. Experiments and Evidence

- [ ] Add stronger and more recent baselines
- [ ] Add efficiency analysis:
  parameters / FLOPs / inference speed if feasible
- [ ] Expand ablation beyond LEVIR-CD
- [ ] Add at least one additional dataset for ablation, preferably SYSU-CD
- [ ] Add window-size ablation for the local neighborhood in FAFM
- [ ] Add wavelet-basis analysis or justification
- [ ] Add finer-grained ablation on loss/module design if feasible
- [ ] Strengthen qualitative visualization examples

### F. Figures and Tables

- [ ] Redraw Figure 1 to show the full pipeline clearly
- [ ] Explicitly explain subfigures `(a)`, `(b)`, `(c)` in both figure and caption
- [ ] Visualize FDAM placement within each encoding stage
- [ ] Improve the presentation of Table III ablation settings
- [ ] Replace or define `Base` explicitly in the table/footnote/text

### G. Dataset and Result Analysis

- [ ] Add WHU-CD sample size and main characteristics
- [ ] Balance the descriptions of all three datasets
- [ ] Revise the SYSU-CD discussion to explain recall/F1 gain vs slight precision drop
- [ ] Strengthen the interpretation of quantitative and qualitative results

### H. Language, Formatting, and References

- [ ] Correct spelling and grammar issues across the manuscript
- [ ] Replace ambiguous wording such as `original dimension` with precise wording
- [ ] Standardize formula punctuation to English punctuation
- [ ] Unify terminology and module naming across the paper
- [ ] Complete and normalize reference formatting
- [ ] Fill missing venue full names / volume / issue / pages where needed

### I. Response Package Preparation

- [ ] Draft a point-by-point response outline for the editor and all reviewers
- [ ] For each comment, record:
  issue -> action taken -> manuscript location -> evidence
- [ ] Mark unresolved items and required follow-up experiments

## Reviewer-to-Action Mapping

| ID | Source | Comment Summary | Manuscript/Code Area | Action Type | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| AE-1 | Associate Editor | Novelty not sufficiently established | Intro / Related Work | text + comparison table | High | Pending | |
| AE-2 | Associate Editor | Method clarity and reproducibility insufficient | Methods / Experiments / Code | text + code cross-check | High | Pending | |
| AE-3 | Associate Editor | Experimental support insufficient | Experiments | experiment | High | Pending | |
| AE-4 | Associate Editor | Concerns on feature alignment and frequency reconstruction soundness | Methods / Discussion / Experiments | text + experiment | High | Pending | |
| AE-5 | Associate Editor | Presentation issues | Full manuscript | language + formatting | Medium | Pending | |
| R1-1 | Reviewer 1 | WHU-CD description incomplete | Experiments | text | Medium | Pending | |
| R1-2 | Reviewer 1 | SYSU-CD precision/recall discussion needs explanation | Results analysis | text | Medium | Pending | |
| R1-3 | Reviewer 1 | `Base` in Table III is ambiguous | Table III / text | text | Medium | Pending | |
| R1-4 | Reviewer 1 | Haar wavelet choice needs justification | Methods / ablation | text or experiment | Medium | Pending | |
| R1-5 | Reviewer 1 | Local neighborhood size needs explanation/ablation | Methods / ablation | text + experiment | High | Pending | |
| R1-6 | Reviewer 1 | English and punctuation issues | Full manuscript | language | Medium | Pending | |
| R1-7 | Reviewer 1 | Reference format issues | References | reference cleanup | Medium | Pending | |
| R2-1 | Reviewer 2 | Novelty is unclear and seems compositional | Intro / Related Work | text + comparison table | High | Pending | |
| R2-2 | Reviewer 2 | Critical implementation details are missing | Methods / Experiments | text | High | Pending | |
| R2-3 | Reviewer 2 | Distinction from closest works is unclear | Related Work | text | High | Pending | |
| R2-4 | Reviewer 2 | Missing recent baselines and efficiency analysis | Experiments | experiment | High | Pending | |
| R2-5 | Reviewer 2 | Ablation is too limited | Experiments | experiment | High | Pending | |
| R2-6 | Reviewer 2 | Language quality needs revision | Full manuscript | language | Medium | Pending | |
| R3-1 | Reviewer 3 | Figure 1 does not clearly show the full method | Figure 1 / Methods | figure + text | High | Pending | |
| R3-2 | Reviewer 3 | Semantic vs non-semantic distinction is unclear | Methods / Discussion | text + analysis | High | Pending | |
| R3-3 | Reviewer 3 | Missing consistency constraints for reconstruction | Methods / Discussion / Experiments | text + possible method change | High | Pending | |
| R3-4 | Reviewer 3 | Wavelet Loss formulation missing | Loss section | text | Medium | Pending | Current manuscript has a basic formula but may need fuller derivation/explanation |
| R3-5 | Reviewer 3 | FAFM lacks direct supervision | Methods / Training / Experiments | text + possible method change | High | Pending | |
| R3-6 | Reviewer 3 | Ablation only on LEVIR-CD | Experiments | experiment | High | Pending | |
| R3-7 | Reviewer 3 | Comparison methods are outdated | Experiments | experiment | High | Pending | |
| R3-8 | Reviewer 3 | Spelling and grammar issues | Full manuscript | language | Medium | Pending | |

## Section-Level Location Mapping

### 中文版：审稿意见到论文/图表/代码位置映射

| ID | 对应论文位置 | 对应图表/公式 | 对应代码位置 | 建议修改动作 |
|---|---|---|---|---|
| AE-1 | Introduction，第 51-77 行附近，尤其是方法定位与贡献段 | 无 | 无直接代码依赖 | 重写新颖性定位，增加与最相关方法的针对性比较表 |
| AE-2 | Methods: Overview（81-86），FDAM（87-112），FAFM（113-204），Loss Function（238-252），Implementation Details（255-258） | Fig. 1（63），Fig. 2（107） | `models/ChangeFormer.py:2251` `class FFormer`；`1946` `class EncoderTransformer_v5`；`2134` `class DecoderTransformer_v5`；`256` `class OverlapPatchEmbed2`；`models/help_funcs.py:458` `class DWT`；`937` `class WaveletLoss` | 依据真实代码补全 backbone、stage 配置、DWT/IDWT、decoder、监督路径等细节 |
| AE-3 | Comparative Experiments（261-265），Ablation Studies（269-335） | Table `tab:comparison`（175），Table `tab:lambda_ablation`（275），Table `tab:ablation_clean`（305），Fig. 3（169） | 可能涉及训练脚本与统计脚本 | 增加近期 baseline、效率分析、更多数据集消融和更细粒度实验 |
| AE-4 | FDAM（87-112），FAFM（113-204），Loss Function（238-252），Conclusion（340） | Fig. 1（63），公式（151-162，244-248） | `models/help_funcs.py:700` `class FAFF`；`823` `class CAGM`；`937` `class WaveletLoss`；`models/trainer_ddp.py:404-405` | 增加概念合理性解释，必要时补 consistency/direct supervision 设计或实验 |
| AE-5 | 全文 | Fig. 1，参考文献区，公式标点 | 无 | 统一图示、语言、格式和参考文献 |
| R1-1 | Datasets and Implementation Details（256） | Table `tab:comparison`（175） | 无 | 补充 WHU-CD 样本数和主要特点 |
| R1-2 | Comparative Experiments（264） | Table `tab:comparison`（175） | 无 | 在 SYSU-CD 结果分析中解释 Precision 略降但 Recall/F1 提升 |
| R1-3 | Ablation Studies（320，335） | Table `tab:ablation_clean`（305） | `models/ChangeFormer.py:2251` `class FFormer` 可作为 Base 对应模型依据 | 在表注或正文中明确 `Base = plain Siamese Transformer without proposed modules` |
| R1-4 | FAFM（117） | 无 | `models/help_funcs.py:458` `class DWT` | 补充 Haar wavelet 选择依据，必要时补 wavelet basis 对比实验 |
| R1-5 | FAFM（147-157） | 公式（151-157） | `models/help_funcs.py:700` `class FAFF` | 说明 `Omega` 和 `3x3` 是否固定，并增加窗口大小消融 |
| R1-6 | Introduction（43），FDAM（95-101），Conclusion（340）等全文 | 全部公式 | 无 | 全文语言修订，修 typo，统一公式英文标点 |
| R1-7 | 参考文献区 | References | `ref.bib` | 补全会议全称、卷期页码，统一参考文献格式 |
| R2-1 | Introduction（56-77） | 无 | 无 | 强化“不是简单拼接”的论证，并区分与 Transformer-based CD 的差异 |
| R2-2 | Overview（83），FDAM（87-112），FAFM（113-204），CAGM（205-237），Implementation Details（258） | Fig. 1（63），Fig. 2（107） | `models/ChangeFormer.py:2251`，`1946`，`2134`，`256`；`models/help_funcs.py:458,700,823,937` | 逐项补充缺失实现细节并与代码保持一致 |
| R2-3 | Introduction（56-71） | 可新增相关工作对比表 | 无 | 明确指出现有 frequency-aware / CD 方法未解决的问题及 FACDNet 的补位点 |
| R2-4 | Comparative Experiments（261-265） | Table `tab:comparison`（175） | 训练与评估脚本 | 增加近期 SOTA baseline，并补参数量/FLOPs/速度 |
| R2-5 | Ablation Studies（269-335） | Table `tab:lambda_ablation`（275），Table `tab:ablation_clean`（305） | `models/help_funcs.py:700,823,937`；`models/trainer_ddp.py:404-405` | 从模块组合扩展到窗口、小波基、损失设计和跨数据集消融 |
| R2-6 | 全文 | 全文 | 无 | 全文英文与格式润色 |
| R3-1 | Overview（83-86） | Fig. 1（63） | `models/ChangeFormer.py:2251`，`1946`，`2134`；`models/help_funcs.py:700,823` | 重画 Figure 1，明确 `(a)(b)(c)` 含义及其在主流水线中的位置 |
| R3-2 | FDAM（88-103） | Fig. 2（107） | `models/help_funcs.py` 中 FDAM 对应实现若有；编码器插入逻辑在 `models/ChangeFormer.py:1946` 附近 | 解释 semantic / non-semantic 区分，补 FDAM 误对齐风险分析 |
| R3-3 | FAFM（147-162），Loss Function（239-248） | 公式（151-162，244-248） | `models/help_funcs.py:700` `class FAFF`；`937` `class WaveletLoss` | 回应 IDWT 重建后缺少 consistency constraint 的问题，必要时补方法或实验 |
| R3-4 | Loss Function（239-248） | 公式 `deqn_Lfreq`、`deqn_Ltotal` | `models/help_funcs.py:937` `class WaveletLoss` | 细化 Wavelet Loss 数学定义、输入输出及与训练流程的关系 |
| R3-5 | CAGM（205-237），Loss Function（238-252） | 无 | `models/help_funcs.py:700` `class FAFF`；`823` `class CAGM`；`models/trainer_ddp.py:404-405` | 解释 FAFM 通过最终预测反向传播获得监督，必要时加直接监督/辅助项 |
| R3-6 | Ablation Studies（269-335） | Table `tab:ablation_clean`（305） | 实验脚本 | 增加 SYSU-CD 或 WHU-CD 上的模块消融 |
| R3-7 | Comparative Experiments（261-265） | Table `tab:comparison`（175） | 实验脚本 | 更新比较方法列表，加入近期代表方法 |
| R3-8 | 全文 | 全文 | 无 | 全文拼写和语法清理 |

### English: Reviewer-to-Manuscript/Code Location Mapping

| ID | Manuscript Location | Figure/Table/Equation Anchor | Code Location | Planned Revision Action |
|---|---|---|---|---|
| AE-1 | Introduction, especially the method-positioning and contribution paragraphs around lines 51-77 | None | None | Rewrite the novelty positioning and add a focused comparison with the closest methods |
| AE-2 | Methods: Overview (81-86), FDAM (87-112), FAFM (113-204), Loss Function (238-252), Implementation Details (255-258) | Fig. 1 (63), Fig. 2 (107) | `models/ChangeFormer.py:2251`, `1946`, `2134`, `256`; `models/help_funcs.py:458`, `937` | Fill in backbone/stage settings, DWT/IDWT details, decoder structure, and supervision path based on the actual implementation |
| AE-3 | Comparative Experiments (261-265), Ablation Studies (269-335) | `tab:comparison` (175), `tab:lambda_ablation` (275), `tab:ablation_clean` (305), Fig. 3 (169) | training/evaluation scripts | Add stronger baselines, efficiency analysis, broader ablations, and stronger evidence |
| AE-4 | FDAM (87-112), FAFM (113-204), Loss Function (238-252), Conclusion (340) | Fig. 1 (63), Eqs. around 151-162 and 244-248 | `models/help_funcs.py:700`, `823`, `937`; `models/trainer_ddp.py:404-405` | Address conceptual soundness, and decide whether consistency/direct supervision changes are needed |
| AE-5 | Entire manuscript | Fig. 1, references, formula punctuation | None | Improve presentation, language, formatting, and references |
| R1-1 | Datasets and Implementation Details (256) | `tab:comparison` (175) | None | Add WHU-CD sample size and key characteristics |
| R1-2 | Comparative Experiments (264) | `tab:comparison` (175) | None | Explain the Recall/F1 gain with slightly lower Precision on SYSU-CD |
| R1-3 | Ablation Studies (320, 335) | `tab:ablation_clean` (305) | `models/ChangeFormer.py:2251` as the practical basis for the baseline model | Explicitly define `Base` in the table note or text |
| R1-4 | FAFM subsection (117) | None | `models/help_funcs.py:458` `class DWT` | Justify the choice of Haar wavelets and possibly add basis comparison |
| R1-5 | FAFM subsection (147-157) | Eqs. around 151-157 | `models/help_funcs.py:700` `class FAFF` | Clarify whether `Omega`/`3x3` is fixed and add window-size analysis |
| R1-6 | Entire manuscript | All formulas | None | Full language cleanup and formula punctuation unification |
| R1-7 | Reference section | References | `ref.bib` | Complete and normalize reference entries |
| R2-1 | Introduction (56-77) | None | None | Strengthen the argument that FACDNet is not a simple composition |
| R2-2 | Overview (83), FDAM (87-112), FAFM (113-204), CAGM (205-237), Implementation Details (258) | Fig. 1 (63), Fig. 2 (107) | `models/ChangeFormer.py:2251`, `1946`, `2134`, `256`; `models/help_funcs.py:458`, `700`, `823`, `937` | Add missing implementation details for reproducibility |
| R2-3 | Introduction (56-71) | a new related-work comparison table is recommended | None | Better articulate unresolved limits of prior methods and FACDNet's specific additions |
| R2-4 | Comparative Experiments (261-265) | `tab:comparison` (175) | training/evaluation scripts | Add recent SOTA baselines and efficiency metrics |
| R2-5 | Ablation Studies (269-335) | `tab:lambda_ablation` (275), `tab:ablation_clean` (305) | `models/help_funcs.py:700`, `823`, `937`; `models/trainer_ddp.py:404-405` | Expand ablations beyond module combinations and `lambda` |
| R2-6 | Entire manuscript | Entire manuscript | None | Language and formatting revision |
| R3-1 | Overview (83-86) | Fig. 1 (63) | `models/ChangeFormer.py:2251`, `1946`, `2134`; `models/help_funcs.py:700`, `823` | Redraw Figure 1 and clarify `(a)`, `(b)`, `(c)` within the full pipeline |
| R3-2 | FDAM subsection (88-103) | Fig. 2 (107) | encoder insertion logic around `models/ChangeFormer.py:1946` and related FDAM implementation | Clarify semantic vs non-semantic components and discuss FDAM misalignment risk |
| R3-3 | FAFM (147-162), Loss Function (239-248) | Eqs. around 151-162 and 244-248 | `models/help_funcs.py:700`, `937` | Address the missing consistency-constraint issue after IDWT reconstruction |
| R3-4 | Loss Function (239-248) | `deqn_Lfreq`, `deqn_Ltotal` | `models/help_funcs.py:937` | Provide a fuller Wavelet Loss formulation and training-path explanation |
| R3-5 | CAGM (205-237), Loss Function (238-252) | None | `models/help_funcs.py:700`, `823`; `models/trainer_ddp.py:404-405` | Explain indirect supervision of FAFM and consider direct auxiliary supervision |
| R3-6 | Ablation Studies (269-335) | `tab:ablation_clean` (305) | experiment scripts | Add ablations on at least one more dataset |
| R3-7 | Comparative Experiments (261-265) | `tab:comparison` (175) | experiment scripts | Update the comparison set with more recent methods |
| R3-8 | Entire manuscript | Entire manuscript | None | Proofread and correct grammar/spelling throughout |

## Numbered Review Comments

### Associate Editor

- `AE-1` The novelty is not sufficiently established beyond a combination of existing ideas.
- `AE-2` Methodological clarity and reproducibility are insufficient, including implementation details, mathematical formulation, key design choices, and architecture visualization.
- `AE-3` Experimental validation is insufficient, including outdated/incomplete baselines, limited ablation scope, and missing efficiency analysis.
- `AE-4` The conceptual soundness of feature alignment and frequency reconstruction is questioned, especially possible misalignment of true semantic changes and lack of explicit consistency/direct supervision.
- `AE-5` Presentation quality needs improvement, including figure clarity, language, formatting, and references.

### Reviewer 1

- `R1-1` Add WHU-CD sample size or key dataset characteristics.
- `R1-2` Explain why SYSU-CD shows improved F1/Recall with slightly lower Precision.
- `R1-3` Clarify the meaning of `Base` in Table III.
- `R1-4` Justify the use of the Haar wavelet, possibly with explanation or citation.
- `R1-5` Explain or analyze the local neighborhood size `Omega` such as `3x3`.
- `R1-6` Fix spelling/grammar issues and standardize formula punctuation.
- `R1-7` Correct incomplete or inconsistent reference formatting.

### Reviewer 2

- `R2-1` Better justify the novelty of the module combination and distinguish it from prior Transformer-based CD work.
- `R2-2` Provide missing implementation details for reproducibility.
- `R2-3` Better articulate differences from the most closely related methods and unresolved limitations of prior work.
- `R2-4` Add stronger recent baselines and efficiency analysis such as parameter count and FLOPs.
- `R2-5` Expand ablation studies beyond module combinations and `lambda`.
- `R2-6` Revise the manuscript for English quality and formatting.

### Reviewer 3

- `R3-1` Redesign Figure 1 to clearly present the full pipeline and the placement of submodules.
- `R3-2` Explain the distinction between semantic and non-semantic components and discuss possible FDAM misalignment of real changes.
- `R3-3` Address the missing consistency constraints in frequency-domain reconstruction.
- `R3-4` Provide a clearer mathematical formulation of the Wavelet Loss.
- `R3-5` Explain or improve the supervision of FAFM, whose output is only indirectly optimized.
- `R3-6` Extend ablation studies beyond LEVIR-CD.
- `R3-7` Compare against newer state-of-the-art methods.
- `R3-8` Fix remaining spelling and grammar issues.

## Planned Work Order

1. Build the numbered reviewer-comment mapping.
2. Strengthen novelty positioning and related-work comparison.
3. Fill method/reproducibility gaps by cross-checking manuscript text with actual code.
4. Decide whether model changes are needed for Reviewer 3 concerns.
5. Plan and run additional experiments:
   stronger baselines, efficiency, broader ablations.
6. Update figures/tables and rewrite result analysis.
7. Perform language/reference cleanup.
8. Draft the point-by-point response letter.

## Change Log

### 2026-04-24

- Created this execution log file.
- Added the initial checklist, reviewer-to-action mapping, and planned work order.
- Established this file as the place for future revision changes and progress records.
- Numbered all editor/reviewer comments into stable IDs (`AE-*`, `R1-*`, `R2-*`, `R3-*`).
- Updated the revision status from planning to review triage in progress.
- Updated all tracked file paths after moving the manuscript, review comments, and execution log into the `ChangeFormer` directory for git synchronization.
- Added a section-level mapping table that links each review comment to manuscript sections, figure/table/equation anchors, and relevant code locations.

## Detailed Modification Record

Use this section to append concrete edits as they happen.

### Template

- Date:
- Target file:
- Related reviewer IDs:
- Modification summary:
- Evidence or result:
- Follow-up needed:

## Progress Updates

### Overall Progress Snapshot

- Review triage: section-level mapping completed
- Novelty/related work revision: not started
- Method detail revision: not started
- Figure/table revision: not started
- Additional experiments: not started
- Language/reference cleanup: not started
- Response letter drafting: not started

### Ongoing Notes

- Stable reviewer IDs are now defined and can be used to track all future manuscript/code/experiment changes.
- The manuscript, review comments, and execution log are now co-located under `ChangeFormer` to simplify version control and synchronization.
- The section-level mapping from review comments to manuscript/code locations is now in place.
- The next step is to turn high-priority mapped items into concrete edit tasks, starting from novelty positioning and method reproducibility.

## Decision Log

Record plan changes here so future revisions remain traceable.

### Template

- Date:
- Previous plan:
- New decision:
- Reason:
- Impact on manuscript/code/experiments:

## Task Package: AE-1 / R2-1 / R2-3

### 中文：新颖性定位与 Related Work 重写计划

#### 目标

针对以下三条高优先级意见，重构引言后半段与相关工作定位逻辑：

- `AE-1` 方法新颖性不足，尚未清楚证明其超出已有思想组合。
- `R2-1` 需要更清楚地证明三模块组合的创新性，并与已有 Transformer-based CD 方法区分开。
- `R2-3` 需要更明确地说明与最相关工作的差异、现有方法未解决的问题以及 FACDNet 为什么更有效。

#### 核心改写目标

- 明确论文要解决的不是泛化的 change detection 问题，而是“高分辨率双时相场景中的非语义风格差异与真实变化混杂”的问题。
- 明确 FACDNet 的创新点不在于单个模块本身绝对全新，而在于三者围绕同一个问题形成了有因果顺序的闭环设计：
  `FDAM -> FAFM -> CAGM`
- 明确与纯 Transformer 方法、纯频域方法、普通特征融合方法之间的边界。
- 明确 prior work 的缺口：已有方法通常只解决全局建模、频域增强或融合中的一个或两个方面，但没有显式处理“先风格对齐、再频域净化、再跨域验证”的联合流程。

#### 具体执行步骤

1. 梳理最相关文献分组
- Transformer-based change detection
- frequency-aware / wavelet-based change detection
- style alignment / domain-robust representation methods
- change-guided fusion or attention-based fusion methods

2. 提炼每组方法的“能力边界”
- 这些方法分别解决了什么问题
- 它们没有解决什么问题
- 它们为什么不足以直接解决本文场景中的 pseudo-change 干扰

3. 重写 Introduction 中的问题动机链条
- 从“phenological pseudo-changes + blurred boundaries”出发
- 指出仅靠空间域建模不足
- 指出仅靠频域增强也不足
- 引出需要一个分阶段的联合建模框架

4. 重写 FACDNet 的定位段
- 先说问题链条
- 再说模块顺序
- 最后说设计逻辑，而不是先平铺三个模块名称

5. 重写贡献列表
- 避免把贡献写成三个模块罗列
- 改成“问题定义 + 方法闭环 + 实验证据”的结构
- 每条贡献尽量体现 why this matters，而不是 only what we add

6. 增加一个 focused comparison table
- 表头建议：
  Method / Global context modeling / Bi-temporal alignment / Explicit wavelet decomposition / Adaptive frequency filtering / Cross-domain guided fusion / Robustness to style discrepancy
- 目标不是堆方法，而是突出 FACDNet 的组合逻辑和缺口填补

7. 在 related work 中加入“closest methods”对照段
- 至少单独讨论 2 到 4 个最接近的方法
- 每个方法写清：
  它做了什么
  它没做什么
  FACDNet 和它的本质差异是什么

8. 在结尾或方法概述前补一句总结定位
- FACDNet is not a direct stacking of alignment, frequency filtering, and attention fusion modules; it is a staged design tailored to disentangle style discrepancy from semantic change before multi-scale consistency verification.

#### 计划改动位置

- Introduction：约第 56-77 行
- Contributions：约第 75-77 行附近
- 如版面允许，插入一个新的 related-work comparison table
- 必要时在 Methods Overview（83-86）补一句与前人差异的承接句

#### 预期产出

- 一版新的引言后半段
- 一版新的 contribution bullets
- 一张 related-work focused comparison table
- 一段 closest-method comparison paragraph

#### 风险与注意事项

- 不要把“组合式创新”硬写成“全新模块发明”，否则容易再次被质疑。
- 重点应放在 problem-solution fit，而不是夸大模块原创性。
- 需要避免 related work 只做罗列，必须体现 targeted comparison。

### English: Rewrite Plan for AE-1 / R2-1 / R2-3

#### Objective

Rewrite the novelty-positioning and related-work logic to directly address:

- `AE-1`: insufficient novelty justification
- `R2-1`: unclear novelty beyond a compositional combination
- `R2-3`: unclear differences from the closest methods and unresolved gaps in prior work

#### Main Goals

- Reframe the paper around the specific problem of disentangling non-semantic style discrepancy from true semantic change in high-resolution bi-temporal remote sensing.
- Present FACDNet as a staged causal design rather than a flat three-module combination:
  `FDAM -> FAFM -> CAGM`
- Clarify the boundaries between FACDNet and:
  Transformer-only CD methods,
  frequency-aware CD methods,
  and generic attention/fusion-based methods.
- Make the unresolved gap explicit:
  prior methods do not jointly perform style alignment, frequency-domain purification, and cross-domain verification in a unified staged pipeline.

#### Execution Steps

1. Re-group the closest prior work into targeted categories.
2. Extract the strengths and limitations of each category.
3. Rewrite the motivation chain in the Introduction.
4. Rewrite the FACDNet positioning paragraph around the staged logic.
5. Rewrite the contribution list to emphasize problem-solution fit and evidence.
6. Add a focused related-work comparison table.
7. Add a dedicated closest-method comparison paragraph.
8. Add one concise positioning sentence near the end of the Introduction or beginning of Methods.

#### Planned Edit Locations

- Introduction: around lines 56-77
- Contribution bullets: around lines 75-77
- Possibly add a new related-work comparison table
- Optionally add a short transition sentence in Methods Overview

#### Expected Outputs

- Revised second half of the Introduction
- Revised contribution bullets
- One focused comparison table
- One closest-method comparison paragraph

#### Cautions

- Do not overclaim module-level originality.
- Emphasize problem-solution fit and staged design logic.
- Avoid a survey-like related-work rewrite; keep it targeted and comparative.
