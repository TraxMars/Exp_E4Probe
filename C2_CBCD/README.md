# C2：在 C1 E3 / E4 上验证 CBCD 通道排序

2026-11-13。承接 `../MT_DS_PL_C1`，本阶段先回答：**现有边界 probe 的通道敏感性排序是否能通过遮挡实验验证？** 这是迁移实验的前置诊断，不把 probe 的边界损失敏感性等同于分割收益。

## 固定方案（运行前制定）

- 使用 C1 的 12 个 `last_checkpoint.pth`：E3 / E4 × LA_8l / Pancreas_6l × seed 42 / 500 / 1337，Student 与 probe 来自同一步。禁止将 best Student 与 last probe 混用。
- 模型与 probe 均为 eval，参数冻结；仅对 detached 的编码器特征求梯度。读取 checkpoint 记录的有标签训练病例，不读取无标签 GT、验证集或测试集。
- 每个有标签病例固定采样 2 个完整训练尺寸 patch，随机种子 20261113；同病例的 patch 在层级和模型 seed 之间相同。增广、padding 和均匀 crop 与 C1 的规则相同；本分析不是训练轨迹重放。
- 分析单位为单 patch（batch=1），边界目标半径 1，平衡 BCE + soft Dice。CBCD = spatial_mean(abs(feature * dL_boundary/dfeature))。
- 固定遮挡 16 个通道；Top、Bottom 与 8 次等预算随机遮挡比较。E3 / E4 的通道数不同，因此 16 个通道对应的比例不同。
- 空边界 patch、全零/完全相同的通道评分不参与排序判断，保留跳过记录，不额外重采样。
- 每个 run 输出所有通道评分、每 patch 的遮挡损失差和随机标准差。随机遮挡用独立 CPU generator。
- 判据：同层级在两个数据集上均满足，才建议进入该层级的迁移试验：① 3 个 seed 的平均 Top−Random 与 Top−Bottom 均 > 0；② 先在病例内平均 crop 和 seed，再按病例 bootstrap 5000 次，两项差值的 95% 区间下界均 > 0；③ 所有有标签病例至少有一个有效 patch。此为小样本诊断判据，不是测试集显著性或分割收益判据。
- 若不通过，记录具体失败环节，不据此自动调 λ、k 或选择更有利的 crop。

## 运行

```bash
cd /aita_data/home/aita_duanxuelong
.conda/envs/SSL29/bin/python -u Exp_E4Probe/C2_CBCD/scripts/analyze.py \
  --device cpu --cpu_threads 4 --output_dir Exp_E4Probe/C2_CBCD/Results_offline
```

默认分析全部 12 个 checkpoint。可用 `--levels e4`、`--datasets LA`、`--seeds 42` 缩小范围做冒烟检查；不完整组合不会给出正式通过结论。结果目录必须不存在，避免覆盖。`--resume` 仅恢复同一参数、同一源码、同一输入文件的未完成分析。

输出 `config.json`（输入文件 SHA256 与源码 SHA256）、`patches.json`（病例、crop、增广）、每 run 的 `channel_scores.csv` / `occlusion.csv` / `summary.json`，以及总表 `summary.json` / `report.md`。原始 C1 结果保持只读。

## 后续承接

通过本诊断后，优先将第一版的最小全局边界统计迁移接入四尺度 MT_DS_PL，E4 优先、E3 作为层级对照；分别比较 V0、C1、Top / Random / Bottom donor。保持现有 warmup、监督损失和评估协议。BDD 接收选择与余弦匹配作为后续独立消融，不一次叠加。

当前环境 CUDA 不可用；CPU 诊断可以进行，正式 GPU 训练需待运行环境具备 GPU。现有 LA / Pancreas 使用 test.list 同时选模和评估；本阶段不使用这些数据，不改变已有结果的统计口径。
