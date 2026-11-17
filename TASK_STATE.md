# E4Probe 延续实验状态

更新：2026-11-13（Asia/Shanghai）。

## 用户目标

在本目录的 E3 / E4 边界监督基础上，延续 `../MT+DuCiSC+BGCE/MT-DuCiSC` 第一版的通道评分与边界特征迁移思路。用户已授权继续实现和实验，不需要再次确认常规代码修改与验证。

## 已确认事实

- `MT_DS_PL_C1`：四尺度 MT_DS_PL + 单层 E3 或 E4 边界 probe，12 次训练全部完成，原始源码、权重、日志与 Excel 保持原样。
- E4 相对 V0 的 Dice：LA +0.2749、Pancreas −0.1616 个百分点；E3：−0.3004、+0.0488。未通过原有辅助边界监督的收益判据。
- 当前 C1 不包含 CBCD、BDD、供体匹配或特征迁移。辅助损失结果不能直接替代完整迁移假设的检验。
- 第一版包含上述机制，但其单输出、无伪标签 warmup 的训练流程与当前框架不同。移植时保留当前四尺度监督和前 2000 步 warmup。
- 沙箱内 `torch.cuda.is_available() == False`，`/dev` 下无 NVIDIA 设备；随后通过沙箱外执行确认 SSL29 可使用 7 张 A100。用户已指定 SSL29 并授权按空闲 GPU 运行，不需要再询问环境或启动许可。

## C2：先验证通道排序

- 目录：`C2_CBCD/`，协议在 `C2_CBCD/README.md`，在正式分析启动前写好。
- 12 个现有配套 final Student / probe checkpoint，仅分析有标签训练病例；eval、冻结参数，仅 detached 特征求梯度。
- 每病例 2 个固定完整尺寸 patch，CBCD Top / Bottom 16 通道遮挡，对照 8 次随机遮挡；保留所有跳过记录；按病例汇总和 bootstrap。
- `tests/verify_analysis.py` 已在 SSL29 / CPU 通过：原编码器特征一致、公式和遮挡独立核对、BN / 模式 / 参数梯度 / RNG 无干扰、本地 crop 确定性。
- 正式离线分析运行命令：

```bash
.conda/envs/SSL29/bin/python -u Exp_E4Probe/C2_CBCD/scripts/analyze.py \
  --device cpu --cpu_threads 4 --output_dir Exp_E4Probe/C2_CBCD/Results_offline
```

- **已完成**：12/12 checkpoint，168/168 有效 patch，16128 条通道评分，耗时 86.1 秒。E4 和 E3 在两个数据集均通过前置敏感性判据。`C2_CBCD/Results_offline/report.md` / `summary.json` 为完整结果；SHA256 记录输入 checkpoint、配置、训练数据和相关源码。这里只证明 probe 的通道敏感性，不证明迁移或分割收益。

| 层级 | 数据集 | Top−Random 边界损失增幅 | 病例 bootstrap 95% |
|---|---|---:|---|
| E4 | LA | 1.32078 | [1.11412, 1.49657] |
| E4 | Pancreas | 0.87606 | [0.68214, 1.07542] |
| E3 | LA | 5.26326 | [4.96471, 5.54045] |
| E3 | Pancreas | 8.26754 | [7.82550, 8.77740] |

## C3：最小迁移实现与正式训练

- 目录 `MT_DS_PL_C3/`，源自 C1 的源码副本，旧实验未修改；`BASE_SOURCE.json` 保留 C1 源码校验值。
- 已实现四尺度网络 encode / decode 接口、全局边界统计迁移、独立随机供体 generator 与 checkpoint 状态、每次训练源码快照与 SHA256。
- 实验首轮 E4 × LA / Pancreas × seed 42 / 500 / 1337 × sham / cbcd / random / bottom = **24 次，从头训练**，每次 15000 步。`none` / 零强度保留 C1 原路径，sham 与启用迁移组有相同前向 / Dropout 顺序但不注入残差。
- 超参数与预设收益判据：见 `MT_DS_PL_C3/README.md`。BDD 与匹配留待单独消融。
- CPU 验收全部通过，记录 `MT_DS_PL_C3/tests/acceptance.txt`：初始化、四尺度前向和梯度、C1 none / 零强度轨迹逐位一致、旧版迁移数值、梯度方向隔离、随机路由状态恢复、LA / Pancreas CLI 训练与 checkpoint 重载、反转隐藏无标签 GT 不影响任何训练状态。
- 完整尺寸 GPU smoke：LA 和 Pancreas 各 6 步，提前到第 3 步启用迁移，仅作实现检查；显存峰值 5729.49 / 5082.31 MiB，迁移残差非零、损失和梯度有限。输出在 `MT_DS_PL_C3/tests/outputs/gpu_smoke_*_cbcd`，不得计作正式结果。
- 正式队列于 **2026-11-13 17:09:37** 启动（Asia/Shanghai），SSL29，调度器 PID **1348557**；至少 8000 MiB 空闲显存，单卡最多一个本阶段进程，自动排队；训练失败停止启动后续 run。
- 调度控制：`MT_DS_PL_C3/queue_control.json`；日志：`MT_DS_PL_C3/queue_20261113_170937.log`。
- 正式 manifest：`MT_DS_PL_C3/Results_transfer/launch_logs/20261113_170940_047368/manifest.json`。它是运行状态的主要依据，记录 24 次任务的 pending/running/finished/failed、GPU、PID 与退出码。
- 首组 sham / LA / seed42 已在 GPU5 开始训练。后续分配以实时 manifest 为准。
- `finalize_c3.py` 监控队列，完成后自动运行固定判据的 `scripts/report_transfer.py`；监控状态在 `MT_DS_PL_C3/queue_status.json`，最终报告为 `MT_DS_PL_C3/summary/transfer_report.md` / `.json`。训练未完成时该报告只列待完成项。
- **源码已冻结**：调度器会检查哈希，修改 C3 `code/`、`scripts/`、README 或基础清单会停止后续排队任务。新增独立分析文件和更新本状态文档不影响队列。

## 最近一次运行快照

2026-11-13 17:13:09（Asia/Shanghai）：5 个 run 已启动，19 个排队，0 个失败。

| 模式 | 数据集 | seed | GPU | PID |
|---|---|---:|---:|---:|
| sham | LA | 42 | 5 | 1349013 |
| cbcd | LA | 42 | 2 | 1353913 |
| random | LA | 42 | 3 | 1359197 |
| bottom | LA | 42 | 0 | 1364179 |
| sham | Pancreas | 42 | 4 | 1370601 |

调度器会按 GPU 显存情况继续排队；此静态快照不替代实时 manifest。完成监控 PID 1369999，控制文件 `MT_DS_PL_C3/finalizer_control.json`。

## 2026-11-16 续跑状态

- 首轮调度器已因 `cbcd / Pancreas / seed42` 在第 420 步遇到共享 GPU 显存 OOM 而停止派发。核对训练 CSV、测试指标和日志后，实际确认 5/24 完整、1/24 中断、18/24 未启动；原 manifest 中 4 个 `running` 状态是过期记录。
- 失败任务的最新 checkpoint 在第 1 步。续跑前已把超过 checkpoint 的部分 CSV 备份到该输出目录的 `pre_resume_*` 子目录，再从 checkpoint 恢复，避免重复统计未保存状态的训练步。
- 续跑脚本：`continue_c3_remaining.py`（位于 `Exp_E4Probe/`，不改动 C3 冻结源码）。父进程 PID **2233114**；最多并行 2 组、要求启动时至少 9000 MiB 空闲显存，遇到 OOM 最多从 checkpoint 重试；完成后自动运行固定判据报告。
- 新 manifest：`MT_DS_PL_C3/Results_transfer/launch_logs/continuation_20261116_153822_536396/manifest.json`；实时状态：`MT_DS_PL_C3/queue_status.json`。
- 最近核实：5 组完成、2 组运行、17 组排队、0 组失败。`cbcd / Pancreas / seed42` 从第 1 步恢复，运行于 GPU3；`random / Pancreas / seed42` 首次运行于 GPU5。

## 2026-11-17 当前检查

- 上述续跑父进程后来退出，manifest 在 12 组完成时停止更新；两项标为 `running` 的 Pancreas / seed500 任务实际已完成 15000 步并写出测试指标。结合每个输出目录的训练 CSV、`best_metrics.json`、`test_metrics.json` 与训练日志，当前实际完成 **14/24**，尚未启动 **10/24**，失败 **0**。
- 已完成任务：seed42 的四种 LA 和四种 Pancreas 设置；seed500 的四种 LA 设置，以及 Pancreas 的 sham、cbcd。待运行为 seed500 的 Pancreas random / bottom，以及 seed1337 的 8 组。
- 14 组阶段性 Dice：LA（seed42、500 两 seed 均值）sham 0.89408、cbcd 0.89788、random 0.89101、bottom 0.89762；Pancreas sham 0.80631、cbcd 0.80646（均为 seed42、500），random 0.80691、bottom 0.80776（目前只有 seed42）。这些尚非最终三 seed 判定；Pancreas 上 CBCD 暂未超过 Random / Bottom 的 seed42 结果。
- 已重启脱离会话调度器，父进程 PID **3127573**。它从磁盘指标确认并跳过 14 组已完成任务，仅排剩余 10 组。新 manifest：`MT_DS_PL_C3/Results_transfer/launch_logs/continuation_20261117_172306_985770/manifest.json`；启动日志：`MT_DS_PL_C3/continue_c3_20261117.log`。
- 当前队列状态：14 完成、10 排队、0 运行、0 失败。检查时 7 张 A100 的空闲显存均低于 9000 MiB，调度器正在等待；空闲显存达到阈值后每次最多启动 2 组。完成矩阵后会自动生成固定判据报告。
