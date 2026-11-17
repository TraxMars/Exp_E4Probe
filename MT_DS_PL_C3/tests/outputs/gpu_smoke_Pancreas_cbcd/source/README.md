# C3：四尺度 MT_DS_PL 上的最小边界统计迁移

承接 `../C2_CBCD/Results_offline/report.md`：12 个配套 C1 checkpoint、168 个有标签 patch 的 Top / Bottom / Random 遮挡诊断完成，E3 与 E4 均通过预先规定的通道敏感性判据。这不等于分割收益，C3 单独检验迁移。

## 固定实验方案

- 首轮只用 E4，E3 留作后续层级对照。不因 E3 遮挡损失更大就认定其更适合迁移（两层的固定 16 通道比例也不同）。
- 沿用 C1 的四尺度 CE+Dice、前 2000 步伪标签 warmup、之后 0.3 伪标签损失、0.03 边界损失、15000 步、SGD 0.01、EMA 0.99、2 labeled + 2 unlabeled；从头训练，不从挑出的 C1 checkpoint 微调。
- 全局迁移沿用第一版：有标签 GT 边界膨胀半径 2 后池化；CBCD Top16 / Bottom16 / Random16 供体通道；供体 token detach；MLP 生成通道残差，sigmoid 门控；仅作用于无标签 Teacher 伪边界带 × 归一化预测熵。Student 编码完后仅修改 E4 skip，不重算更深编码器。
- 从第 2001 个更新开始，迁移强度经 2000 步线性升至 0.10。Teacher 全分辨率输出生成迁移 mask；四个尺度的伪标签损失仍分别用该尺度 Teacher argmax。
- 不使用 BDD 或供体匹配，不增加其他损失。无标签 GT 仍仅在原有 logging-only oracle diagnostics 中使用。
- 随机供体使用独立 CPU generator，checkpoint 保存其状态；新增模块独立初始化，不消耗基线采样/dropout RNG。空供体边界、下采样后空边界和空迁移 mask 跳过迁移；若 batch 只有部分有效供体，token 只对有效供体平均。这是相对旧代码的明确处理差异。

首轮 {sham, cbcd, random, bottom} × {LA_8l, Pancreas_6l} × {42,500,1337}，共 24 次。已有 V0 / C1 E4 为历史对照。

`none` 完整保留 C1 路径；零强度和 warmup 也使用 C1 原前向顺序。启用迁移后需要先获得 Teacher mask，Student 的无标签编码、Teacher、Student 解码的顺序与 C1 不同，因此额外设置 `sham`：与迁移组采用完全相同的前向/Dropout 顺序，但不注入残差。对照迁移机制时以 sham 为直接对照，避免把随机前向顺序变化归因于迁移。

首轮判据在正式训练前固定：cbcd 在**两个数据集**的三 seed nms0 Dice 均值，相对 sham 与已有 C1 E4 都提高 >0.4 个百分点；病例先平均 seed 后 bootstrap 95% 区间下界 >0；cbcd 三 seed 均值分别高于 random 和 bottom，且最后 5 次验证均值不低于 sham。未满足则记为尚未支持，不自动扫强度或 k。沿用 test.list 同时选模和评估的现有协议，故仅作内部配对比较，正式泛化结论需独立验证集。

## 使用

```bash
cd /aita_data/home/aita_duanxuelong
PY=.conda/envs/SSL29/bin/python
$PY Exp_E4Probe/MT_DS_PL_C3/scripts/launch_transfer.py --dry_run \
  --plan Exp_E4Probe/MT_DS_PL_C3/launch_plan.json
# GPU 环境中执行完整 24 次矩阵；按空闲显存排队，每卡默认同时一个进程。
$PY Exp_E4Probe/MT_DS_PL_C3/scripts/launch_transfer.py --gpus 0 1
```

单次：

```bash
CUDA_VISIBLE_DEVICES=0 $PY Exp_E4Probe/MT_DS_PL_C3/code/train.py \
  --dataset LA --probe_level e4 --transfer_mode cbcd --seed 42 \
  --output_dir Exp_E4Probe/MT_DS_PL_C3/Results_transfer/cbcd_e4/seed_42/LA_8l
```

本目录源码从 C1 源码复制并记录 `BASE_SOURCE.json`，不复制旧结果。每次训练保存源码 SHA256 和源码快照；断点恢复只接受相同 C3 源码，保存 Student、Teacher、probe、transfer、optimizer 与 RNG。DataLoader workers 的状态与 C1 一样不保证逐位恢复。推理仍为普通 VNetDS，`best_model.pth` 仅含 Student；配套 probe / transfer 保存在训练 checkpoint。

完成训练后运行 `scripts/report_transfer.py`，自动按上述固定判据比较。缺少 run 时仅列出待完成项，不给出通过结论。

CPU 小尺寸验证已通过，仅用于实现验收，不报告为新分割结果。沙箱内不可见 GPU，沙箱外 SSL29 可见 7 张 A100；启动时按实际空闲显存选择，每卡默认最多一个本阶段进程，后台调度器记录 PID、GPU 与退出状态，失败即停止启动后续 run。正式运行状态见上层 `TASK_STATE.md`。
