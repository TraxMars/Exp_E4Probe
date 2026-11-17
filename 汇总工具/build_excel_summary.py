#!/usr/bin/env python3
"""从 C1 E3/E4 和配对 V0 的原始结果生成 Excel；依赖 XlsxWriter。

运行：PYTHONPATH=/tmp/codex_e4probe_excel_deps python Exp_E4Probe/汇总工具/build_excel_summary.py
"""
import csv
import hashlib
import io
import json
import math
import re
import statistics as st
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import xlsxwriter


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'Exp_E4Probe'
PROJECT = EXP / 'MT_DS_PL_C1'
NOW = datetime.now(ZoneInfo('Asia/Shanghai'))
OUTPUT = EXP / '实验结果汇总_2026-11-12.xlsx'
SEEDS = (42, 500, 1337)
DATASETS = ('LA_8l', 'Pancreas_6l')
METHODS = ('V0', 'E4', 'E3')
ROOTS = {'V0': ROOT / 'Baseline/MT_DS_DICE/Results_baseline_pl',
         'E4': PROJECT / 'Results_C1_E4', 'E3': PROJECT / 'Results_C1_E3'}
METRICS = ('dice', 'jc', 'hd95', 'asd')
COLORS = {'V0': '#718096', 'E4': '#008C95', 'E3': '#E8873A'}
SOURCES = []


def read(path, method='', dataset='', seed=''):
    content = path.read_bytes()
    SOURCES.append([method, dataset, seed, path.name, len(content),
                    hashlib.sha256(content).hexdigest(), str(path.relative_to(ROOT))])
    return content.decode('utf-8')


def rows_by_iter(text, kind):
    rows = (list(csv.DictReader(io.StringIO(text))) if kind == 'csv'
            else [json.loads(line) for line in text.splitlines() if line.strip()])
    unique = {int(row['iteration']): row for row in rows}
    return [unique[i] for i in sorted(unique)]


def finite(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
    return [v for v in values if math.isfinite(v)]


def median(rows, key):
    values = finite(rows, key)
    return st.median(values) if values else None


def ms(values):
    return st.mean(values), st.stdev(values)


def display(values, decimals=2):
    mean, sd = ms(values)
    return f'{mean:.{decimals}f} ± {sd:.{decimals}f}'


def case_map(run, mode):
    data = run['test'] if mode == 0 else run['test']['nms1']
    return {case['case']: case for case in data['per_case']}


def metric(run, mode, key):
    data = run['test'] if mode == 0 else run['test']['nms1']
    return data[key] * (100 if key in ('dice', 'jc') else 1)


RUNS = {}
MANIFESTS = {}
for method, root in ROOTS.items():
    for path in sorted(root.glob('launch_logs/*/manifest.json')):
        for item in json.loads(read(path, method)):
            MANIFESTS[(method, item['dataset'], item['seed'])] = item
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / f'seed_{seed}' / dataset
            raw = {name: read(path / name, method, dataset, seed)
                   for name in ('config.json', 'test_metrics.json', 'best_metrics.json',
                                'validation_metrics.jsonl', 'training_metrics.csv',
                                'diagnostics.csv', 'log.txt')}
            run = {'method': method, 'dataset': dataset, 'seed': seed, 'path': path,
                   'config': json.loads(raw['config.json']),
                   'test': json.loads(raw['test_metrics.json']),
                   'best': json.loads(raw['best_metrics.json']),
                   'val': rows_by_iter(raw['validation_metrics.jsonl'], 'jsonl'),
                   'train': rows_by_iter(raw['training_metrics.csv'], 'csv'),
                   'diag': rows_by_iter(raw['diagnostics.csv'], 'csv'),
                   'boundary': []}
            if method != 'V0':
                run['boundary'] = rows_by_iter(read(path / 'boundary_metrics.csv', method, dataset, seed), 'csv')
            args = run['config']['args']
            run['last5'] = st.mean(row['dice'] * 100 for row in run['val'][-5:])
            run['last'] = run['val'][-1]['dice'] * 100
            run['best_val'] = max(run['val'], key=lambda row: row['dice'])
            run['hours'] = float(run['train'][-1]['elapsed_seconds']) / 3600
            stamps = re.findall(r'(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', raw['log.txt'])
            run['start'] = datetime.strptime(stamps[0], '%Y-%m-%d %H:%M:%S,%f')
            run['end'] = datetime.strptime(stamps[-1], '%Y-%m-%d %H:%M:%S,%f')
            run['finished'] = int(run['train'][-1]['iteration']) == args['max_iterations'] and 'finished at iteration' in raw['log.txt']
            assert run['finished'], f'Incomplete run: {path}'
            assert len(run['val']) == 72
            assert run['best_val']['iteration'] == run['best']['iteration']
            assert math.isclose(run['best']['dice'], run['test']['dice'], abs_tol=1e-12)
            for mode in (0, 1):
                data = run['test'] if mode == 0 else run['test']['nms1']
                assert len(data['per_case']) == data['cases'] == 20
                assert len(case_map(run, mode)) == 20
                for key in METRICS:
                    values = [case[key] for case in data['per_case']]
                    assert math.isclose(st.mean(values), data[key], abs_tol=1e-12)
                    assert math.isclose(st.pstdev(values), data[f'{key}_std'], abs_tol=1e-12)
            RUNS[method, dataset, seed] = run

read(PROJECT / 'README.md', '文档')
read(PROJECT / '实验日志/2026-11-12.md', '文档')


def group(method, dataset):
    return [RUNS[method, dataset, s] for s in SEEDS]


def deltas(method, dataset, mode, key):
    return [metric(RUNS[method, dataset, s], mode, key) - metric(RUNS['V0', dataset, s], mode, key) for s in SEEDS]


def last5_delta(method, dataset):
    return [RUNS[method, dataset, s]['last5'] - RUNS['V0', dataset, s]['last5'] for s in SEEDS]


def case_differences(method, dataset, mode, seed):
    a, b = case_map(RUNS[method, dataset, seed], mode), case_map(RUNS['V0', dataset, seed], mode)
    assert a.keys() == b.keys()
    return {case: (a[case]['dice'] - b[case]['dice']) * 100 for case in a}


def interval(values):
    # 所有主病例区间 n=20，双侧 95% Student t，df=19。
    assert len(values) == 20
    mean, sd = ms(values)
    half = 2.093024054408263 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


CASES = {}
for method in ('E4', 'E3'):
    for dataset in DATASETS:
        for mode in (0, 1):
            by_seed = [case_differences(method, dataset, mode, s) for s in SEEDS]
            avg = {case: st.mean(values[case] for values in by_seed) for case in by_seed[0]}
            CASES[method, dataset, mode] = {'values': avg, 'ci': interval(list(avg.values()))}


wb = xlsxwriter.Workbook(str(OUTPUT))
wb.set_properties({'title': 'C1 编码器边界 Probe 实验结果汇总',
                   'subject': 'E4 / E3 vs MT_DS_PL V0, 3 seeds',
                   'author': '实验结果汇总', 'comments': '由原始 JSON/CSV 重新计算；原始实验文件未改动。'})
BASE = {'font_name': 'Microsoft YaHei', 'font_size': 10, 'valign': 'vcenter'}
fmt = {}
for name, props in {
    'title': {'font_size': 19, 'bold': True, 'font_color': '#153F54'},
    'note': {'font_color': '#536776', 'text_wrap': True},
    'head': {'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#153F54', 'text_wrap': True, 'border': 0},
    'text': {}, 'num': {'num_format': '0.00'}, 'int': {'num_format': '0'},
    'small': {'num_format': '0.0000'}, 'delta': {'num_format': '+0.00;-0.00;0.00'},
    'section': {'bold': True, 'font_size': 12, 'font_color': '#153F54', 'bg_color': '#E8F2F5'},
    'good': {'font_color': '#16754A', 'bg_color': '#E6F3EC'},
    'bad': {'font_color': '#A23D3D', 'bg_color': '#FAE9E7'},
    'link': {'font_color': '#1674A5', 'underline': 1},
    'date': {'num_format': 'yyyy-mm-dd hh:mm:ss'},
}.items():
    fmt[name] = wb.add_format({**BASE, **props})


def sheet(name, title, note, headers, widths=None):
    ws = wb.add_worksheet(name)
    ws.hide_gridlines(2)
    ws.set_tab_color('#008C95' if name == '结果总览' else '#718096')
    ws.set_default_row(23)
    ws.merge_range(0, 0, 0, len(headers) - 1, title, fmt['title'])
    ws.set_row(0, 36)
    ws.merge_range(1, 0, 2, len(headers) - 1, note, fmt['note'])
    ws.set_row(1, 23)
    ws.set_row(2, 23)
    ws.set_row(4, 36)
    ws.write_row(4, 0, headers, fmt['head'])
    ws.set_column(0, len(headers) - 1, 16, fmt['num'])
    for col, width in (widths or {}).items():
        ws.set_column(col, col, width)
    ws.freeze_panes(5, 3)
    ws.set_landscape()
    ws.set_paper(9)
    ws.fit_to_pages(1, 0)
    ws.repeat_rows(4)
    ws.set_margins(0.25, 0.25, 0.4, 0.4)
    ws.set_footer('&LExp_E4Probe&C第 &P 页 / 共 &N 页&R2026-11-12')
    return ws


def table(ws, data, formats=None):
    for row_idx, row in enumerate(data, 5):
        for col, value in enumerate(row):
            style = (formats or {}).get(col, fmt['num'] if isinstance(value, (float, int)) else fmt['text'])
            if value is None:
                ws.write_blank(row_idx, col, None, style)
            else:
                ws.write(row_idx, col, value, style)
    if data:
        ws.autofilter(4, 0, len(data) + 4, len(data[0]) - 1)
        ws.conditional_format(5, 0, len(data) + 4, len(data[0]) - 1,
                              {'type': 'formula', 'criteria': '=MOD(ROW(),2)=0',
                               'format': wb.add_format({'bg_color': '#F1F6F8'})})
    return len(data) + 5


def color_delta(ws, start, end, col, higher=True):
    ws.conditional_format(start, col, end, col, {'type': 'cell', 'criteria': '>' if higher else '<', 'value': 0, 'format': fmt['good']})
    ws.conditional_format(start, col, end, col, {'type': 'cell', 'criteria': '<' if higher else '>', 'value': 0, 'format': fmt['bad']})


overview = sheet('结果总览', 'C1 编码器边界 Probe · 实验结果汇总',
    'E4 / E3 × LA_8l / Pancreas_6l × seed 42 / 500 / 1337；12 次实验全部完成，另含 6 次 V0 对照。主指标为最佳验证模型的 test nms0。',
    ['数据集', '方法', 'seed 数', 'Dice (%) ↑', 'JC (%) ↑', 'HD95 (体素) ↓', 'ASD (体素) ↓',
     'ΔDice (百分点)', '最后5次验证 Dice (%)', 'Δ最后5次 (百分点)', '记录耗时 (h)', '预设判定'],
    {0: 19, 1: 12, 2: 9, 3: 19, 4: 19, 5: 18, 6: 18, 7: 19, 8: 25, 9: 23, 10: 18, 11: 17})
summary_data = []
for dataset in DATASETS:
    for method in METHODS:
        runs = group(method, dataset)
        delta = None if method == 'V0' else st.mean(deltas(method, dataset, 0, 'dice'))
        summary_data.append([dataset, 'V0 基线' if method == 'V0' else f'C1 {method}', 3,
             *[display([metric(r, 0, k) for r in runs]) for k in METRICS], delta,
             display([r['last5'] for r in runs]), None if method == 'V0' else st.mean(last5_delta(method, dataset)),
             display([r['hours'] for r in runs]), '对照' if method == 'V0' else '不通过'])
table(overview, summary_data, {2: fmt['int'], 7: fmt['delta'], 9: fmt['delta']})
color_delta(overview, 5, 10, 7)
color_delta(overview, 5, 10, 9)
overview.merge_range('A14:L14', '预设判据核对 · 主指标 test nms0 Dice', fmt['section'])
decision_headers = ['数据集', '层级', 'ΔDice (百分点)', '病例95% t下界', '病例95% t上界', '区间跨0', '均值差 > 0.4', '区间全为正', '单数据集判定', '层级最终判定', '病例数', 'seed数']
overview.write_row(14, 0, decision_headers, fmt['head'])
overview.set_row(14, 36)
for rowidx, (method, dataset) in enumerate(((m, d) for m in ('E4', 'E3') for d in DATASETS), 15):
    mean, sd, low, high = CASES[method, dataset, 0]['ci']
    passed = mean > 0.4 and low > 0
    overview.write_row(rowidx, 0, [dataset, method, mean, low, high, '是' if low <= 0 <= high else '否',
                       '是' if mean > 0.4 else '否', '是' if low > 0 else '否',
                       '通过' if passed else '不通过', '不通过', 20, 3], fmt['text'])
    for col in (2, 3, 4): overview.write_number(rowidx, col, (mean, low, high)[col-2], fmt['delta'])
overview.merge_range('A21:L21', '结论与阅读口径', fmt['section'])
notes = [
    '结论：E4、E3 在两个数据集上均未满足预设条件；实验日志据此决定停止编码器边界 Probe 方向。',
    'E4 / LA：Dice +0.27 个百分点，病例配对区间跨 0；Pancreas 为 −0.16 个百分点，未形成跨数据集收益。',
    'E3 / LA：最后 5 次验证均值比 V0 低 1.74 ± 0.90 个百分点；后期验证表现变差。',
    '表中“±”均为 3 个 seed 的样本标准差；Δ 为实验 − 同 seed V0。Dice/JC 差值单位为百分点。',
    '病例区间：同病例的 3 个 seed 差值先平均，再对 20 个病例计算 95% t 区间（df=19）；该区间算法为日志事后确定。',
    '评估协议：LA/Pancreas 的验证和最终评估都使用 test.list，最佳验证值等于 test nms0；此处沿用现有实验协议。',
]
for rowidx, note in enumerate(notes, 21):
    overview.merge_range(rowidx, 0, rowidx, 11, note, fmt['note'])
    overview.set_row(rowidx, 28)

stats = sheet('指标汇总', '多 seed 测试指标统计',
    '每行汇总同一数据集、方法和后处理模式下的 3 个 seed。均值、样本标准差分别保留为数值，可筛选和继续计算。',
    ['数据集', '方法', '后处理', 'seed数', 'Dice均值 (%)', 'Dice样本SD', 'JC均值 (%)', 'JC样本SD',
     'HD95均值 (体素)', 'HD95样本SD', 'ASD均值 (体素)', 'ASD样本SD', 'ΔDice均值 (百分点)',
     '配对ΔDice样本SD', '最后5次验证均值 (%)', '最后5次验证样本SD', '测试病例数/seed'], {0: 19, 12: 24, 14: 25})
data = []
for dataset in DATASETS:
    for method in METHODS:
        for mode in (0, 1):
            runs = group(method, dataset)
            values = [dataset, method, f'nms{mode}', 3]
            for key in METRICS: values += list(ms([metric(r, mode, key) for r in runs]))
            values += list(ms(deltas(method, dataset, mode, 'dice'))) if method != 'V0' else [None, None]
            data.append(values + list(ms([r['last5'] for r in runs])) + [20])
table(stats, data, {3: fmt['int'], 12: fmt['delta'], 16: fmt['int']})
color_delta(stats, 5, 16, 12)

details = sheet('逐次实验', '逐次实验测试结果 · 18 个 run',
    '来自各 run 的 test_metrics.json；nms0 为主结果，nms1 为保留最大 26 连通区域后的结果。模型为最佳验证 Student。',
    ['数据集', '方法', 'seed', '训练状态', '最佳步数', 'nms0 Dice (%)', 'nms0 JC (%)', 'nms0 HD95', 'nms0 ASD',
     'nms1 Dice (%)', 'nms1 JC (%)', 'nms1 HD95', 'nms1 ASD', 'nms0空预测数', 'nms1空预测数',
     '病例数', 'Δnms0 Dice (百分点)', '最后5次验证 Dice (%)', '结果文件'],
    {0: 19, 16: 23, 17: 26, 18: 95})
ordered = [RUNS[m, d, s] for d in DATASETS for m in METHODS for s in SEEDS]
data = []
for run in ordered:
    method, dataset, seed = run['method'], run['dataset'], run['seed']
    data.append([dataset, method, seed, '完成 15000/15000', run['best']['iteration'],
        *[metric(run, mode, key) for mode in (0, 1) for key in METRICS],
        run['test']['empty_prediction_cases'], run['test']['nms1']['empty_prediction_cases'], run['test']['cases'],
        None if method == 'V0' else metric(run, 0, 'dice') - metric(RUNS['V0', dataset, seed], 0, 'dice'),
        run['last5'], str((run['path']/'test_metrics.json').relative_to(EXP)) if method != 'V0' else str((run['path']/'test_metrics.json').relative_to(ROOT))])
table(details, data, {2: fmt['int'], 4: fmt['int'], 13: fmt['int'], 14: fmt['int'], 15: fmt['int'], 16: fmt['delta']})
for idx, run in enumerate(ordered, 5):
    details.write_url(idx, 18, 'external:' + str(run['path'] / 'test_metrics.json'), fmt['link'], data[idx-5][-1])
color_delta(details, 5, 22, 16)

paired = sheet('基线配对', '按 seed 配对比较 · C1 − V0',
    '逐 seed 行为同数据集、同 seed 的实验结果减 V0；“均值”行为 3 个配对差值的均值。SD 仅在均值行展示。Dice/JC 差值为百分点，HD95/ASD 为体素。',
    ['数据集', '方法', 'seed/汇总', 'Δnms0 Dice ↑', 'Δnms0 JC ↑', 'Δnms0 HD95 ↓', 'Δnms0 ASD ↓',
     'Δnms1 Dice ↑', 'Δnms1 JC ↑', 'Δnms1 HD95 ↓', 'Δnms1 ASD ↓', 'Δ最后5次验证 ↑', 'Δnms0 Dice样本SD', 'Δ最后5次样本SD'],
    {0: 19, 11: 23, 12: 24, 13: 23})
data = []
for dataset in DATASETS:
    for method in ('E4', 'E3'):
        differences = [[*[(metric(RUNS[method, dataset, seed], mode, key) - metric(RUNS['V0', dataset, seed], mode, key)) for mode in (0, 1) for key in METRICS],
                        RUNS[method, dataset, seed]['last5'] - RUNS['V0', dataset, seed]['last5']] for seed in SEEDS]
        for seed, row in zip(SEEDS, differences): data.append([dataset, method, seed, *row, None, None])
        data.append([dataset, method, '均值 (n=3)', *[st.mean(row[c] for row in differences) for c in range(9)],
                     st.stdev(row[0] for row in differences), st.stdev(row[8] for row in differences)])
table(paired, data, {col: fmt['delta'] for col in range(3, 12)})
for col in range(3, 12): color_delta(paired, 5, 20, col, col not in (5, 6, 9, 10))

case_stats = sheet('病例配对区间', '逐病例 Dice 配对差值与 95% t 区间',
    '“先平均seed”行为每病例先平均 3 个 seed 的差值，再用 20 个病例计算区间；“单seed”行为各 seed 的 20 个病例区间。不是跨 seed 的置信区间。',
    ['数据集', '方法', '后处理', '计算层级', 'seed', '病例数', 'ΔDice均值 (百分点)', '病例差值样本SD',
     '95% t下界', '95% t上界', '区间跨0', '正病例数', '负病例数', '零病例数'], {0: 19, 3: 22, 6: 26, 7: 23})
data = []
for dataset in DATASETS:
    for method in ('E4', 'E3'):
        for mode in (0, 1):
            for seed in ('先平均seed', *SEEDS):
                values = list(CASES[method, dataset, mode]['values'].values()) if isinstance(seed, str) else list(case_differences(method, dataset, mode, seed).values())
                mean, sd, low, high = interval(values)
                data.append([dataset, method, f'nms{mode}', '每病例先平均3个seed' if isinstance(seed, str) else '单seed',
                     '42 / 500 / 1337' if isinstance(seed, str) else seed, 20, mean, sd, low, high,
                     '是' if low <= 0 <= high else '否', sum(v > 0 for v in values), sum(v < 0 for v in values), sum(v == 0 for v in values)])
table(case_stats, data, {6: fmt['delta'], 8: fmt['delta'], 9: fmt['delta']})

training = sheet('验证与资源', '验证表现、运行时间与资源',
    '记录耗时 = training_metrics.csv 最后一行 elapsed_seconds / 3600；包含截至该记录的开销。完整日志耗时 = 最后日志时间 − 首条日志时间，含末尾评估。时间按原日志记录展示。',
    ['数据集', '方法', 'seed', '完成步数', '验证次数', '最佳验证 Dice (%)', '最佳验证步数', '最后5次验证 Dice (%)',
     '最后一次验证 Dice (%)', '13k–15k最低 Dice (%)', '13k–15k低于85次数', '13k–15k验证次数',
     '记录耗时 (h)', '完整日志耗时 (h)', '峰值显存 (MiB)', 'GPU编号', '日志开始时间', '日志结束时间'],
    {0: 19, 5: 25, 7: 28, 8: 27, 9: 28, 10: 27, 16: 24, 17: 24})
data = []
for run in ordered:
    late = [r for r in run['val'] if 13000 <= r['iteration'] <= 15000]
    manifest = MANIFESTS.get((run['method'], run['config']['args']['dataset'], run['seed']), {})
    data.append([run['dataset'], run['method'], run['seed'], int(run['train'][-1]['iteration']), len(run['val']),
        run['best_val']['dice'] * 100, run['best_val']['iteration'], run['last5'], run['last'], min(r['dice'] for r in late)*100,
        sum(r['dice'] < .85 for r in late), len(late), run['hours'], (run['end']-run['start']).total_seconds()/3600,
        max(float(r['peak_memory_mib']) for r in run['train']), manifest.get('gpu'), run['start'], run['end']])
table(training, data, {2: fmt['int'], 3: fmt['int'], 4: fmt['int'], 6: fmt['int'], 10: fmt['int'], 11: fmt['int'], 15: fmt['int'], 16: fmt['date'], 17: fmt['date']})

boundary = sheet('边界Probe', '边界 Probe 诊断 · 各阶段中位数',
    '每个 run 在闭区间 [1,2000] 或 [6000,15000] 内取有限值中位数；grad_ratio/grad_cos 只使用实际记录步。boundary_dice 是 soft Dice 损失，越低越好，不是分割 Dice。',
    ['数据集', '方法', 'seed', '迭代区间', '边界日志行数', '梯度记录数', 'loss_boundary中位数', 'boundary_bce中位数',
     'boundary_dice损失中位数', 'boundary_fraction中位数', 'grad_ratio中位数', 'grad_cos中位数'],
    {0: 19, 3: 20, 6: 26, 7: 26, 8: 31, 9: 30, 10: 25, 11: 23})
data = []
for run in ordered:
    if not run['boundary']: continue
    for low, high in ((1, 2000), (6000, 15000)):
        rows = [row for row in run['boundary'] if low <= int(row['iteration']) <= high]
        data.append([run['dataset'], run['method'], run['seed'], f'[{low}, {high}]', len(rows), len(finite(rows, 'grad_ratio')),
                     *[median(rows, key) for key in ('loss_boundary', 'boundary_bce', 'boundary_dice', 'boundary_fraction', 'grad_ratio', 'grad_cos')]])
table(boundary, data, {col: fmt['small'] for col in range(6, 12)})

diag_keys = ('band_iou', 'ce_gt_in', 'ce_pl_in', 'ce_lab_in', 'teacher_err_gt_band', 'conf_wrong_095',
             'cos_pl_true_b7', 'cos_pl_true_b3', 'cos_sup_true_b7', 'cos_sup_true_b3')
diagnostics = sheet('边界诊断', '边界与梯度诊断 · 训练后期中位数',
    '每个 run 在闭区间 [6000,15000] 内取中位数，忽略 NaN；NaN 不是 0。末列给出 ce_gt_in 非有限值数。隐藏 GT 仅用于诊断，未进入训练目标。',
    ['数据集', '方法', 'seed', '迭代区间', '行数', *diag_keys, 'ce_gt_in非有限值数'], {0: 19, 3: 20, **{i: 25 for i in range(5, 16)}})
data = []
for run in ordered:
    rows = [row for row in run['diag'] if 6000 <= int(row['iteration']) <= 15000]
    data.append([run['dataset'], run['method'], run['seed'], '[6000, 15000]', len(rows),
                 *[median(rows, key) for key in diag_keys], len(rows) - len(finite(rows, 'ce_gt_in'))])
table(diagnostics, data, {col: fmt['small'] for col in range(5, 15)})

percase = sheet('逐病例结果', '逐病例测试结果 · 720 行',
    '18 个 run × 2 种后处理 × 20 个病例；差值按同数据集、同 seed、同病例、同后处理与 V0 配对。基线的差值留空。',
    ['数据集', '方法', 'seed', '后处理', '病例ID', 'Dice (%)', 'JC (%)', 'HD95 (体素)', 'ASD (体素)', '空预测',
     'ΔDice (百分点)', 'ΔJC (百分点)', 'ΔHD95 (体素)', 'ΔASD (体素)'], {0: 19, 4: 28, 10: 22, 11: 22})
data = []
for run in ordered:
    for mode in (0, 1):
        baseline = case_map(RUNS['V0', run['dataset'], run['seed']], mode)
        for case_id, case in case_map(run, mode).items():
            diffs = [(case[key]-baseline[case_id][key]) * (100 if key in ('dice', 'jc') else 1) for key in METRICS] if run['method'] != 'V0' else [None]*4
            data.append([run['dataset'], run['method'], run['seed'], f'nms{mode}', case_id,
                 *[case[key] * (100 if key in ('dice', 'jc') else 1) for key in METRICS], '是' if case['empty_prediction'] else '否', *diffs])
table(percase, data, {2: fmt['int'], **{col: fmt['delta'] for col in range(10, 14)}})
color_delta(percase, 5, 724, 10)

curve_headers = ['iteration'] + [f'{d} {m} 均值' for d in DATASETS for m in METHODS] + [f'{d} {m} SD' for d in DATASETS for m in METHODS] + [f'{r["dataset"]} {r["method"]} seed{r["seed"]}' for r in ordered]
curves = sheet('验证曲线', '完整验证轨迹 · Dice (%)',
    '72 个验证点（800–15000，每 200 步）。B:G 为 3 seed 均值，H:M 为对应样本 SD，其后是 18 条逐次实验轨迹。总览图只展示均值。',
    curve_headers, {0: 12, **{i: 29 for i in range(1, len(curve_headers))}})
iterations = [r['iteration'] for r in ordered[0]['val']]
assert all([r['iteration'] for r in run['val']] == iterations for run in ordered)
data = []
for index, iteration in enumerate(iterations):
    means, sds = [], []
    for dataset in DATASETS:
        for method in METHODS:
            mean, sd = ms([r['val'][index]['dice']*100 for r in group(method, dataset)])
            means.append(mean)
            sds.append(sd)
    data.append([iteration, *means, *sds, *[r['val'][index]['dice']*100 for r in ordered]])
table(curves, data, {0: fmt['int']})
for index, dataset in enumerate(DATASETS):
    chart = wb.add_chart({'type': 'line'})
    for method_index, method in enumerate(METHODS):
        col = 1 + index*3 + method_index
        chart.add_series({'name': method, 'categories': ['验证曲线', 5, 0, 76, 0],
             'values': ['验证曲线', 5, col, 76, col], 'line': {'color': COLORS[method], 'width': 1.8}})
    chart.set_title({'name': f'{dataset} · 验证 Dice（3 seed均值）'})
    chart.set_x_axis({'name': 'iteration', 'interval_unit': 10, 'label_position': 'low'})
    chart.set_y_axis({'name': 'Dice (%)', 'min': 72 if index == 0 else 60, 'max': 91 if index == 0 else 82,
                      'major_gridlines': {'visible': True, 'line': {'color': '#E6EDF0'}}})
    chart.set_legend({'position': 'bottom'})
    chart.set_chartarea({'border': {'none': True}})
    chart.set_size({'width': 680, 'height': 340})
    overview.insert_chart(29, index*6, chart)
overview.merge_range('A48:L49', '图为验证均值轨迹，纵轴范围已标明；具体 seed 波动见“验证曲线”。完整测试指标、配对区间与数据来源见对应工作表。', fmt['note'])
overview.fit_to_pages(1, 1)
overview.print_area('A1:L49')

argkeys = ['dataset', 'labelnum', 'patch_size', 'batch_size', 'labeled_bs', 'max_iterations',
           'warmup_iterations', 'consistency', 'base_lr', 'ema_decay', 'val_start', 'val_interval',
           'diag_interval', 'probe_level', 'boundary_weight', 'ratio_interval']
config = sheet('实验配置', '实验配置与可比性',
    '取自 config.json。V0 无 Probe，boundary_weight 按有效值记 0；C1 两数据集 λ=0.03。训练样本ID一致性为与同 seed V0 的列表逐项比较。',
    ['数据集', '方法', 'seed', *argkeys, '有标签样本数', '无标签样本数', '训练ID与V0一致', '共同参数与V0一致', 'git_commit', 'run目录'],
    {0: 19, 5: 22, 21: 24, 22: 26, 23: 46, 24: 100})
data = []
common = [key for key in argkeys if key not in ('probe_level', 'boundary_weight', 'ratio_interval')]
for run in ordered:
    args = run['config']['args']
    baseline = RUNS['V0', run['dataset'], run['seed']]['config']
    idmatch = all(run['config'][key] == baseline[key] for key in ('labeled_ids', 'unlabeled_ids'))
    argmatch = all(args.get(key) == baseline['args'].get(key) for key in common)
    assert idmatch and argmatch
    vals = []
    for key in argkeys:
        val = args.get(key, 'none' if key == 'probe_level' else 0 if key == 'boundary_weight' else None)
        if isinstance(val, list): val = ' × '.join(str(v) for v in val)
        vals.append(val)
    data.append([run['dataset'], run['method'], run['seed'], *vals, len(run['config']['labeled_ids']),
         len(run['config']['unlabeled_ids']), '是' if idmatch else '否', '是' if argmatch else '否',
         run['config']['git_commit'], str(run['path'].relative_to(ROOT))])
table(config, data)

explain = sheet('统计口径', '统计口径与工作表导航',
    f'本工作簿由原始 JSON/CSV 重新计算。生成时间：{NOW.strftime("%Y-%m-%d %H:%M:%S")}（Asia/Shanghai）。结果文件未改动；数据来源表保留 SHA-256。',
    ['项目', '说明'], {0: 30, 1: 135})
explanations = [
    ['实验范围', 'C1 E4/E3 各 6 次正式实验，共 12 次；V0 基线 6 次；不纳入 tests/outputs 中的冒烟、验收或旧框架实验。'],
    ['基线目录', str(ROOTS['V0'])],
    ['实验目录', str(PROJECT)],
    ['主结果', 'test_metrics.json 顶层 nms0，最佳验证 Student；不是训练最后一步模型。'],
    ['nms0 / nms1', 'nms0 不做最大连通区域后处理；nms1 保留最大 26 连通区域。'],
    ['指标单位与方向', 'Dice、JC = 原始值 × 100（%），越高越好；HD95、ASD 单位体素，越低越好；Dice/JC 之差为百分点。'],
    ['均值 ± SD', '先取每个 seed 的病例平均指标，再求 3 seed 均值及样本 SD（ddof=1）；不是汇合 60 病例求 SD。'],
    ['原 JSON 中的 *_std', '为单 run 跨 20 病例的总体标准差（ddof=0），不等于本表的跨 seed 样本 SD；保留于源文件。'],
    ['基线配对', '实验 − V0，严格匹配数据集、seed、后处理；病例分析另匹配病例ID。'],
    ['病例95%区间', '对每病例的 3 seed 差值先求平均，得到 n=20 差值；mean ± t(0.975,19)×sample_SD/√20，t=2.093024054408263。'],
    ['区间解释', '描述病例层面的平均差值不确定性；不是对 seed 随机性的联合区间；未将 60 个 seed×病例当作独立样本。'],
    ['判据来源', 'README.md C1 节：某层仅在两个数据集均满足 test nms0 Dice 比 V0 高 >0.4，且病例配对差值区间不跨0，才算有效；另参考最后5次验证。'],
    ['区间算法来源', 'README 启动前只规定区间不跨0；日志 2026-11-12.md 第3.5节事后确定先平均seed再算 t 区间，本表沿用。'],
    ['最终判定', '四组均不通过；两个层级均不通过。日志第4节决定停止编码器边界 Probe，不再调整 λ 或层级。'],
    ['验证和测试协议', 'LA/Pancreas 都在 test.list 上进行验证并选择最佳模型，最终也在同一 test.list 上评估；因此最佳验证 Dice 与 test nms0 相同，不是独立留出验证集。'],
    ['最后5次验证', '逐 run 最后 5 条去重验证 Dice 的算术均值，即 14200、14400、14600、14800、15000 步。'],
    ['重复迭代记录', 'JSONL/CSV 按 iteration 去重，保留最后一次，再按 iteration 升序。'],
    ['边界/诊断统计', '先在每 run 的指定闭区间内取有限值中位数；NaN 忽略，缺项留空。Probe 梯度每200步记录一次，不将其余行的 NaN 计为0。'],
    ['boundary_dice', 'soft Dice 损失，越低越好；不是作为主结果的分割 Dice。'],
    ['grad_ratio', '||∇(λL_b)|| / ||∇L_sup||，只在到达 Probe 层的编码器块上计算；不参与训练目标。'],
    ['时间和显存', '资源页分别列出末行 elapsed_seconds 和完整日志时间差；训练结束后的末尾验证/测试不一定包含在前者。显存取记录中的 peak_memory_mib 最大值。'],
    ['空预测', '按现有评估实现 Dice/JC/HD95/ASD 记0并计入平均；本次18个run的两种后处理均无空预测。'],
    ['结果总览', '主结果、通过条件核对、已记录结论和两数据集的验证均值图。'],
    ['指标汇总 / 逐次实验', '完整 nms0/nms1 指标均值、样本SD和18次实验明细。'],
    ['基线配对 / 病例配对区间', '按seed的配对差值；病例层面及单seed的95% t区间。'],
    ['验证与资源 / 验证曲线', '最佳/后期验证表现、时间/显存、72个验证点的均值/SD和各run轨迹。'],
    ['边界Probe / 边界诊断', 'Probe前期/后期中位数，以及边界与梯度核心诊断量。'],
    ['逐病例结果', '720行病例指标，包含配对V0的差值。'],
    ['实验配置 / 数据来源', '配置、训练ID一致性、commit及原始文件的校验摘要和可点击位置。'],
]
table(explain, explanations)
for rowidx in range(5, 5+len(explanations)):
    explain.set_row(rowidx, 44)
    explain.write(rowidx, 1, explanations[rowidx-5][1], wb.add_format({**BASE, 'text_wrap': True}))

sources = sheet('数据来源', '原始文件来源与校验摘要',
    '每行是生成时实际读取的源文件；SHA-256 可验证之后文件是否发生变化。路径相对于工作区根目录，点击可打开本地源文件。',
    ['方法', '数据集', 'seed', '文件名', '字节数', 'SHA-256', '源文件路径'], {0: 12, 1: 19, 2: 10, 3: 28, 4: 14, 5: 68, 6: 115})
table(sources, SOURCES, {2: fmt['int'], 4: fmt['int']})
for rowidx, row in enumerate(SOURCES, 5):
    sources.write_url(rowidx, 6, 'external:' + str(ROOT / row[6]), fmt['link'], row[6])

overview.activate()
overview.set_first_sheet()
overview.freeze_panes(5, 2)
overview.set_zoom(75)
for ws in wb.worksheets():
    if ws.name != '结果总览': ws.set_zoom(85)
wb.close()
print(f'Wrote {OUTPUT}')
print(f'Runs: {len(RUNS)}; source files: {len(SOURCES)}; sheets: {len(wb.worksheets())}')
for method in ('E4', 'E3'):
    for dataset in DATASETS:
        mean, sd, low, high = CASES[method, dataset, 0]['ci']
        print(f'{method} {dataset}: ΔDice={mean:+.6f}, case 95% CI=[{low:+.6f}, {high:+.6f}]')
