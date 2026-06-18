#!/usr/bin/env python3
# -- coding: UTF-8
"""Generate an icon-only Chinese overview of shape-hole collection layouts."""

import argparse
import glob
import html
import itertools
import json
import os
import random
import re
from collections import Counter

import h5py


SHAPES = [
    {"key": "hexagonal_prism", "zh": "六边形柱", "color": "#2f80ed"},
    {"key": "cuboid", "zh": "长方体", "color": "#e07a2f"},
    {"key": "cube", "zh": "正方体", "color": "#2f9e44"},
    {"key": "triangular_prism", "zh": "三角形柱", "color": "#9b51e0"},
]

TOP_POSITIONS = ["左", "中左", "中右", "右"]
HOLE_POSITIONS = ["左", "中", "右"]
TOP_SHAPE_CYCLE = list(itertools.permutations(range(len(SHAPES)), len(TOP_POSITIONS)))
HOLE_SHAPE_CYCLE = list(itertools.permutations(range(len(SHAPES)), len(HOLE_POSITIONS)))
DEFAULT_SEED = 20260516


def build_layout_pair_cycle(seed):
    rng = random.Random(seed)
    count = len(TOP_SHAPE_CYCLE)
    top_order = list(range(count))
    hole_order = list(range(count))
    block_offsets = list(range(count))
    rng.shuffle(top_order)
    rng.shuffle(hole_order)
    rng.shuffle(block_offsets)

    pairs = []
    for offset in block_offsets:
        for pos, top_idx in enumerate(top_order):
            pairs.append((top_idx, hole_order[(pos + offset) % count]))
    return pairs


def episode_idx_from_path(path):
    match = re.search(r"episode_(\d+)\.hdf5$", os.path.basename(path))
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def read_plan(path):
    with h5py.File(path, "r") as root:
        raw = root.attrs.get("shape_hole_layout")
    if raw is None:
        raise ValueError(f"{path} missing shape_hole_layout")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def normalize_existing_plan(path):
    plan = read_plan(path)
    episode_idx = episode_idx_from_path(path)
    top_idx = int(plan["top_cycle_idx"])
    hole_idx = int(plan["hole_cycle_idx"])
    return build_plan(episode_idx, top_idx, hole_idx, source="已有")


def build_plan(episode_idx, top_cycle_idx, hole_cycle_idx, source="待采"):
    top_indices = TOP_SHAPE_CYCLE[top_cycle_idx]
    hole_indices = HOLE_SHAPE_CYCLE[hole_cycle_idx]
    omitted_idx = ({0, 1, 2, 3} - set(hole_indices)).pop()
    return {
        "episode_idx": int(episode_idx),
        "top_cycle_idx": int(top_cycle_idx),
        "hole_cycle_idx": int(hole_cycle_idx),
        "source": source,
        "top": [
            {"position": pos, "shape_idx": int(shape_idx)}
            for pos, shape_idx in zip(TOP_POSITIONS, top_indices)
        ],
        "holes": [
            {"position": pos, "shape_idx": int(shape_idx)}
            for pos, shape_idx in zip(HOLE_POSITIONS, hole_indices)
        ],
        "omitted_shape_idx": int(omitted_idx),
    }


def build_sequence(data_dir, target_count, seed):
    paths = sorted(
        glob.glob(os.path.join(data_dir, "episode_*.hdf5")),
        key=episode_idx_from_path,
    )
    plans = [normalize_existing_plan(path) for path in paths]
    used_pairs = {
        (plan["top_cycle_idx"], plan["hole_cycle_idx"])
        for plan in plans
    }
    pair_cycle = build_layout_pair_cycle(seed)
    next_episode_idx = (max([plan["episode_idx"] for plan in plans]) + 1) if plans else 0

    while len(plans) < target_count:
        for top_idx, hole_idx in pair_cycle:
            pair = (top_idx, hole_idx)
            if pair in used_pairs:
                continue
            plans.append(build_plan(next_episode_idx, top_idx, hole_idx, source="待采"))
            used_pairs.add(pair)
            next_episode_idx += 1
            break
        else:
            raise RuntimeError("576 种组合已经全部用完")
    return plans[:target_count]


def shape_svg(shape_idx, filled):
    shape = SHAPES[shape_idx]
    color = shape["color"]
    fill = color if filled else "#ffffff"
    stroke = "#222222" if filled else color
    width = "2.5" if filled else "6"
    key = shape["key"]
    if key == "hexagonal_prism":
        body = (
            '<polygon points="96,50 73,90 27,90 4,50 27,10 73,10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'
        )
    elif key == "cuboid":
        body = (
            '<rect x="35" y="6" width="30" height="88" rx="1" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'
        )
    elif key == "cube":
        body = (
            '<rect x="21" y="21" width="58" height="58" rx="1" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'
        )
    else:
        body = (
            '<polygon points="50,8 10,82 90,82" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'
        )
    return f'<svg class="图标" viewBox="0 0 100 100" aria-label="{shape["zh"]}">{body}</svg>'


def icon_cell(item, filled):
    shape_idx = item["shape_idx"]
    shape = SHAPES[shape_idx]
    return (
        '<div class="格">'
        f'<div class="位置">{html.escape(item["position"])}</div>'
        f'{shape_svg(shape_idx, filled)}'
        f'<div class="名称">{shape["zh"]}</div>'
        '</div>'
    )


def render_plan_card(plan):
    top = "".join(icon_cell(item, filled=True) for item in plan["top"])
    holes = "".join(icon_cell(item, filled=False) for item in plan["holes"])
    omitted = SHAPES[plan["omitted_shape_idx"]]
    return f"""
<section class="卡片">
  <header>
    <span class="编号">第 {plan["episode_idx"]:03d} 条</span>
    <span class="来源">{plan["source"]}</span>
    <span class="组合">组 {plan["top_cycle_idx"] + 1:02d}-{plan["hole_cycle_idx"] + 1:02d}</span>
  </header>
  <div class="行">
    <div class="行名">物体</div>
    <div class="图标行 四列">{top}</div>
  </div>
  <div class="行">
    <div class="行名">孔</div>
    <div class="图标行 三列">{holes}</div>
  </div>
  <div class="缺少">无孔：{shape_svg(plan["omitted_shape_idx"], filled=False)}<span>{omitted["zh"]}</span></div>
</section>
"""


def render_html(plans, data_dir, output_path):
    hole_counts = Counter()
    for plan in plans:
        for item in plan["holes"]:
            hole_counts[SHAPES[item["shape_idx"]]["zh"]] += 1
    count_text = "　".join(f"{name}：{count}" for name, count in sorted(hole_counts.items()))
    cards = "\n".join(render_plan_card(plan) for plan in plans)
    source_count = Counter(plan["source"] for plan in plans)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>形状入孔摆放总表</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: #222;
      background: #f2f3f5;
    }}
    .页眉 {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 12px 16px;
      border-bottom: 1px solid #d8d8d8;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .摘要 {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 24px;
      font-size: 14px;
      line-height: 1.5;
    }}
    .图例 {{
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 13px;
    }}
    .图例项 {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .图例项 .图标 {{ width: 24px; height: 24px; }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 10px;
      padding: 12px;
    }}
    .卡片 {{
      min-width: 0;
      border: 1px solid #d4d7dc;
      border-radius: 6px;
      background: #ffffff;
      padding: 8px;
      break-inside: avoid;
    }}
    .卡片 header {{
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid #ededed;
      padding-bottom: 5px;
      margin-bottom: 6px;
      white-space: nowrap;
    }}
    .编号 {{
      font-weight: 700;
      font-size: 15px;
    }}
    .来源, .组合 {{
      padding: 1px 5px;
      border: 1px solid #dadada;
      border-radius: 4px;
      font-size: 12px;
      color: #555;
      background: #f7f7f7;
    }}
    .行 {{
      display: grid;
      grid-template-columns: 28px 1fr;
      align-items: center;
      gap: 5px;
      margin: 4px 0;
    }}
    .行名 {{
      font-weight: 700;
      font-size: 13px;
      text-align: center;
      color: #333;
    }}
    .图标行 {{
      display: grid;
      gap: 4px;
      align-items: start;
    }}
    .四列 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .三列 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .格 {{
      min-width: 0;
      text-align: center;
      border: 1px solid #ececec;
      border-radius: 4px;
      background: #fafafa;
      padding: 2px 1px 3px;
    }}
    .位置 {{
      font-size: 11px;
      font-weight: 700;
      color: #555;
      line-height: 1.1;
    }}
    .名称 {{
      font-size: 10px;
      color: #333;
      line-height: 1.1;
      white-space: nowrap;
    }}
    .图标 {{
      width: 38px;
      height: 38px;
      display: block;
      margin: 0 auto;
    }}
    .缺少 {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 5px;
      margin-top: 4px;
      font-size: 12px;
      color: #8a3b00;
      font-weight: 700;
    }}
    .缺少 .图标 {{ width: 22px; height: 22px; }}
    @media print {{
      .页眉 {{ position: static; }}
      main {{ grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 6px; }}
      .卡片 {{ padding: 6px; }}
    }}
  </style>
</head>
<body>
  <div class="页眉">
    <h1>形状入孔摆放总表</h1>
    <div class="摘要">
      <span>总数：{len(plans)} 条</span>
      <span>已有：{source_count.get("已有", 0)} 条</span>
      <span>待采：{source_count.get("待采", 0)} 条</span>
      <span>{count_text}</span>
    </div>
    <div class="图例">
      {''.join(f'<span class="图例项">{shape_svg(i, True)}{shape["zh"]}</span>' for i, shape in enumerate(SHAPES))}
    </div>
  </div>
  <main>
    {cards}
  </main>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/agilex/data/shape_hole_pipeline/put_the_shapes_into_the_matching_holes",
    )
    parser.add_argument("--output", default="docs/shape_hole_200_layouts.html")
    parser.add_argument("--target_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    plans = build_sequence(args.data_dir, args.target_count, args.seed)
    render_html(plans, args.data_dir, args.output)
    print(f"生成完成：{args.output}")
    print(f"总数：{len(plans)}")
    print(f"已有：{sum(1 for plan in plans if plan['source'] == '已有')}")
    print(f"待采：{sum(1 for plan in plans if plan['source'] == '待采')}")


if __name__ == "__main__":
    main()
