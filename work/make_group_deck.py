"""Build the group-meeting deck (2026-11-28) from the weekly report numbers.

Run with the workspace Python (python-pptx 1.0.2).  Every number here is copied from
work/weekly-report-2026-11-27.md; nothing is invented in the slides.

Note on quoting: inner quotations inside Chinese text use 「」 on purpose, so that no ASCII double
quote ever appears inside a Python string literal.
"""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

WORK = os.path.dirname(os.path.abspath(__file__))
FIG_ALT = os.path.join(WORK, "results", "fig_altitude_sr.png")
FIG_ARMS = os.path.join(WORK, "results", "fig_arms_sr.png")
OUT = os.path.join(WORK, "group-meeting-2026-11-28.pptx")

FONT = "Microsoft YaHei"
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x60, 0x60, 0x60)
BLUE = RGBColor(0x1F, 0x5F, 0x94)
RED = RGBColor(0xB2, 0x1F, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def textbox(slide, left, top, width, height, text, *, size=16, bold=False, color=INK,
            align=PP_ALIGN.LEFT, space_after=6, line_spacing=1.2):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT
        run.font.color.rgb = color
    return box


def title_slide(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, Inches(0), Inches(2.55), Inches(13.333), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    textbox(s, 0.9, 1.45, 11.5, 1.0, title, size=36, bold=True)
    textbox(s, 0.9, 2.85, 11.5, 1.8, subtitle, size=17, color=GREY, line_spacing=1.4)
    return s


def section(prs, kicker, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(s, 0.85, 0.42, 11.8, 0.4, kicker, size=13, color=BLUE, bold=True)
    textbox(s, 0.85, 0.82, 11.8, 0.9, title, size=26, bold=True)
    return s


def table(slide, left, top, width, rows, col_widths=None, *, font_size=12, header_size=12,
          row_h=0.36):
    nrows, ncols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top), Inches(width),
                                   Inches(row_h * nrows))
    tbl = shape.table
    if col_widths:
        total = float(sum(col_widths))
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(width * w / total)
    for r, row in enumerate(rows):
        for c, cell_text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(cell_text)
            run.font.size = Pt(header_size if r == 0 else font_size)
            run.font.name = FONT
            run.font.bold = r == 0
            run.font.color.rgb = WHITE if r == 0 else INK
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
    return tbl


def main() -> int:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_slide(
        prs,
        "UAV VLN 的任务逻辑层：这周做到哪、没做到哪",
        "2026-11-27 周报 ｜ 组会 11-28\n"
        "一句话：把「只有一个很小的测试」做成了「三个点各自可复现的实验平台」，\n"
        "但三个点都没有拿到经得起检验的成功率提升。",
    )

    # ---- 2 ----
    s = section(prs, "1 / 本周产出", "三个点各自可复现，并加了外部对照")
    table(
        s, 0.8, 1.85, 11.7,
        [
            ["点", "本周状态", "关键数字"],
            ["idea1  LTLf 任务逻辑层", "上轮实现，本周扩到 5 张地图", "任务 124/145/165/281/40；MCM RT 组对照 N=21"],
            ["idea2  STL 定量鲁棒层", "新增：3 模块 + 18 测试 + 离线 + 在线", "单测 80 全通过；STL 判据 ≡ LTLf（8 组全部 1.000）"],
            ["idea3  逻辑对齐后训练", "新增：数据管线 + 3 次训练 + 评测", "Stage-A 数据 13,715 条（零新标注）"],
            ["外部对照  RefineFly", "代码未开源、权重匿名开源", "已下载 397 MB adapter，跑完 12 条对照"],
        ],
        col_widths=[2.0, 4.0, 5.7], font_size=13,
    )
    textbox(s, 0.85, 5.35, 11.6, 1.3,
            "平台侧结论：仿真器按 map_name 自动开场景 → 加地图零代码改动；\n"
            "但 GPU 是共享的（他人占用在 21 到 79 GB 之间波动），本轮 4 次 CUDA OOM 全部来自这里。",
            size=13, color=GREY)

    # ---- 3 ----
    s = section(prs, "2 / 必须先讲的三件事", "三个负面结果")
    textbox(s, 0.85, 1.75, 5.7, 2.3,
            "① idea1 的成功率增益不成立\n"
            "N=21：.333 vs .333，按任务配对 McNemar p=1.00。\n"
            "上轮 N=8 的 .625 vs .250 是小样本噪声。\n"
            "能站住的只有过程指标：阶段完成率 .683 vs .333。",
            size=14)
    textbox(s, 6.8, 1.75, 5.7, 2.3,
            "② idea2 的两次高度干预都无效\n"
            "下限 1.72→6 m：中位高度反而 8.76→6.09 m（贴着地板飞），SR .333→.167；\n"
            "带约束 [7,12] m：带内时间 37.4%→71.4%（干预真生效），SR 仍 .250。",
            size=14)
    textbox(s, 0.85, 4.2, 11.7, 2.4,
            "③ idea3 从裸底座训 → 策略塌缩；从 checkpoint 续训 → 治好\n"
            "v1/v3 从 openvla-7b 起步：139 步只有 2 种动作、SR 0 → 不是 lr 问题，而是「用 2k 样本重建策略」；\n"
            "v4 从 AeroVLA checkpoint 续训：动作多样性 258/358 步、SR .417（vs 原权重 .333，N=12，p=1.00）。",
            size=14)

    # ---- 4 ----
    s = section(prs, "3 / 诊断", "失败的主因是碰撞，而且集中在起飞后十几步")
    if os.path.exists(FIG_ALT):
        s.shapes.add_picture(FIG_ALT, Inches(0.65), Inches(1.7), width=Inches(7.7))
    textbox(s, 8.55, 1.7, 4.25, 4.6,
            "碰撞率：两臂都约 66%\n\n"
            "成功 episode 平均高度 8.5 m\n碰撞 episode 只有 4.7 m\n（专家中位 9.8 m）\n\n"
            "但把高度压上去并不改善 SR\n→ 这是相关，不是因果。\n\n"
            "真正的位置规律：\n撞在第 10–20 步（p10=3–4 步），\n离目标还有约 50 m。",
            size=13)
    textbox(s, 0.85, 6.05, 11.7, 0.9,
            "→ 下周最该做的不是改提示词，而是「起飞段净空保护」：先爬升再巡航 + 深度图近场净空 + 起点可达性检查。",
            size=14, bold=True, color=RED)

    # ---- 5 ----
    s = section(prs, "4 / idea2", "STL 层：机制可用、判据可信，但两版规范都被实验否定")
    table(
        s, 0.8, 1.75, 11.7,
        [
            ["项", "结果"],
            ["实现与测试", "stl.py / stl_tasks.py / stl_layer.py + 18 个测试；全套 80 passed"],
            ["判据一致性（关键）", "8 组、上百条 episode：STL 布尔语义 ≡ LTLf 离线判据，一致率 1.000"],
            ["机制是否真触发", "shield 触发 14 / 103 / 109 次；截止期提示 16–18 次；rho_safety 逐步落盘"],
            ["在线对照", "stl vs fixed_order：12 条配对 2 胜 2 负 6 平，p=1.00"],
            ["高度下限 6 m", "中位高度 8.76→6.09 m（裁剪到边界 = 贴着地板飞），SR .333→.167"],
            ["高度带 [7,12] m", "带内时间 37.4%→71.4%（干预生效），SR .250 vs .333"],
        ],
        col_widths=[2.6, 9.1], font_size=12.5,
    )
    textbox(s, 0.85, 5.75, 11.6, 1.3,
            "结论：STL 在这套 baseline 上的价值是「可验证的进度与裕度」（评价与诊断），\n"
            "而不是「直接提升成功率」；要提升成功率，抓手是障碍距离（深度图），不是高度。",
            size=13, color=GREY)

    # ---- 6 ----
    s = section(prs, "5 / idea3", "三次训练：从塌缩到治好，原因是初始化错了")
    table(
        s, 0.8, 1.75, 11.7,
        [
            ["#", "初始化", "数据 / lr / 步数", "不同动作数", "最高频动作", "SR"],
            ["v1", "openvla-7b 从零", "1k / 2e-4 / 125", "2 / 139", "98%", "0/10"],
            ["v3", "openvla-7b 从零", "2k / 2e-5 / 200", "10 / 105", "42%", "0/8"],
            ["v4", "AeroVLA checkpoint", "3k / 2e-5 / 200", "258 / 358", "4%", ".417"],
        ],
        col_widths=[0.7, 2.6, 2.9, 2.0, 1.7, 1.8], font_size=12.5,
    )
    textbox(s, 0.85, 3.7, 11.6, 3.0,
            "诊断链：loss 6.20→0.82 的同时策略变成常量输出 → 先怀疑 lr，降到 2e-5 仍然塌 →\n"
            "查「塌缩输出的到底是什么」：示范动作分布极度单峰（41% 样本 |fwd| 约 4.9），塌缩输出的正是这个众数 →\n"
            "真正的错在上游：从裸底座起步等于「用 2k 条样本重建一个 41 万条训出来的策略」。\n\n"
            "修正：加 --init-adapter 从 AeroVLA checkpoint 继续训 → 动作多样性 2→258，SR 0→.417。",
            size=13.5)

    # ---- 7 ----
    s = section(prs, "6 / 平台可信度", "用外部权重校准这把尺子：+8.3 pp vs 论文 +8.37 pp")
    if os.path.exists(FIG_ARMS):
        s.shapes.add_picture(FIG_ARMS, Inches(0.5), Inches(1.7), width=Inches(8.4))
    textbox(s, 9.15, 1.75, 3.6, 4.4,
            "RefineFly（同底座的后训练）\n"
            "12 条：SR .250 vs AeroVLA .167\n"
            "= +8.3 pp\n\n"
            "论文报告：+8.37 pp\n\n"
            "→ 我们的评测管线与文献口径一致，\n是一个可信的尺子。\n\n"
            "（N=12，p=1.00，不显著）",
            size=13)
    textbox(s, 0.85, 6.1, 11.6, 0.8,
            "同时注意：所有臂的 95% 置信区间都彼此重叠 → 本周任何「更好 / 更差」的 SR 结论都不成立。",
            size=13, color=GREY)

    # ---- 8 ----
    s = section(prs, "7 / 下周与决策", "下周计划 + 需要老师拍板的三件事")
    table(
        s, 0.8, 1.75, 6.5,
        [
            ["优先级", "事情"],
            ["P0", "起飞段净空保护（先爬升 + 深度近场净空 + 可达性检查）"],
            ["P0", "idea3 继续：Stage B 偏好对齐 + 扩到 N≥30 验证 +8.3 pp"],
            ["P1", "补统计：每臂 ≥30 条、同 split 重复 2–3 次"],
            ["P2", "扩展摘要（中文骨架已出）"],
        ],
        col_widths=[1.2, 5.3], font_size=12.5,
    )
    textbox(s, 7.6, 1.75, 5.2, 3.4,
            "需要拍板：\n"
            "① 指标：SR 被碰撞主导，主指标是否加「任务进度类」？\n"
            "② 碰撞：66% 是否说明仿真过苛刻？要不要报无碰撞子集？\n"
            "③ 算力：全量 Stage-A 需约 7 小时独占 GPU，能否申请夜间时段？",
            size=13)
    textbox(s, 0.85, 5.35, 11.6, 1.7,
            "诚实的自评：本周交付的是「平台 + 三个可信的负面结论 + 一个方向修正」，\n"
            "而不是「指标提升」。三个点都还需要一轮统计强度足够的实验，才能写成结论。",
            size=14, bold=True, color=BLUE)

    prs.save(OUT)
    print("wrote", OUT, "slides:", len(prs.slides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
