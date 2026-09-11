from __future__ import annotations

import csv
import html
import math
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "scientific_report"
PDF_OUT = ROOT / "YEAST_DIGITAL_TWIN_SCIENTIFIC_REPORT.pdf"
MD_OUT = ROOT / "YEAST_DIGITAL_TWIN_SCIENTIFIC_REPORT.md"
TEX_OUT = ROOT / "YEAST_DIGITAL_TWIN_SCIENTIFIC_REPORT.tex"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


wall = read_csv("data/simulated_dbtl_wallclock_final_table_moderate.csv")
claim_mod = read_csv("data/simulated_dbtl_final_claim_audit_moderate.csv").iloc[0]
claim_hard = read_csv("data/simulated_dbtl_final_claim_audit_hard.csv").iloc[0]
completion = read_csv("data/simulated_dbtl_hpc_completion_audit.csv").iloc[0]
pilot = read_csv("data/simulated_dbtl_exact_pilot_summary.csv").iloc[0]
state_conclusion = read_csv("data/gem_state_space_diagnostic_conclusion.csv").iloc[0]
reporter_frontier = read_csv("data/reporter_quality_frontier.csv")
design_accept = read_csv("data/design_benchmark_acceptance.csv")

state_metrics = read_csv("data/gem_state_space_metrics.csv")
design_regret = read_csv("data/design_benchmark_regret.csv")
wall_summary = read_csv("data/simulated_dbtl_wallclock_summary_moderate.csv")
best_found_summary = read_csv("data/simulated_dbtl_fixed_budget_best_found_summary.csv")
sensor_proxy_deltas = read_csv("data/simulated_dbtl_sensor_assimilation_proxy_deltas.csv")
sensor_flow = read_csv("data/simulated_dbtl_sensor_information_flow.csv")


def summary_row(workflow_id: str, axis: str, budget: float) -> pd.Series:
    sub = best_found_summary[
        best_found_summary["workflow_id"].eq(workflow_id)
        & best_found_summary["budget_axis"].eq(axis)
        & best_found_summary["budget"].astype(float).eq(float(budget))
    ]
    if sub.empty:
        raise KeyError(f"Missing fixed-budget summary row: {workflow_id}, {axis}, {budget}")
    return sub.iloc[0]


conv_wall = wall[wall["workflow_id"].eq("conventional_dbtl")].iloc[0]
hybrid_wall = wall[wall["workflow_id"].eq("biosensor_informed_modular_hybrid__full_reporters")].iloc[0]
stage_reduction = (conv_wall.mean_sequential_biological_stages - hybrid_wall.mean_sequential_biological_stages) / conv_wall.mean_sequential_biological_stages
calendar_reduction = (conv_wall.mean_simulated_calendar_days_excluding_compute - hybrid_wall.mean_simulated_calendar_days_excluding_compute) / conv_wall.mean_simulated_calendar_days_excluding_compute
wallclock_speedup = conv_wall.mean_simulated_calendar_days_excluding_compute / hybrid_wall.mean_simulated_calendar_days_excluding_compute


def state_space_table() -> list[list[str]]:
    sub = state_metrics[state_metrics["split"].eq("heldout_combination")]
    rows = []
    mapping = {
        "polynomial_environment_pca": "Polynomial env-to-PCA",
        "multi_output_mlp": "Multi-output MLP",
        "coordinate_conditioned_mlp": "Coordinate-conditioned MLP",
        "mechanistic_rate_state_space": "Mechanistic-rate state-space",
        "product_only_state_space": "Product-only state-space",
        "atp_reporter_state_space": "ATP reporter state-space",
        "oxidative_reporter_state_space": "Oxidative reporter state-space",
        "combined_reporter_state_space": "Combined reporter state-space",
        "oracle_state_capacity_state_space": "Oracle state/capacity labels",
    }
    for mid, label in mapping.items():
        part = sub[sub["model"].eq(mid)]
        if len(part):
            rows.append([label, f"{part['trajectory_normalized_rmse'].mean():.6f}"])
    rows.sort(key=lambda r: float(r[1]))
    return [["Model", "Held-out trajectory-normalized RMSE"]] + rows


def design_ranking_table() -> list[list[str]]:
    grouped = (
        design_regret.groupby("method_id")
        .agg(mean_exact_pool_regret=("exact_pool_regret", "mean"), mean_best_observed_objective=("best_verified_objective", "mean"))
        .reset_index()
        .sort_values("mean_exact_pool_regret")
    )
    labels = {
        "space_filling_parallel": "Space filling",
        "direct_trajectory_ucb": "Direct trajectory UCB",
        "hybrid_digital_twin_ucb": "Hybrid digital twin UCB",
        "black_box_bo": "Black-box BO",
        "random_parallel": "Random parallel",
        "static_gem_parallel": "Static GEM parallel",
    }
    rows = [["Rank", "Method", "Mean exact-pool regret", "Mean best observed objective"]]
    for i, row in enumerate(grouped.itertuples(index=False), start=1):
        rows.append([str(i), labels.get(row.method_id, row.method_id), f"{row.mean_exact_pool_regret:.6f}", f"{row.mean_best_observed_objective:.6f}"])
    return rows


workflow_labels = {
    "conventional_dbtl": "Conventional DBTL",
    "bayesian_optimization": "Bayesian optimisation",
    "black_box_digital_twin": "Black-box twin",
    "modular_hybrid_no_biosensors": "Modular hybrid, no biosensors",
    "biosensor_informed_modular_hybrid__full_reporters": "Biosensor modular hybrid",
    "biosensor_informed_modular_hybrid": "Biosensor modular hybrid",
}


def wallclock_table() -> list[list[str]]:
    rows = [["Workflow", "Target", "Cultures", "Stages", "Days", "Quality", "Speedup"]]
    for row in wall.itertuples(index=False):
        rows.append([
            row.workflow_label,
            f"{row.target_attainment:.3f}",
            f"{row.mean_total_physical_cultures:.2f}",
            f"{row.mean_sequential_biological_stages:.2f}",
            f"{row.mean_simulated_calendar_days_excluding_compute:.1f}",
            f"{row.mean_best_verified_quality:.3f}",
            f"{row.mean_wall_clock_speedup_vs_conventional:.2f}x",
        ])
    return rows


def core_wallclock_table() -> list[list[str]]:
    return [
        ["Metric", "Conventional DBTL", "Hybrid digital twin"],
        ["Moderate target attainment", f"{conv_wall.target_attainment:.3f}", f"{hybrid_wall.target_attainment:.3f}"],
        ["Mean physical cultures", f"{conv_wall.mean_total_physical_cultures:.2f}", f"{hybrid_wall.mean_total_physical_cultures:.2f}"],
        ["Mean sequential biological stages", f"{conv_wall.mean_sequential_biological_stages:.2f}", f"{hybrid_wall.mean_sequential_biological_stages:.2f}"],
        ["Scenario calendar days excluding compute", f"{conv_wall.mean_simulated_calendar_days_excluding_compute:.1f}", f"{hybrid_wall.mean_simulated_calendar_days_excluding_compute:.1f}"],
        ["Mean best verified quality", f"{conv_wall.mean_best_verified_quality:.3f}", f"{hybrid_wall.mean_best_verified_quality:.3f}"],
        ["Stage reduction", "reference", f"{100 * stage_reduction:.1f}%"],
        ["Calendar-time reduction", "reference", f"{100 * calendar_reduction:.1f}%"],
        ["Wall-clock speedup", "1.00x", f"{wallclock_speedup:.2f}x"],
    ]


def fixed_budget_quality_table() -> list[list[str]]:
    workflows = [
        ("conventional_dbtl", "Conventional DBTL"),
        ("biosensor_informed_modular_hybrid__full_reporters", "Hybrid digital twin"),
        ("biosensor_informed_modular_hybrid__atp_only", "ATP-only hybrid variant"),
        ("black_box_digital_twin", "Black-box twin control"),
    ]
    rows = [["Workflow", "24-culture best utility", "2-stage best utility", "18-day best utility", "Target attainment at 24 cultures"]]
    for wid, label in workflows:
        phys = summary_row(wid, "physical_cultures", 24)
        stages = summary_row(wid, "sequential_biological_stages", 2)
        days = summary_row(wid, "scenario_calendar_days", 18.0)
        rows.append([
            label,
            f"{phys.mean_best_verified_utility:.4f}",
            f"{stages.mean_best_verified_utility:.4f}",
            f"{days.mean_best_verified_utility:.4f}",
            f"{phys.target_attainment_probability:.3f}",
        ])
    return rows


def reporter_information_table() -> list[list[str]]:
    bio = sensor_flow[sensor_flow["method_id"].eq("biosensor_informed_modular_hybrid")].iloc[0]
    pathway = sensor_proxy_deltas[sensor_proxy_deltas["reporter_test"].eq("pathway_reporter")]
    full = sensor_proxy_deltas[sensor_proxy_deltas["reporter_test"].eq("full_reporters")]
    pathway_delta = pathway[["delta_utility_rmse_vs_nominal", "delta_utility_spearman_vs_nominal", "delta_top20_recovery_vs_nominal"]].mean()
    full_delta = full[["delta_utility_rmse_vs_nominal", "delta_utility_spearman_vs_nominal", "delta_top20_recovery_vs_nominal"]].mean()
    return [
        ["Question", "Answer from final cache"],
        ["Were reporters allowed for the biosensor method?", "Yes"],
        ["Were exact reporters available during virtual search?", "No"],
        ["Did exact reporters update regulatory state online?", "No"],
        ["Did later candidate selection use previous exact reporters?", "No"],
        ["Were early reporter time series saved?", "No; reporter files contain one culture-level row"],
        ["Pathway reporter proxy delta", f"RMSE {pathway_delta.delta_utility_rmse_vs_nominal:.4f}; Spearman +{pathway_delta.delta_utility_spearman_vs_nominal:.4f}; top-20 +{pathway_delta.delta_top20_recovery_vs_nominal:.4f}"],
        ["Full reporter proxy delta", f"RMSE {full_delta.delta_utility_rmse_vs_nominal:.4f}; Spearman +{full_delta.delta_utility_spearman_vs_nominal:.4f}; top-20 +{full_delta.delta_top20_recovery_vs_nominal:.4f}"],
    ]


def reporter_summary_table() -> list[list[str]]:
    sub = reporter_frontier[reporter_frontier["split"].eq("heldout_combination")]
    grouped = (
        sub.groupby("quality_condition")
        .agg(product_rmse=("product_rmse", "mean"), paired=("paired_product_only_rmse", "mean"), improvement=("relative_improvement", "mean"))
        .reset_index()
        .sort_values("product_rmse")
    )
    rows = [["Reporter condition", "Product RMSE", "Paired product-only RMSE", "Relative improvement"]]
    for r in grouped.itertuples(index=False):
        rows.append([str(r.quality_condition), f"{r.product_rmse:.4f}", f"{r.paired:.4f}", f"{r.improvement:.3f}"])
    return rows


strict_table = [
    ["Target", "Conventional target attainment", "Hybrid target attainment", "Hybrid median cultures", "Hybrid median rounds", "Cultures saved vs conventional", "Quality delta vs conventional", "Outcome"],
    ["Moderate", f"{claim_mod.conventional_target_attainment_probability:.3f}", f"{claim_mod.hybrid_target_attainment_probability:.3f}", f"{claim_mod.hybrid_median_cultures_to_target_censored:.1f}", f"{claim_mod.hybrid_median_rounds_to_target_censored:.1f}", f"{claim_mod.hybrid_vs_conventional_mean_cultures_saved:.2f}", f"{claim_mod.hybrid_vs_conventional_quality_delta:.4f}", str(claim_mod.claim_outcome)],
    ["Hard", f"{claim_hard.conventional_target_attainment_probability:.3f}", f"{claim_hard.hybrid_target_attainment_probability:.3f}", f"{claim_hard.hybrid_median_cultures_to_target_censored:.1f}", f"{claim_hard.hybrid_median_rounds_to_target_censored:.1f}", f"{claim_hard.hybrid_vs_conventional_mean_cultures_saved:.2f}", f"{claim_hard.hybrid_vs_conventional_quality_delta:.4f}", str(claim_hard.claim_outcome)],
]


def fig_textbox(title: str, items: Sequence[str], footer: str | None = None) -> Drawing:
    d = Drawing(420, 205)
    d.add(Rect(0, 0, 420, 205, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(16, 180, title, fontName="Helvetica-Bold", fontSize=13, fillColor=colors.HexColor("#1f2937")))
    y = 154
    for item in items:
        d.add(Rect(18, y - 10, 10, 10, fillColor=colors.HexColor("#dbeafe"), strokeColor=colors.HexColor("#60a5fa")))
        d.add(String(36, y - 8, item, fontName="Helvetica", fontSize=8.6, fillColor=colors.HexColor("#111827")))
        y -= 22
    if footer:
        d.add(Line(16, 32, 404, 32, strokeColor=colors.HexColor("#d1d5db")))
        d.add(String(16, 16, footer, fontName="Helvetica-Oblique", fontSize=8.5, fillColor=colors.HexColor("#374151")))
    return d


def fig_flow(title: str, boxes: Sequence[str], footer: str | None = None) -> Drawing:
    width, height = 420, 198
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(15, 173, title, fontName="Helvetica-Bold", fontSize=12.5, fillColor=colors.HexColor("#111827")))
    x0, y0, bw, bh, gap = 18, 131, 118, 28, 18
    for i, text in enumerate(boxes):
        row = i // 3
        col = i % 3
        x = x0 + col * (bw + gap)
        y = y0 - row * 48
        d.add(Rect(x, y, bw, bh, rx=4, ry=4, fillColor=colors.HexColor("#eef2ff"), strokeColor=colors.HexColor("#64748b")))
        for j, line in enumerate(textwrap.wrap(text, 22)[:2]):
            d.add(String(x + 6, y + 17 - 10 * j, line, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#111827")))
        if i < len(boxes) - 1:
            if col < 2:
                d.add(Line(x + bw, y + bh / 2, x + bw + gap - 3, y + bh / 2, strokeColor=colors.HexColor("#64748b")))
            else:
                d.add(Line(x + bw / 2, y, x + bw / 2, y - 20, strokeColor=colors.HexColor("#64748b")))
    if footer:
        d.add(String(18, 7, footer[:110], fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#374151")))
    return d


def fig_bar(title: str, labels: Sequence[str], values: Sequence[float], ylabel: str, highlight: int | None = None) -> Drawing:
    d = Drawing(420, 260)
    d.add(Rect(0, 0, 420, 260, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(18, 236, title, fontName="Helvetica-Bold", fontSize=12.5, fillColor=colors.HexColor("#111827")))
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 65
    chart.height = 140
    chart.width = 320
    chart.data = [list(values)]
    chart.categoryAxis.categoryNames = list(labels)
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.categoryAxis.labels.dx = -2
    chart.categoryAxis.labels.dy = -4
    chart.categoryAxis.labels.angle = 35
    chart.categoryAxis.labels.fontSize = 6.5
    chart.valueAxis.valueMin = 0
    vmax = max(values) if values else 1
    chart.valueAxis.valueMax = math.ceil(vmax * 1.2 * 10) / 10
    chart.valueAxis.valueStep = max(chart.valueAxis.valueMax / 4, 0.1)
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#60a5fa")
    d.add(chart)
    if highlight is not None and 0 <= highlight < len(values):
        pass
    d.add(String(18, 216, ylabel, fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#374151")))
    return d


def fig_pareto() -> Drawing:
    d = Drawing(420, 270)
    d.add(Rect(0, 0, 420, 270, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(18, 246, "Cultures versus sequential biological stages", fontName="Helvetica-Bold", fontSize=12.5))
    x0, y0, w, h = 55, 55, 310, 165
    d.add(Line(x0, y0, x0 + w, y0, strokeColor=colors.black))
    d.add(Line(x0, y0, x0, y0 + h, strokeColor=colors.black))
    maxx = wall["mean_total_physical_cultures"].max() * 1.08
    maxy = wall["mean_sequential_biological_stages"].max() * 1.12
    minx = wall["mean_total_physical_cultures"].min() * 0.9
    short = {
        "conventional_dbtl": "Conventional",
        "bayesian_optimization": "BO",
        "black_box_digital_twin": "Black-box",
        "modular_hybrid_no_biosensors": "Hybrid no biosensor",
        "biosensor_informed_modular_hybrid__full_reporters": "Biosensor hybrid",
    }
    offsets = {
        "conventional_dbtl": (6, 5),
        "bayesian_optimization": (6, 5),
        "black_box_digital_twin": (6, 8),
        "modular_hybrid_no_biosensors": (6, -2),
        "biosensor_informed_modular_hybrid__full_reporters": (6, -12),
    }
    for row in wall.itertuples(index=False):
        x = x0 + (row.mean_total_physical_cultures - minx) / (maxx - minx) * w
        y = y0 + row.mean_sequential_biological_stages / maxy * h
        col = colors.HexColor("#2563eb") if "hybrid" in row.workflow_id else colors.HexColor("#6b7280")
        d.add(Rect(x - 4, y - 4, 8, 8, fillColor=col, strokeColor=col))
        dx, dy = offsets.get(row.workflow_id, (6, 3))
        d.add(String(x + dx, y + dy, short.get(row.workflow_id, row.workflow_label), fontName="Helvetica", fontSize=7))
    d.add(String(x0 + 95, 22, "Mean total physical cultures", fontName="Helvetica", fontSize=8))
    d.add(String(8, y0 + 75, "Stages", fontName="Helvetica", fontSize=8))
    return d


def fig_dependency_graph() -> Drawing:
    d = Drawing(420, 260)
    d.add(Rect(0, 0, 420, 260, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(18, 236, "Sequential dependency graph", fontName="Helvetica-Bold", fontSize=12.5))
    d.add(String(22, 216, "Conventional DBTL: dependent rounds", fontName="Helvetica-Bold", fontSize=9))
    conv_labels = ["Design", "Build/Test", "Learn", "Redesign", "Build/Test", "Learn"]
    x, y = 20, 178
    for i, label in enumerate(conv_labels):
        xx = x + i * 63
        d.add(Rect(xx, y, 54, 28, rx=4, ry=4, fillColor=colors.HexColor("#fef3c7"), strokeColor=colors.HexColor("#b45309")))
        d.add(String(xx + 5, y + 11, label, fontName="Helvetica", fontSize=7))
        if i < len(conv_labels) - 1:
            d.add(Line(xx + 54, y + 14, xx + 63, y + 14, strokeColor=colors.HexColor("#92400e")))
    d.add(String(22, 134, "Hybrid digital twin: calibration, computation, parallel verification", fontName="Helvetica-Bold", fontSize=9))
    hybrid = [("Parallel calibration", 22, 92, 105), ("Train / virtual search", 154, 92, 112), ("Parallel verification", 296, 92, 105)]
    for i, (label, xx, yy, ww) in enumerate(hybrid):
        d.add(Rect(xx, yy, ww, 34, rx=4, ry=4, fillColor=colors.HexColor("#dbeafe"), strokeColor=colors.HexColor("#2563eb")))
        for j, line in enumerate(textwrap.wrap(label, 20)):
            d.add(String(xx + 8, yy + 22 - 10 * j, line, fontName="Helvetica", fontSize=7.5))
        if i < len(hybrid) - 1:
            d.add(Line(xx + ww, yy + 17, hybrid[i + 1][1], yy + 17, strokeColor=colors.HexColor("#1d4ed8")))
    d.add(String(22, 52, "Dependency audit: post-calibration hybrid proposals did not depend on newly revealed exact outcomes.", fontName="Helvetica-Oblique", fontSize=8))
    d.add(String(22, 34, "Therefore verification can be collapsed into one parallel stage without future-information leakage.", fontName="Helvetica-Oblique", fontSize=8))
    return d


def fig_best_utility_support() -> Drawing:
    labels = ["Hybrid 2 stages", "Conv. 2 stages", "Hybrid 18 days", "Conv. 18 days"]
    values = [
        summary_row("biosensor_informed_modular_hybrid__full_reporters", "sequential_biological_stages", 2).mean_best_verified_utility,
        summary_row("conventional_dbtl", "sequential_biological_stages", 2).mean_best_verified_utility,
        summary_row("biosensor_informed_modular_hybrid__full_reporters", "scenario_calendar_days", 18.0).mean_best_verified_utility,
        summary_row("conventional_dbtl", "scenario_calendar_days", 18.0).mean_best_verified_utility,
    ]
    return fig_bar("Best verified utility at matched depth/time", labels, values, "Mean best utility")


def fig_hypotheses() -> Drawing:
    rows = [
        ("Latent state improves prediction", "Synthetic only", "#fbbf24"),
        ("Mechanistic realism makes state-space necessary", "Not supported", "#f87171"),
        ("Learned interface can drive Yeast9", "Technically supported", "#34d399"),
        ("Hybrid reduces total cultures", "Not supported", "#f87171"),
        ("Hybrid reduces sequential depth", "Supported with tradeoffs", "#34d399"),
        ("Live biosensors drive final result", "Not tested", "#fbbf24"),
    ]
    d = Drawing(420, 190)
    d.add(Rect(0, 0, 420, 190, fillColor=colors.white, strokeColor=colors.lightgrey))
    d.add(String(18, 168, "Final hypothesis audit", fontName="Helvetica-Bold", fontSize=12.5))
    y = 138
    for question, status, col in rows:
        d.add(Rect(20, y - 8, 14, 14, fillColor=colors.HexColor(col), strokeColor=colors.HexColor("#374151")))
        d.add(String(44, y - 3, question, fontName="Helvetica", fontSize=8.5))
        d.add(String(300, y - 3, status, fontName="Helvetica-Bold", fontSize=8.5))
        y -= 24
    return d


figures: list[tuple[str, str, Drawing]] = []
figures.append(("fig01_architecture", "Overall digital-twin hypothesis and deployment architecture.", fig_flow("Proposed acceleration topology", ["Parallel calibration cultures", "Reporter-grounded physiological learning", "Compact regulation-metabolism interface", "Exact Yeast9 dynamic pFBA", "Large virtual strain x environment search", "Parallel verification batch"], "Acceleration is expected from replacing sequential feedback with computation plus parallel verification.")))
figures.append(("fig02_hypothesis_chain", "Scientific development and hypothesis chain.", fig_flow("How each result motivated the next question", ["Simple fixed-environment curves", "Hidden physiological state", "Reporter supervision", "State-machine metabolism", "Exact Yeast9 dynamic pFBA", "Teacher and hybrid interface", "Strain x environment design", "Complete simulated DBTL benchmark"], "Negative results narrowed the final claim from prediction superiority to deployment-stage acceleration.")))
rep = reporter_frontier[reporter_frontier["split"].eq("heldout_combination")].groupby("quality_condition")["product_rmse"].mean().sort_values()
figures.append(("fig03_reporter", "Synthetic reporter supervision on held-out combinations.", fig_bar("Synthetic reporter quality frontier", list(rep.index), list(rep.values), "Mean held-out product RMSE")))
figures.append(("fig04_yeast9", "Exact Yeast9 dynamic-pFBA simulator loop.", fig_flow("One simulated culture in the exact hidden wet lab", ["Apply environment and physiological constraints", "Maximize growth", "Preserve state-dependent growth fraction", "Maximize beta-carotene", "Run parsimonious FBA", "Update biomass, product, burdens and capacities"], "Each full exact culture task records 144 LP solves in the definitive benchmark.")))
st_rows = state_space_table()[1:7]
figures.append(("fig05_state_space", "Direct models versus state-space models on held-out real-GEM trajectories.", fig_bar("Direct models remained strongest", [r[0][:18] for r in st_rows], [float(r[1]) for r in st_rows], "Trajectory-normalized RMSE")))
figures.append(("fig06_hybrid", "Teacher to compact interface to Yeast9 hybrid architecture.", fig_flow("Hybrid regulation-metabolism coupling", ["Environment [T, pH, DO]", "Reporter-grounded teacher", "Distilled hybrid student", "Compact interface controls", "Fresh Yeast9 pFBA solves", "Product, biomass, fluxes"], "The primary hybrid predicts controls; Yeast9 produces the metabolic outcome.")))
dr = design_regret.groupby("method_id")["exact_pool_regret"].mean().sort_values()
figures.append(("fig07_finite_pool", "Exact finite-pool optimization comparison.", fig_bar("Exact finite-pool regret", [x.replace("_", " ")[:18] for x in dr.index], list(dr.values), "Mean exact-pool regret")))
figures.append(("fig08_definitive", "Definitive simulated DBTL benchmark design.", fig_textbox("Definitive exact hidden-wet-lab benchmark", ["6 hidden biological worlds", "10 campaign seeds", "5 workflows and 7 reporter conditions", "15,840 workflow culture requests", "13,920 unique exact Yeast9 tasks", "Strict method-access and culture accounting"], f"Protocol hash: {completion.protocol_hash[:16]}...")))
figures.append(("fig09_dependency", "Conventional DBTL versus hybrid digital-twin dependency graph.", fig_dependency_graph()))
figures.append(("fig10_stages", "Sequential biological stages by workflow.", fig_bar("Sequential biological depth", list(wall["workflow_label"]), list(wall["mean_sequential_biological_stages"]), "Mean stages")))
figures.append(("fig11_days", "Scenario calendar time by workflow.", fig_bar("Scenario calendar time", list(wall["workflow_label"]), list(wall["mean_simulated_calendar_days_excluding_compute"]), "Days excluding compute")))
figures.append(("fig12_pareto", "Total cultures versus sequential biological depth.", fig_pareto()))
figures.append(("fig13_best_utility", "Best verified utility at matched biological depth and calendar time.", fig_best_utility_support()))
figures.append(("fig14_hypotheses", "Supported, bounded, and unsupported hypotheses.", fig_hypotheses()))


def save_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, _caption, drawing in figures:
        renderPDF.drawToFile(drawing, str(FIG_DIR / f"{name}.pdf"))


@dataclass
class Section:
    title: str
    body: list[object]


@dataclass
class FigureRef:
    idx: int


@dataclass
class TableRef:
    title: str
    rows: list[list[str]]


@dataclass
class Equation:
    text: str
    label: str


def p(text: str) -> str:
    return textwrap.dedent(text).strip()


def bullets(items: Sequence[str]) -> list[str]:
    return list(items)


abstract = p(f"""
Bioprocess design-build-test-learn (DBTL) is constrained by sequential biological feedback: strain construction, cultivation, measurement, analysis, and redesign often must occur in ordered rounds even when individual cultures can be run in parallel. This project evaluated whether a biosensor-informed regulation-metabolism digital twin could accelerate simulated yeast beta-carotene DBTL by learning physiological regulation, coupling learned controls to mechanistic Yeast9 dynamic pFBA, performing large virtual strain-by-environment searches, and physically verifying selected designs in parallel.

The development proceeded through falsifiable stages. Synthetic fixed-environment experiments showed that reporter supervision can help when informative hidden physiological states are deliberately present, with held-out reporter-supervised product RMSE as low as 0.0047. However, increasing biological realism did not make latent state-space models superior predictors. In an expanded exact Yeast9 dynamic-capacity dataset, direct environment-to-trajectory models outperformed state-space variants; the best state-space versus best direct trajectory-normalized RMSE delta was +{state_conclusion.state_space_mean_delta_vs_direct:.6f}, with state-space win fraction {state_conclusion.state_space_win_fraction:.1f}. Exact finite-pool optimization also did not favor the hybrid: space filling had the lowest mean exact-pool regret.

The final validation therefore tested complete simulated DBTL workflows rather than prediction RMSE. A non-mock Yeast9 pilot and full Vanda run completed {int(completion.valid_tasks):,}/{int(completion.expected_tasks):,} exact hidden-wet-lab tasks with zero missing, failed, duplicate, hash-mismatch, or invalid outputs. The main comparison was conventional iterative DBTL versus a biosensor-informed modular hybrid digital twin that performs parallel calibration, virtual strain-by-environment exploration, and parallel final verification.

In strict cultures-to-target accounting, the hybrid did not reduce total physical cultures relative to conventional DBTL. For the moderate target, conventional DBTL attained target in {claim_mod.conventional_target_attainment_probability:.3f} of campaigns, while the canonical full-reporter hybrid attained {claim_mod.hybrid_target_attainment_probability:.3f}; mean cultures saved versus conventional was {claim_mod.hybrid_vs_conventional_mean_cultures_saved:.1f}. However, dependency auditing showed that post-calibration hybrid proposals did not require newly revealed exact wet-lab outcomes and could be collapsed into one parallel verification stage without future-information leakage. Under the stated high-parallel-capacity scenario, the hybrid reduced mean sequential biological depth from {conv_wall.mean_sequential_biological_stages:.2f} to {hybrid_wall.mean_sequential_biological_stages:.2f} stages and scenario calendar time from {conv_wall.mean_simulated_calendar_days_excluding_compute:.1f} to {hybrid_wall.mean_simulated_calendar_days_excluding_compute:.1f} days, a {100 * calendar_reduction:.1f}% simulated time reduction and {wallclock_speedup:.2f}x speedup. The strongest supported practical result is therefore reduced sequential experimental depth, not fewer total cultures.
""")


sections: list[Section] = [
    Section("Introduction", [
        p("""
        The engineering problem addressed here is not merely prediction of a product curve. In yeast bioprocess DBTL, useful design decisions often require repeated sequential cycles of strain construction, cultivation, measurement, analysis, and redesign. Even a laboratory that can run many cultures simultaneously is limited when the next designs cannot be selected until the current biological round is measured.

        The project therefore distinguishes total experimental breadth from sequential experimental depth. Breadth is the number of physical cultures consumed. Depth is the number of biological feedback stages that must occur in sequence. A digital twin may fail to reduce total cultures and still accelerate DBTL if it replaces later biological feedback rounds with computation and a larger parallel verification batch.
        """),
        Equation("T_DBTL approx N_stages T_culture + T_planning", "1"),
        Equation("T_twin approx T_calibration + T_compute + T_verification", "2"),
        p("""
        The proposed workflow uses an initial parallel culture batch to characterize physiology, uses biosensors to expose otherwise hidden regulatory state, learns a regulatory model, connects that model to mechanistic metabolism, performs many virtual strain-by-environment experiments, and verifies a selected batch in parallel. The central hypothesis refined over the project was that a digital twin may accelerate DBTL not necessarily by requiring fewer cultures, but by replacing sequential biological rounds with computational search and parallel verification.
        """),
        FigureRef(1),
    ]),
    Section("Overview of the Scientific Development", [
        p("""
        The scientific development is best read as a sequence of hypotheses tested and revised. The early synthetic experiments asked whether hidden physiology and reporters could improve trajectory prediction. The dFBA state-machine and real Yeast9 phases then tested whether increasing biological realism made latent state-space models necessary. Those experiments produced important negative results. The project therefore shifted from asking whether the hybrid was the best predictor to asking whether learned regulation could provide a compact interface into mechanistic metabolism and whether that interface could improve a complete DBTL workflow.
        """),
        FigureRef(2),
        TableRef("Major scientific questions and outcomes", [
            ["Question", "Expected outcome", "Observed outcome", "Consequence"],
            ["Do hidden states justify state-space models?", "Reporter-supervised latent models should generalize better.", "Supported only in deliberately synthetic systems.", "Move to biologically grounded generators."],
            ["Does mechanistic realism make latent dynamics necessary?", "State-space models should improve as simulator complexity rises.", "Not supported; direct models remained strongest.", "Reframe from prediction to digital-twin workflow."],
            ["Can learned regulation drive mechanistic metabolism?", "Compact controls should reproduce exact pFBA when correct.", "Oracle replay matched original GEM trajectory to numerical precision.", "Use compact interface in hybrid design."],
            ["Does hybrid search dominate finite-pool optimization?", "Hybrid UCB should beat classical baselines.", "Space filling had lowest exact-pool regret.", "Use deployment-style rather than equal-query benchmark."],
            ["Can the twin reduce sequential DBTL time?", "Virtual search should collapse biological feedback depth.", "Supported after wall-clock correction, with more cultures.", "Final bounded claim."],
        ]),
    ]),
    Section("Question 1: Does Hidden Physiological State Justify a State-Space Model?", [
        p("""
        The first question was whether hidden physiological state can improve prediction of production trajectories and whether biosensors make that state learnable. If different physiological states produce different dynamics under similar environmental conditions, explicitly learning latent state should improve held-out generalization. If reporters provide informative measurements of those hidden states, reporter-supervised models should outperform product-only models.

        The synthetic fixed-environment generator held one environment vector constant for each culture and generated a complete product trajectory through a dynamic product equation. Later synthetic variants introduced hidden physiological states and reporter channels. Splits were culture-level, including held-out environmental combinations, so no time point from a held-out culture appeared in training.
        """),
        Equation("dP/dt = F(t, e, z) - k_P P", "3"),
        Equation("R_k(t) = h_k(z(t)) + noise + lag", "4"),
        p("""
        The synthetic experiments supported the local reporter hypothesis. In the held-out-combination setting, the best reporter-supervised model in Experiment 2B reached product RMSE of approximately 0.0047. Reporter corruption controls showed that clean, sparse, missing, noisy, mild-lag, gain/offset, and combined conditions could remain useful by the predeclared improvement criterion, whereas slow lag was harmful. This established that reporter supervision can help when informative hidden state is deliberately present.

        The interpretation was bounded. Synthetic latent variables were constructed by the simulator, so recovering or exploiting them did not prove that comparable latent variables exist in yeast. The result motivated the next question: does the advantage survive when the production system becomes biologically grounded?
        """),
        FigureRef(3),
        TableRef("Reporter-quality frontier on held-out combinations", reporter_summary_table()),
    ]),
    Section("Question 2: Does Increasing Biological Realism Make Dynamic Latent Models Necessary?", [
        p("""
        The second hypothesis was that early direct models might have succeeded because the synthetic generator was too simple. If so, adding mechanistic metabolism and dynamic physiological constraints should make latent state more valuable. The project tested this first with a dFBA state-machine surrogate and then with exact Yeast9 dynamic pFBA.

        The state-machine generator mapped fixed temperature, pH, and dissolved oxygen to hidden oxidative, ATP, pathway, and bottleneck states. These states modified metabolic constraints, and product and biomass emerged from the constrained metabolic system. Direct models nevertheless remained competitive or superior, which motivated a generator audit rather than neural tuning.
        """),
        p("""
        The real Yeast9 transition made the simulator substantially more mechanistic. The selected Yeast9 asset loaded with approximately 4,131 reactions, 2,806 metabolites, and 1,161 genes. The beta-carotene augmentation reused native GGPP formation and added pathway reactions for phytoene synthase, phytoene desaturase, lycopene cyclase, and beta-carotene demand. In each interval of a simulated culture, the verifier applied environment and physiological constraints, solved maximum growth, preserved a state-dependent growth fraction, optimized beta-carotene production, ran parsimonious FBA, updated biomass and product, and used solved metabolic behavior to update physiological burdens and pathway capacities.
        """),
        FigureRef(4),
        p("""
        Dynamic pathway capacity states for PSY, DES, and CYC were introduced because the first real-GEM trajectories were flux-rich but product-shape simple. The selected dynamic-congestion run completed 3,888 real Yeast9 LP solves and changed pathway capacities substantially, but accumulated product remained highly compressible: in the full 49-point grid, amplitude-normalized product PC1 explained 99.8049% of variance. The final dynamic-capacity classification was `gem_capacity_feedback_too_weak`.

        The definitive direct-versus-state-space diagnostic used 125 fixed environments, 49 time points, 48 intervals, and 18,000 actual LP solves. Direct models led held-out-combination prediction. The best state-space versus best direct trajectory-normalized RMSE delta was +0.008895 state-space-minus-direct; the state-space win fraction was 0.0, with bootstrap upper confidence bound 0.011401. This rejected the stronger hypothesis that mechanistic realism would automatically make state-space models superior predictors.
        """),
        FigureRef(5),
        TableRef("Direct versus state-space diagnostic", state_space_table()),
    ]),
    Section("Question 3: Can a Modular Regulation-Metabolism Digital Twin Be Built?", [
        p("""
        The revised hypothesis was that learning physiology need not directly improve product prediction. Instead, a learned regulatory model could predict the compact metabolic constraints under which a mechanistic genome-scale model should operate. In this architecture, the primary hybrid does not simply use a neural network to predict final product. The learned model predicts regulatory and metabolic controls; Yeast9 produces the metabolic outcome.
        """),
        Equation("c(t) = g_theta(e, z(t));  v(t) = pFBA_Yeast9(c(t));  P(t+dt) = P(t) + dt f_product(v(t), X(t))", "5"),
        p("""
        The reporter-grounded teacher took environment as input and learned product, oxidative reporter, ATP reporter, pathway reporter, compact-interface, and selected flux-summary heads. Dense Latin-hypercube teacher pseudo-data were then generated inside the original support for distillation. The hybrid student predicted reporter channels, compact interface controls, and flux summaries from environment. Reporter information was used for training and declared reporter-condition workflows, but hidden simulator state and future exact outcomes were prohibited.

        Exact replay validated the compact interface. Oracle compact controls reproduced the original GEM trajectory to numerical precision in the smoke culture, with B_total RMSE 2.61e-16. The one-culture oracle/teacher/hybrid replay completed 432 LP stages with zero infeasible, unbounded, solver-error, skipped-interval, or surrogate counts. Preliminary completed-culture metrics showed hybrid-control pFBA close to teacher-control pFBA, but this architecture validation did not yet prove optimization or DBTL acceleration.
        """),
        FigureRef(6),
    ]),
    Section("Question 4: Can the Digital Twin Identify Better Strain x Environment Designs?", [
        p("""
        The next question moved from prediction to engineering. Strain and environment must be optimized jointly because pathway capacity, precursor supply, oxygen availability, ATP demand, growth allocation, and product accumulation interact. The design space therefore included competing sinks, precursor supply, PSY/DES/CYC capacities, ATP or cofactor support, oxygen support, export, glucose uptake, and product degradation or loss.

        The benchmark distinguished cheap virtual evaluations from exact Yeast9 verification. The exact finite-pool benchmark froze weak and strong dynamic-regulation regimes and evaluated random parallel search, space filling, static GEM ranking, black-box Bayesian optimization, direct trajectory UCB, and hybrid digital twin UCB against a declared finite exact pool.
        """),
        p("""
        The result was another important negative finding. All 96 finite-pool candidates completed exact dynamic pFBA, consuming 13,824 actual LP solves with zero solver errors or surrogate evaluations. Space filling had the lowest mean exact-pool regret, followed by direct trajectory UCB and then hybrid digital twin UCB. The hybrid architecture therefore did not automatically create a superior optimizer in a small bounded candidate pool.
        """),
        FigureRef(7),
        TableRef("Exact finite-pool optimization ranking", design_ranking_table()),
    ]),
    Section("Question 5: Can a Digital Twin Accelerate the Complete DBTL Workflow?", [
        p("""
        Equal-query optimization is not the natural deployment model for a digital twin. A deployed twin is useful because virtual experiments are cheap and can be run at enormous scale. The operational hypothesis became: perform physical calibration, run many virtual experiments, and verify a small physical batch.

        This question is fundamentally about time and dependency. Conventional DBTL is iterative: design, build, test, learn, redesign, build, test, and learn again. Even if each batch contains many parallel cultures, the next batch cannot be chosen until the previous biological batch has completed and been measured. The hybrid deployment is structurally different: parallel calibration cultures, train or calibrate the twin, large virtual strain x environment exploration, and parallel final verification.

        Intermediate deployment-style experiments showed both promise and failure. Cached deployment was operationally cheaper but quality-inferior. Open-ended scientist-versus-hybrid campaigns were mixed: the scientist often found the best individual design, while the hybrid often had stronger mean batch quality. Readiness audits also found effective-phenotype collapse, where distinct requested edits mapped to repeated phenotypes. Those failures motivated intervention-space redesign and a definitive benchmark with explicit culture accounting.
        """),
    ]),
    Section("Definitive Simulated DBTL Benchmark", [
        p(f"""
        The definitive benchmark treated exact Yeast9 dynamic pFBA as a hidden wet lab. It froze six biological worlds, ten campaign seeds, hidden culture-to-culture physiology, seven reporter conditions, five workflows, physical culture accounting, virtual evaluation accounting, and strict method-access boundaries. The five workflows were conventional DBTL, Bayesian optimisation, black-box digital twin, modular hybrid without biosensors, and biosensor-informed modular hybrid.

        The prepared full manifest contained 15,840 workflow culture requests and 13,920 unique exact hidden-wet-lab tasks. The accepted Vanda exact pilot covered 18 tasks across six worlds, three campaign seeds, all five methods, and all seven reporter conditions. It passed all acceptance gates: complete exact dynamic pFBA status, non-mock `yeast_gem_lp` backend, 144 LP solves per task, zero solver errors, correct physical-culture charge, reporter consistency, latent variability, and no hidden-state or outcome access. Mean pilot runtime was {pilot.mean_wall_time_seconds:.3f} seconds.

        The full execution completed {int(completion.completed_tasks):,}/{int(completion.expected_tasks):,} exact tasks. The strict local audit found {int(completion.valid_tasks):,} valid tasks and zero missing, failed, duplicate, hash-mismatch, or invalid outputs. This established that the final result is an exact simulated wet-lab result rather than a surrogate analysis.

        The purpose of the multiple methods and reporter branches was benchmark rigor. The main engineering comparison in the definitive result is conventional iterative DBTL versus the biosensor-informed modular hybrid digital twin. Black-box and no-biosensor workflows remain important controls, but they do not define the primary deployment claim.
        """),
        FigureRef(8),
        TableRef("Definitive workflow definitions", [
            ["Workflow", "Role in benchmark", "Physical/virtual distinction"],
            ["Conventional DBTL", "Adaptive public DBTL baseline", "Physical cultures drive later choices"],
            ["Bayesian optimisation", "Outcome-only adaptive optimizer", "Physical observations update BO"],
            ["Black-box digital twin", "Virtual search without modular interface", "Virtual evaluations are not cultures"],
            ["Modular hybrid, no biosensors", "Regulation-metabolism interface without reporter benefit", "Parallel post-calibration verification"],
            ["Biosensor modular hybrid", "Canonical full-reporter twin", "Reporter-informed virtual search plus exact verification"],
        ]),
    ]),
    Section("Result 1: Does the Digital Twin Reduce Total Physical Experiments?", [
        p("""
        The first analysis asked a resource-efficiency question: how many physical cultures are required before the engineering threshold is reached? This strict cultures-to-target endpoint did not support total physical-culture reduction versus conventional DBTL. For the moderate target, conventional DBTL attained target in 0.533 of campaigns, while the canonical full-reporter hybrid attained 0.483. The hybrid median censored cultures-to-target was 25.0 and median censored rounds-to-target was 4.0; mean cultures saved versus conventional was -0.3. For the hard target, the hybrid attained 0.150 versus 0.367 for conventional and saved -1.9 cultures versus conventional.

        The quality result was different: mean best-utility delta versus conventional was small and positive at 0.003432. Thus, the strong claim that the digital twin reduces total physical culture consumption while maintaining quality was not supported. The strict outcome was `NEUTRAL`, not a hybrid win. This result remains important because total cultures are real resources; the later wall-clock analysis does not erase that cost.
        """),
        TableRef("Strict culture-efficiency result", strict_table),
    ]),
    Section("Experimental Breadth Versus Sequential Depth", [
        p("""
        The strict cultures-to-target metric did not fully answer the original engineering question because the project was motivated by time. It treats many cultures in one parallel batch too similarly to the same number of cultures spread across several dependent rounds. The final analysis therefore separated experimental breadth from experimental depth.

        Experimental breadth is the total number of physical cultures run. Experimental depth is the number of sequential biological stages whose outcomes must be observed before the next decisions can be made. One hundred cultures run simultaneously can represent approximately one biological culture-duration, whereas twenty cultures distributed across four dependent DBTL rounds can require approximately four biological culture-durations. This distinction assumes the stated high-parallel-capacity laboratory scenario; it should not be read as unlimited laboratory capacity.

        The workflow histories were audited to determine whether later digital-twin proposal batches genuinely depended on previous exact wet-lab outcomes. The manifest-generation logic used placeholder observations for digital-twin proposal batches, not newly revealed exact outcomes. Therefore later hybrid proposals could have been generated before the first verification batch completed and can be collapsed into one parallel post-calibration verification stage without future-information leakage. Conventional DBTL remains genuinely adaptive because later design decisions depend on previous physical culture outcomes.
        """),
        FigureRef(9),
    ]),
    Section("Main Result: Parallel Wall-Clock Acceleration", [
        p("""
        The corrected wall-clock analysis asked whether virtual strain x environment experimentation could replace sequential biological feedback with computation and parallel verification. This is the primary result of the final exact benchmark.
        """),
        Equation("Speedup = T_conventional / T_twin", "6"),
        p(f"""
        Conventional DBTL used a mean {conv_wall.mean_total_physical_cultures:.2f} cultures, {conv_wall.mean_sequential_biological_stages:.2f} sequential biological stages, and {conv_wall.mean_simulated_calendar_days_excluding_compute:.1f} scenario days excluding compute. The biosensor-informed modular hybrid used more cultures, {hybrid_wall.mean_total_physical_cultures:.2f}, but only {hybrid_wall.mean_sequential_biological_stages:.2f} sequential biological stages and {hybrid_wall.mean_simulated_calendar_days_excluding_compute:.1f} scenario days. This is a {100 * stage_reduction:.1f}% reduction in sequential biological depth, a {100 * calendar_reduction:.1f}% reduction in simulated calendar time, and a {wallclock_speedup:.2f}x wall-clock speedup.

        The central final result is therefore topological. The hybrid digital twin did not accelerate DBTL by using fewer total cultures, finding dramatically higher-utility designs, or proving every architectural component superior. It accelerated DBTL by converting sequential biological feedback depth into parallel experimental breadth and virtual computation.
        """),
        FigureRef(10),
        FigureRef(11),
        FigureRef(12),
        TableRef("Conventional DBTL versus hybrid digital-twin wall-clock result", core_wallclock_table()),
    ]),
    Section("Supporting Result: Best-Found Utility At Fixed Budget", [
        p("""
        A cache-only follow-up asked whether the accelerated workflow produced much worse engineering designs when the same budget was spent. This is a supporting endpoint, not a replacement for the time-to-target analysis. At the full matched 24-culture budget, conventional DBTL reached mean best verified utility 1.8808 and the full-reporter hybrid reached 1.8842; the ATP-only hybrid variant reached 1.8861, and the black-box twin control reached 1.8641. These values indicate that the wall-clock acceleration was not achieved by accepting a dramatically poorer final solution.

        The same fixed-budget analysis reinforces the depth mechanism. At two biological stages, the full-reporter hybrid reached mean best utility 1.8842, while conventional DBTL reached 1.6942. At approximately 18 scenario days, the hybrid again reached 1.8842, while conventional DBTL had only completed the initial stage and reached 1.2214. These are fixed-depth and fixed-time comparisons within the frozen evaluated universe; they do not prove global optimality.
        """),
        FigureRef(13),
        TableRef("Fixed-budget best-found utility support", fixed_budget_quality_table()),
    ]),
    Section("Reporter And Architecture Controls In The Final Benchmark", [
        p("""
        Physiological reporter branches were included in the definitive benchmark, but the final workflow did not perform live time-resolved reporter assimilation. Exact reporter values were not available during virtual search, did not update the regulatory state online, and did not drive later candidate selection from previous physical cultures. Consequently, the final benchmark should not be used as a definitive test of the independent value of live biosensors.

        A bounded proxy analysis found that culture-level pathway-reporter summaries contained some additional information about eventual utility: pathway reporters improved utility RMSE by approximately 0.0194, Spearman ranking by approximately 0.0487, and top-20% recovery by approximately 0.0601 relative to nominal design-only prediction. Full reporters showed smaller positive signals. This is preliminary evidence that physiological measurements can contain useful additional information, but proper live early-time reporter assimilation remains future work.

        Additional digital-twin control architectures showed similar stage-depth behavior, indicating that the present benchmark primarily validates the deployment advantage of virtual experimentation rather than a unique performance advantage of the modular architecture. This does not invalidate the hybrid result, but it bounds the architecture-specific claim.
        """),
        TableRef("Reporter information regime in the final benchmark", reporter_information_table()),
    ]),
    Section("Discussion", [
        p("""
        The project progressively rejected stronger hypotheses and arrived at a narrower operational claim. Hypothesis 1, that latent state improves product prediction, was supported in simple synthetic systems but not in the real-GEM benchmark. Hypothesis 2, that increasing mechanistic realism would make state-space modeling advantageous, was not supported; accumulated product trajectories remained highly compressible. Hypothesis 3, that a learned regulatory interface can drive mechanistic metabolism, was technically supported by exact replay. Hypothesis 4, that hybrid optimization should require fewer cultures, was not supported against conventional DBTL. Hypothesis 5, that a digital twin can reduce sequential DBTL time by converting biological feedback rounds into virtual search and parallel verification, was supported under the corrected deployment assumptions, with tradeoffs.

        What was not shown is as important as what was shown. The project did not show universal state-space superiority, fewer total physical cultures, universal reporter advantage, or unique hybrid superiority over every digital surrogate. The final exact benchmark also did not test live early-time reporter assimilation, so biosensor-specific conclusions must remain bounded.

        What was shown is a technically viable regulation-metabolism hybrid, exact Yeast9 hidden-wet-lab benchmarking with complete solver and cache accounting, joint strain x environment virtual experimentation, and a plausible reduction in sequential DBTL wall-clock time under a stated high-parallel-capacity deployment scenario. The final positive result appears only after several easier positive stories failed. It is therefore best understood as a workflow result: the digital twin changes the dependency graph of experimentation.
        """),
        FigureRef(14),
        TableRef("Supported, bounded, and unsupported claims", [
            ["Claim", "Status", "Evidence"],
            ["Reporter supervision can help hidden-state trajectory prediction.", "Supported synthetically", "Experiment 2B and reporter-quality frontier."],
            ["State-space models are best for real-GEM product prediction.", "Unsupported", "State-space win fraction 0.0 in diagnostic benchmark."],
            ["Compact learned controls can drive exact Yeast9 replay.", "Supported technically", "Oracle replay B_total RMSE 2.61e-16."],
            ["Hybrid search dominates exact finite-pool optimization.", "Unsupported", "Space filling had lowest mean exact-pool regret."],
            ["Hybrid reduces total physical cultures versus conventional DBTL.", "Unsupported", "Moderate cultures saved versus conventional -0.3."],
            ["Hybrid reduces sequential biological depth under parallel deployment.", "Supported with tradeoffs", "3.20 to 1.97 stages; 31.0 to 17.7 days."],
            ["Live biosensor assimilation drives the final DBTL result.", "Not tested definitively", "No time-resolved exact reporter assimilation in candidate selection."],
        ]),
    ]),
    Section("Limitations", [
        bullets([
            "The validation is entirely computational; Yeast9 dynamic pFBA is still a model, not a real culture.",
            "Hidden biological worlds and culture-to-culture variability are simulated.",
            "Calendar-time numbers depend on the specified culture-duration and operational scenario assumptions.",
            "The wall-clock speedup assumes sufficient parallel culture capacity for calibration and verification batches.",
            "The twin used slightly more total cultures and had slightly lower target-attainment probability than conventional DBTL in the moderate target analysis.",
            "The final benchmark did not perform live time-resolved reporter assimilation, so biosensor-specific deployment value remains unresolved.",
            "Modularity did not clearly outperform black-box twins on the headline stage-depth metrics.",
            "Product trajectories were often low-dimensional, making direct baselines strong.",
            "Intervention-space actionability was a recurring challenge before final redesign.",
            "No physical engineered yeast validation has yet been performed.",
        ]),
    ]),
    Section("Conclusion", [
        p("""
        This work investigated whether a hybrid regulation-metabolism digital twin could accelerate yeast bioprocess DBTL. Several stronger hypotheses were rejected: state-space models were not consistently better predictors, reporters were not universally beneficial, the hybrid did not dominate classical optimization, and total culture reduction was not demonstrated against conventional DBTL.

        The final exact simulated DBTL benchmark nevertheless revealed a different operational advantage. Under the stated high-parallel-capacity scenario, the hybrid digital-twin workflow reduced mean biological stage depth from 3.20 to 1.97 stages and scenario calendar time from 31.0 to 17.7 days, corresponding to a 42.9% simulated time reduction and a 1.75x wall-clock speedup. It did so while using more total cultures, not fewer.

        The strongest supported practical claim is therefore that virtual strain x environment experimentation can reduce sequential biological DBTL depth by replacing dependent physical feedback rounds with computation and parallel verification. Physical yeast validation, practical laboratory parallelization, and the independent value of live biosensor assimilation remain future questions.
        """),
    ]),
]


class NumberedCanvas:
    def __init__(self, canvas, doc):
        self.canvas = canvas
        self.doc = doc


class DocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == "Heading1":
                self.notify("TOCEntry", (0, text, self.page))
            elif style == "Heading2":
                self.notify("TOCEntry", (1, text, self.page))


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleMain", parent=styles["Title"], fontName="Times-Bold", fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Times-Italic", fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle("Abstract", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.3, leading=12, alignment=TA_JUSTIFY, spaceAfter=8))
    styles.add(ParagraphStyle("BodyJustify", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.6, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontName="Times-Italic", fontSize=8.4, leading=10.2, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=8))
    styles.add(ParagraphStyle("Equation", parent=styles["Code"], fontName="Courier", fontSize=8.8, leading=11, alignment=TA_CENTER, spaceBefore=5, spaceAfter=7))
    styles["Heading1"].fontName = "Times-Bold"
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 17
    styles["Heading1"].spaceBefore = 14
    styles["Heading1"].spaceAfter = 7
    styles["Heading2"].fontName = "Times-Bold"
    styles["Heading2"].fontSize = 11.5
    styles["Heading2"].leading = 14
    return styles


class PdfFigure(Flowable):
    def __init__(self, drawing: Drawing, width: float):
        super().__init__()
        self.drawing = drawing
        self.width = width
        self.scale = width / drawing.width
        self.height = drawing.height * self.scale

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        self.canv.saveState()
        self.canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def para(text: str, styles) -> Paragraph:
    return Paragraph(html.escape(text).replace("`", ""), styles["BodyJustify"])


def make_table(rows: list[list[str]], styles, font_size=7.2) -> Table:
    data = [[Paragraph(html.escape(str(cell)), ParagraphStyle("TableCell", fontName="Times-Roman", fontSize=font_size, leading=font_size + 2)) for cell in row] for row in rows]
    tbl = Table(data, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(A4[0] / 2, 0.55 * cm, str(doc.page))
    canvas.restoreState()


def build_pdf() -> None:
    styles = build_styles()
    doc = DocTemplate(str(PDF_OUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=0.9 * cm)
    story = []
    story.append(Paragraph("A Hybrid Regulation-Metabolism Digital Twin for Accelerating Simulated Yeast Bioprocess DBTL", styles["TitleMain"]))
    story.append(Paragraph("Development, Validation, Negative Results, and Deployment-Style Evaluation", styles["Subtitle"]))
    story.append(Paragraph("Abstract", styles["Heading1"]))
    story.append(Paragraph(html.escape(abstract), styles["Abstract"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName="Times-Roman", fontSize=9.5, name="TOCHeading1", leftIndent=0, firstLineIndent=0, spaceBefore=2, leading=11),
        ParagraphStyle(fontName="Times-Roman", fontSize=8.5, name="TOCHeading2", leftIndent=12, firstLineIndent=0, spaceBefore=1, leading=10),
    ]
    story.append(Paragraph("Contents", styles["Heading1"]))
    story.append(toc)
    story.append(PageBreak())
    fig_counter = 0
    table_counter = 0
    eq_counter = 0
    for section in sections:
        story.append(Paragraph(section.title, styles["Heading1"]))
        for block in section.body:
            if isinstance(block, str):
                story.append(para(block, styles))
            elif isinstance(block, list):
                story.append(ListFlowable([ListItem(para(x, styles)) for x in block], bulletType="bullet", leftIndent=14))
            elif isinstance(block, FigureRef):
                fig_counter += 1
                name, caption, drawing = figures[block.idx - 1]
                story.append(KeepTogether([
                    PdfFigure(drawing, doc.width * 0.85),
                    Paragraph(f"Figure {fig_counter}. {html.escape(caption)}", styles["Caption"]),
                ]))
            elif isinstance(block, TableRef):
                table_counter += 1
                table_flow = [
                    Paragraph(f"Table {table_counter}. {html.escape(block.title)}", styles["Caption"]),
                    make_table(block.rows, styles),
                    Spacer(1, 6),
                ]
                story.append(KeepTogether(table_flow))
            elif isinstance(block, Equation):
                eq_counter += 1
                story.append(Paragraph(f"{html.escape(block.text)}    ({eq_counter})", styles["Equation"]))
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)


def tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def table_tex(rows: list[list[str]]) -> str:
    n = len(rows[0])
    spec = "p{0.20\\linewidth}" + "".join(["p{0.12\\linewidth}" for _ in range(n - 1)])
    out = ["\\begin{tabularx}{\\linewidth}{" + ">{\\raggedright\\arraybackslash}" + spec + "}", "\\toprule"]
    out.append(" & ".join(tex_escape(c) for c in rows[0]) + r" \\")
    out.append("\\midrule")
    for row in rows[1:]:
        out.append(" & ".join(tex_escape(c) for c in row) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{tabularx}")
    return "\n".join(out)


def build_tex() -> None:
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{longtable}",
        r"\usepackage{amsmath}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}",
        r"\title{A Hybrid Regulation-Metabolism Digital Twin for Accelerating Simulated Yeast Bioprocess DBTL\\\large Development, Validation, Negative Results, and Deployment-Style Evaluation}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_escape(abstract),
        r"\end{abstract}",
        r"\tableofcontents",
        r"\newpage",
    ]
    fig_counter = table_counter = eq_counter = 0
    for section in sections:
        lines.append("\\section{" + tex_escape(section.title) + "}")
        for block in section.body:
            if isinstance(block, str):
                lines.append(tex_escape(block) + "\n")
            elif isinstance(block, list):
                lines.append(r"\begin{itemize}")
                for item in block:
                    lines.append(r"\item " + tex_escape(item))
                lines.append(r"\end{itemize}")
            elif isinstance(block, FigureRef):
                fig_counter += 1
                name, caption, _ = figures[block.idx - 1]
                lines.extend([
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.82\linewidth]{{figures/scientific_report/{name}.pdf}}",
                    r"\caption{" + tex_escape(caption) + "}",
                    r"\end{figure}",
                ])
            elif isinstance(block, TableRef):
                table_counter += 1
                lines.extend([
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\caption{" + tex_escape(block.title) + "}",
                    r"\scriptsize",
                    table_tex(block.rows),
                    r"\end{table}",
                ])
            elif isinstance(block, Equation):
                eq_counter += 1
                lines.append(r"\begin{equation}")
                lines.append(r"\text{" + tex_escape(block.text) + "}")
                lines.append(r"\end{equation}")
    lines.append(r"\end{document}")
    TEX_OUT.write_text("\n".join(lines) + "\n")


def build_md() -> None:
    lines = [
        "# A Hybrid Regulation-Metabolism Digital Twin for Accelerating Simulated Yeast Bioprocess DBTL",
        "",
        "*Development, Validation, Negative Results, and Deployment-Style Evaluation*",
        "",
        "## Abstract",
        "",
        abstract,
        "",
    ]
    fig_counter = table_counter = eq_counter = 0
    for section in sections:
        lines.extend(["", f"## {section.title}", ""])
        for block in section.body:
            if isinstance(block, str):
                lines.extend([block, ""])
            elif isinstance(block, list):
                lines.extend([f"- {item}" for item in block])
                lines.append("")
            elif isinstance(block, FigureRef):
                fig_counter += 1
                name, caption, _ = figures[block.idx - 1]
                lines.extend([f"![Figure {fig_counter}. {caption}](figures/scientific_report/{name}.pdf)", ""])
            elif isinstance(block, TableRef):
                table_counter += 1
                lines.append(f"**Table {table_counter}. {block.title}**")
                rows = block.rows
                lines.append("| " + " | ".join(rows[0]) + " |")
                lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                for row in rows[1:]:
                    lines.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
                lines.append("")
            elif isinstance(block, Equation):
                eq_counter += 1
                lines.extend([f"Equation ({eq_counter}): `{block.text}`", ""])
    MD_OUT.write_text("\n".join(lines).replace("\u2013", "-").replace("\u2014", "-") + "\n")


def main() -> None:
    save_figures()
    build_md()
    build_tex()
    build_pdf()


if __name__ == "__main__":
    main()
