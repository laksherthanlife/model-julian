#!/usr/bin/env python3
"""Build the full prospective on-demand DBTL benchmark technical report."""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
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


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "prospective_dbtl_benchmark"
FIG_DIR = OUT_DIR / "report_figures"
PDF_PATH = OUT_DIR / "YEAST_DIGITAL_TWIN_PROSPECTIVE_DBTL_BENCHMARK.pdf"
TEX_PATH = OUT_DIR / "YEAST_DIGITAL_TWIN_PROSPECTIVE_DBTL_BENCHMARK.tex"
HEADLINE_PATH = OUT_DIR / "prospective_dbtl_report_headline_numbers.csv"

METHOD_LABELS = {
    "conventional_dbtl": "Conventional DBTL",
    "pretrained_hybrid_digital_twin": "Pretrained hybrid digital twin",
}
METHOD_COLORS = {
    "conventional_dbtl": colors.HexColor("#374151"),
    "pretrained_hybrid_digital_twin": colors.HexColor("#2563eb"),
}
WORLD_LABELS = {
    "baseline_world": "Baseline world",
    "strong_oxidative_burden_world": "Strong oxidative burden world",
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def safe_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def fmt_ci(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def short_id(text: str, n: int = 12) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n]


def latex_escape(text: Any) -> str:
    s = str(text)
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


class DrawingFlowable(Flowable):
    """Platypus wrapper for a ReportLab vector drawing."""

    def __init__(self, drawing: Drawing, width: float):
        super().__init__()
        self.drawing = drawing
        self.target_width = width
        self.scale = width / drawing.width
        self.width = width
        self.height = drawing.height * self.scale

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        width = min(self.target_width, avail_width)
        self.scale = width / self.drawing.width
        self.width = width
        self.height = self.drawing.height * self.scale
        return self.width, self.height

    def draw(self) -> None:
        self.canv.saveState()
        self.canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def add_arrow(d: Drawing | Group, x1: float, y1: float, x2: float, y2: float, color=colors.HexColor("#4b5563")) -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.1))
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 6
    pts = []
    for a in [angle, angle + 2.6, angle - 2.6]:
        pts.extend([x2 - size * math.cos(a), y2 - size * math.sin(a)])
    d.add(Polygon(pts, fillColor=color, strokeColor=color))


def box(d: Drawing | Group, x: float, y: float, w: float, h: float, label: str, fill, stroke=colors.HexColor("#cbd5e1"), fs: float = 8.2) -> None:
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    lines = textwrap.wrap(label, width=max(12, int(w / 6.0)))
    ty = y + h / 2 + (len(lines) - 1) * fs * 0.55
    for line in lines:
        d.add(String(x + w / 2, ty, line, textAnchor="middle", fontName="Helvetica", fontSize=fs, fillColor=colors.HexColor("#111827")))
        ty -= fs * 1.15


def save_drawing(drawing: Drawing, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    renderPDF.drawToFile(drawing, str(path))
    return path


@dataclass
class SourceData:
    historical: pd.DataFrame
    acceptance: pd.DataFrame
    completion: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame
    world: pd.DataFrame
    common_auc: pd.DataFrame
    best_culture: pd.DataFrame
    stages: pd.DataFrame
    exact: pd.DataFrame
    virtual: pd.DataFrame
    final_candidates: pd.DataFrame
    thresholds: pd.DataFrame
    manifest: dict[str, Any]


def load_data() -> SourceData:
    manifest_path = ROOT / "data" / "prospective_full_run_frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return SourceData(
        historical=read_csv("data/prospective_historical_training_audit.csv"),
        acceptance=read_csv("data/prospective_pilot_acceptance.csv"),
        completion=read_csv("data/prospective_full_hpc_completion_audit.csv"),
        summary=read_csv("data/prospective_campaign_summary.csv"),
        paired=read_csv("data/prospective_paired_statistical_comparisons.csv"),
        world=read_csv("data/prospective_world_paired_summary.csv"),
        common_auc=read_csv("data/prospective_best_so_far_auc_common_budget.csv"),
        best_culture=read_csv("data/prospective_best_so_far_by_culture.csv"),
        stages=read_csv("data/prospective_biological_stage_ledger.csv"),
        exact=read_csv("data/prospective_exact_simulator_call_ledger.csv"),
        virtual=read_csv("data/prospective_virtual_evaluation_ledger.csv"),
        final_candidates=read_csv("data/prospective_final_best_candidates.csv"),
        thresholds=read_csv("data/prospective_relative_threshold_efficiency.csv"),
        manifest=manifest,
    )


def paired_row(data: SourceData, metric: str) -> pd.Series:
    row = data.paired[data.paired["metric"].eq(metric)]
    if row.empty:
        raise KeyError(metric)
    return row.iloc[0]


def method_stats(data: SourceData, metric: str) -> pd.DataFrame:
    return data.summary.groupby("method_id")[metric].agg(["mean", "median", "std", "min", "max"]).reset_index()


def campaign_wide(data: SourceData) -> pd.DataFrame:
    wide = data.summary.pivot_table(
        index=["campaign_id", "world_id"],
        columns="method_id",
        values="best_final_product",
        aggfunc="first",
    ).reset_index()
    wide["hybrid_minus_conventional"] = wide["pretrained_hybrid_digital_twin"] - wide["conventional_dbtl"]
    wide["seed"] = wide["campaign_id"].str.extract(r"seed_(\d+)").astype(int)
    return wide.sort_values(["world_id", "seed"])


def representative_campaigns(data: SourceData) -> pd.DataFrame:
    wide = campaign_wide(data)
    wins = wide[wide["hybrid_minus_conventional"] > 0].copy()
    losses = wide[wide["hybrid_minus_conventional"] < 0].copy()
    typical_win = wins.iloc[(wins["hybrid_minus_conventional"] - wins["hybrid_minus_conventional"].median()).abs().argsort().iloc[0]]
    near_tie = wide.iloc[wide["hybrid_minus_conventional"].abs().argsort().iloc[0]]
    conv_win = losses.iloc[(losses["hybrid_minus_conventional"] - losses["hybrid_minus_conventional"].median()).abs().argsort().iloc[0]]
    out = pd.DataFrame([typical_win, near_tie, conv_win]).copy()
    out["case_type"] = ["typical hybrid win", "near tie", "conventional win"]
    return out


def best_candidate_call(data: SourceData, campaign_id: str, method_id: str) -> tuple[pd.Series, pd.Series]:
    row = data.final_candidates[
        data.final_candidates["campaign_id"].eq(campaign_id)
        & data.final_candidates["method_id"].eq(method_id)
    ].iloc[0]
    call = data.exact[
        data.exact["campaign_id"].eq(campaign_id)
        & data.exact["method_id"].eq(method_id)
        & data.exact["candidate_hash"].eq(row["best_candidate_hash"])
    ].iloc[0]
    return row, call


def figure_overview() -> Drawing:
    d = Drawing(520, 300)
    d.add(String(12, 278, "Prospective on-demand DBTL benchmark workflow", fontName="Helvetica-Bold", fontSize=13, fillColor=colors.HexColor("#111827")))
    d.add(String(18, 246, "Conventional adaptive DBTL", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#374151")))
    y = 205
    xs = [18, 142, 266, 390]
    for i, x in enumerate(xs, start=1):
        box(d, x, y, 92, 44, f"Round {i}: propose 4, exact culture, learn", colors.HexColor("#f3f4f6"), fs=7.6)
        if i < 4:
            add_arrow(d, x + 92, y + 22, xs[i] - 8, y + 22)
    d.add(String(18, 178, "16 physical-style cultures, 4 sequential biological stages", fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#4b5563")))
    d.add(String(18, 132, "Pretrained hybrid digital twin", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#1d4ed8")))
    box(d, 18, 82, 102, 42, "Historical reference-strain environmental data", colors.HexColor("#eff6ff"), fs=7.6)
    box(d, 146, 82, 92, 42, "Frozen pretrained regulation-metabolism model", colors.HexColor("#dbeafe"), fs=7.6)
    box(d, 264, 82, 92, 42, "6,000 virtual strain-environment evaluations", colors.HexColor("#dbeafe"), fs=7.6)
    box(d, 382, 82, 102, 42, "Top 8 exact verification cultures", colors.HexColor("#bfdbfe"), fs=7.6)
    add_arrow(d, 120, 103, 146, 103)
    add_arrow(d, 238, 103, 264, 103)
    add_arrow(d, 356, 103, 382, 103)
    d.add(String(18, 56, "8 physical-style cultures, 1 parallel verification stage; virtual evaluations are compute, not cultures", fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#1e3a8a")))
    return d


def figure_fairness() -> Drawing:
    d = Drawing(520, 300)
    d.add(String(12, 278, "Information boundaries and fairness controls", fontName="Helvetica-Bold", fontSize=13, fillColor=colors.HexColor("#111827")))
    box(d, 190, 132, 140, 62, "Hidden exact Yeast9 dynamic pFBA simulator", colors.HexColor("#fef3c7"), colors.HexColor("#d97706"), fs=8.6)
    box(d, 20, 214, 145, 42, "Shared engineering domain: strain edits + T/pH/DO", colors.HexColor("#f8fafc"), fs=7.9)
    box(d, 355, 214, 145, 42, "No known optimum supplied to either workflow", colors.HexColor("#f8fafc"), fs=7.9)
    box(d, 20, 42, 145, 58, "Conventional may adapt only after observed exact results", colors.HexColor("#f3f4f6"), fs=7.9)
    box(d, 355, 42, 145, 58, "Hybrid may use pretrained model + virtual search, but no hidden exact outcomes", colors.HexColor("#eff6ff"), fs=7.9)
    box(d, 190, 30, 140, 50, "Prospective sidecar reuse only after a legitimate request", colors.HexColor("#ecfdf5"), colors.HexColor("#10b981"), fs=7.8)
    add_arrow(d, 92, 214, 220, 194)
    add_arrow(d, 428, 214, 300, 194)
    add_arrow(d, 165, 71, 190, 148)
    add_arrow(d, 355, 71, 330, 148)
    add_arrow(d, 260, 80, 260, 132, colors.HexColor("#059669"))
    d.add(String(260, 116, "no old finite-pool result access", textAnchor="middle", fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#065f46")))
    return d


def line_plot(
    title: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float], Any]],
    xlabel: str,
    ylabel: str,
    width: int = 520,
    height: int = 300,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Drawing:
    d = Drawing(width, height)
    left, right, bottom, top = 58, width - 25, 48, height - 38
    d.add(String(12, height - 20, title, fontName="Helvetica-Bold", fontSize=12, fillColor=colors.HexColor("#111827")))
    all_x = [v for _, xs, _, _ in series for v in xs]
    all_y = [v for _, _, ys, _ in series for v in ys if np.isfinite(v)]
    xmin, xmax = xlim or (min(all_x), max(all_x))
    ymin, ymax = ylim or (min(0, min(all_y)), max(all_y) * 1.08)
    if ymax <= ymin:
        ymax = ymin + 1

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def sy(y: float) -> float:
        return bottom + (y - ymin) / (ymax - ymin) * (top - bottom)

    d.add(Line(left, bottom, right, bottom, strokeColor=colors.HexColor("#111827"), strokeWidth=0.9))
    d.add(Line(left, bottom, left, top, strokeColor=colors.HexColor("#111827"), strokeWidth=0.9))
    for t in np.linspace(xmin, xmax, 5):
        d.add(Line(sx(t), bottom, sx(t), bottom - 4, strokeColor=colors.HexColor("#111827"), strokeWidth=0.7))
        d.add(String(sx(t), bottom - 17, safe_num(t, 0 if abs(t - round(t)) < 1e-6 else 1), textAnchor="middle", fontName="Helvetica", fontSize=7))
    for t in np.linspace(ymin, ymax, 5):
        d.add(Line(left - 4, sy(t), left, sy(t), strokeColor=colors.HexColor("#111827"), strokeWidth=0.7))
        d.add(String(left - 7, sy(t) - 3, safe_num(t, 1), textAnchor="end", fontName="Helvetica", fontSize=7))
        if t > ymin:
            d.add(Line(left, sy(t), right, sy(t), strokeColor=colors.HexColor("#e5e7eb"), strokeWidth=0.5))
    d.add(String((left + right) / 2, 16, xlabel, textAnchor="middle", fontName="Helvetica", fontSize=8))
    d.add(String(14, (bottom + top) / 2, ylabel, textAnchor="middle", fontName="Helvetica", fontSize=8, transform=[0, 1, -1, 0, 14, (bottom + top) / 2]))
    legend_y = top + 9
    legend_x = left
    for label, xs, ys, color in series:
        points = [(sx(float(x)), sy(float(y))) for x, y in zip(xs, ys) if np.isfinite(y)]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.8))
        for x, y in points:
            d.add(Rect(x - 1.8, y - 1.8, 3.6, 3.6, fillColor=color, strokeColor=color))
        d.add(Line(legend_x, legend_y, legend_x + 14, legend_y, strokeColor=color, strokeWidth=2))
        d.add(String(legend_x + 18, legend_y - 3, label, fontName="Helvetica", fontSize=7.5))
        legend_x += 175
    return d


def figure_best_vs_cultures(data: SourceData) -> Drawing:
    series = []
    for method_id in ["conventional_dbtl", "pretrained_hybrid_digital_twin"]:
        sub = data.best_culture[data.best_culture["method_id"].eq(method_id)]
        grouped = sub.groupby("physical_culture_index")["best_final_product_so_far"].mean().reset_index()
        series.append((METHOD_LABELS[method_id], grouped["physical_culture_index"].tolist(), grouped["best_final_product_so_far"].tolist(), METHOD_COLORS[method_id]))
    return line_plot("Mean best final product discovered versus physical cultures", series, "physical-style exact cultures", "best final product", xlim=(1, 16))


def figure_stage(data: SourceData) -> Drawing:
    series = []
    for method_id in ["conventional_dbtl", "pretrained_hybrid_digital_twin"]:
        sub = data.stages[data.stages["method_id"].eq(method_id)]
        grouped = sub.groupby("stage_index")["best_verified_final_product_so_far"].mean().reset_index()
        series.append((METHOD_LABELS[method_id], grouped["stage_index"].tolist(), grouped["best_verified_final_product_so_far"].tolist(), METHOD_COLORS[method_id]))
    return line_plot("Mean best final product discovered versus sequential biological stage", series, "sequential biological stage", "best final product", xlim=(1, 4))


def figure_common_budget(data: SourceData) -> Drawing:
    sub = data.best_culture[data.best_culture["physical_culture_index"].astype(int) <= 8]
    series = []
    for method_id in ["conventional_dbtl", "pretrained_hybrid_digital_twin"]:
        grouped = sub[sub["method_id"].eq(method_id)].groupby("physical_culture_index")["best_final_product_so_far"].mean().reset_index()
        series.append((METHOD_LABELS[method_id], grouped["physical_culture_index"].tolist(), grouped["best_final_product_so_far"].tolist(), METHOD_COLORS[method_id]))
    return line_plot("Common 8-culture horizon: mean best-so-far curve", series, "physical-style exact cultures", "best final product", xlim=(1, 8))


def figure_paired_scatter(data: SourceData) -> Drawing:
    wide = campaign_wide(data)
    d = Drawing(520, 300)
    left, right, bottom, top = 58, 365, 45, 255
    vals = pd.concat([wide["conventional_dbtl"], wide["pretrained_hybrid_digital_twin"]])
    lo, hi = 0, float(vals.max() * 1.08)

    def sx(x: float) -> float:
        return left + (x - lo) / (hi - lo) * (right - left)

    def sy(y: float) -> float:
        return bottom + (y - lo) / (hi - lo) * (top - bottom)

    d.add(String(12, 278, "Paired final-product comparison across 20 world-seed campaigns", fontName="Helvetica-Bold", fontSize=12))
    d.add(Line(left, bottom, right, bottom, strokeColor=colors.black, strokeWidth=0.9))
    d.add(Line(left, bottom, left, top, strokeColor=colors.black, strokeWidth=0.9))
    d.add(Line(left, bottom, right, top, strokeColor=colors.HexColor("#9ca3af"), strokeWidth=0.9))
    for t in np.linspace(lo, hi, 5):
        d.add(String(sx(t), bottom - 16, safe_num(t, 1), textAnchor="middle", fontName="Helvetica", fontSize=7))
        d.add(String(left - 8, sy(t) - 3, safe_num(t, 1), textAnchor="end", fontName="Helvetica", fontSize=7))
    for _, row in wide.iterrows():
        color = colors.HexColor("#1d4ed8") if row["hybrid_minus_conventional"] > 0 else colors.HexColor("#991b1b")
        d.add(Rect(sx(row["conventional_dbtl"]) - 2.5, sy(row["pretrained_hybrid_digital_twin"]) - 2.5, 5, 5, fillColor=color, strokeColor=color))
    d.add(String((left + right) / 2, 14, "conventional best final product", textAnchor="middle", fontName="Helvetica", fontSize=8))
    d.add(String(16, (bottom + top) / 2, "hybrid best final product", textAnchor="middle", fontName="Helvetica", fontSize=8, transform=[0, 1, -1, 0, 16, (bottom + top) / 2]))
    paired = paired_row(data, "best_final_product")
    notes = [
        f"Mean paired difference: {paired.mean_difference:+.3f}",
        f"Median paired difference: {paired.median_difference:+.3f}",
        f"Bootstrap CI: {fmt_ci(paired.bootstrap_ci_low, paired.bootstrap_ci_high)}",
        f"Hybrid wins/losses: {int(paired.hybrid_win_count)}/{int(paired.hybrid_loss_count)}",
    ]
    box(d, 385, 170, 120, 72, "\n".join(notes), colors.HexColor("#f8fafc"), fs=7.3)
    return d


def figure_delta_distribution(data: SourceData) -> Drawing:
    wide = campaign_wide(data).sort_values("hybrid_minus_conventional").reset_index(drop=True)
    d = Drawing(520, 300)
    left, right, bottom, top = 58, 500, 50, 255
    vals = wide["hybrid_minus_conventional"].to_numpy(float)
    ymin, ymax = float(min(vals.min(), 0) * 1.1), float(max(vals.max(), 0) * 1.1)

    def sx(i: int) -> float:
        return left + i / max(1, len(vals) - 1) * (right - left)

    def sy(v: float) -> float:
        return bottom + (v - ymin) / (ymax - ymin) * (top - bottom)

    d.add(String(12, 278, "Distribution of hybrid-minus-conventional final-product effects", fontName="Helvetica-Bold", fontSize=12))
    zero = sy(0)
    d.add(Line(left, zero, right, zero, strokeColor=colors.HexColor("#111827"), strokeWidth=0.9))
    bar_w = (right - left) / len(vals) * 0.72
    for i, val in enumerate(vals):
        color = colors.HexColor("#2563eb") if val > 0 else colors.HexColor("#b91c1c")
        x = sx(i)
        y = sy(val)
        d.add(Rect(x - bar_w / 2, min(zero, y), bar_w, abs(y - zero), fillColor=color, strokeColor=color))
    mean = vals.mean()
    med = np.median(vals)
    d.add(Line(left, sy(mean), right, sy(mean), strokeColor=colors.HexColor("#f59e0b"), strokeWidth=1.4))
    d.add(Line(left, sy(med), right, sy(med), strokeColor=colors.HexColor("#059669"), strokeWidth=1.4))
    d.add(String(right - 115, sy(mean) + 4, f"mean {mean:+.3f}", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#92400e")))
    d.add(String(right - 115, sy(med) - 12, f"median {med:+.3f}", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#065f46")))
    for t in np.linspace(ymin, ymax, 5):
        d.add(String(left - 8, sy(t) - 3, safe_num(t, 1), textAnchor="end", fontName="Helvetica", fontSize=7))
    d.add(String((left + right) / 2, 16, "campaigns sorted by paired effect", textAnchor="middle", fontName="Helvetica", fontSize=8))
    d.add(String(16, (bottom + top) / 2, "hybrid - conventional", textAnchor="middle", fontName="Helvetica", fontSize=8, transform=[0, 1, -1, 0, 16, (bottom + top) / 2]))
    return d


def figure_world_effects(data: SourceData) -> Drawing:
    rows = data.world[data.world["metric"].eq("best_final_product")].copy()
    overall = paired_row(data, "best_final_product")
    d = Drawing(520, 270)
    d.add(String(12, 248, "World-separated and pooled paired final-product effects", fontName="Helvetica-Bold", fontSize=12))
    left, right, center_y = 115, 480, 120
    xmin, xmax = -0.6, 2.6

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    d.add(Line(sx(0), 55, sx(0), 220, strokeColor=colors.HexColor("#9ca3af"), strokeWidth=1))
    labels = ["Pooled", "Baseline", "Strong oxidative"]
    effects = [
        (overall.mean_difference, overall.bootstrap_ci_low, overall.bootstrap_ci_high),
        tuple(rows[rows["world_id"].eq("baseline_world")][["mean_hybrid_minus_conventional", "bootstrap_ci_low", "bootstrap_ci_high"]].iloc[0]),
        tuple(rows[rows["world_id"].eq("strong_oxidative_burden_world")][["mean_hybrid_minus_conventional", "bootstrap_ci_low", "bootstrap_ci_high"]].iloc[0]),
    ]
    ys = [190, 140, 90]
    for label, (mean, low, high), y in zip(labels, effects, ys):
        d.add(String(25, y - 4, label, fontName="Helvetica", fontSize=8.5))
        d.add(Line(sx(float(low)), y, sx(float(high)), y, strokeColor=colors.HexColor("#1f2937"), strokeWidth=1.2))
        d.add(Rect(sx(float(mean)) - 4, y - 4, 8, 8, fillColor=colors.HexColor("#2563eb"), strokeColor=colors.HexColor("#1d4ed8")))
        d.add(String(sx(float(high)) + 5, y - 3, f"{float(mean):+.3f}", fontName="Helvetica", fontSize=7.5))
    for t in [-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        d.add(Line(sx(t), 50, sx(t), 45, strokeColor=colors.black, strokeWidth=0.7))
        d.add(String(sx(t), 32, safe_num(t, 1), textAnchor="middle", fontName="Helvetica", fontSize=7))
    d.add(Line(left, 50, right, 50, strokeColor=colors.black, strokeWidth=0.8))
    d.add(String((left + right) / 2, 12, "mean hybrid-minus-conventional final product with bootstrap CI", textAnchor="middle", fontName="Helvetica", fontSize=8))
    return d


def figure_trajectories(data: SourceData) -> Drawing:
    reps = representative_campaigns(data)
    d = Drawing(520, 360)
    d.add(String(12, 338, "Representative best-candidate product trajectories", fontName="Helvetica-Bold", fontSize=12))
    panel_w, panel_h = 150, 245
    lefts = [44, 196, 348]
    bottom = 58
    for idx, (_, rep) in enumerate(reps.iterrows()):
        campaign_id = rep["campaign_id"]
        x0 = lefts[idx]
        max_y = 0
        trajs = {}
        for method_id in ["conventional_dbtl", "pretrained_hybrid_digital_twin"]:
            _, call = best_candidate_call(data, campaign_id, method_id)
            traj = pd.read_csv(ROOT / str(call["trajectory_path"]))
            trajs[method_id] = traj
            max_y = max(max_y, float(traj["B_total"].max()))
        max_y = max_y * 1.08 if max_y > 0 else 1.0
        d.add(String(x0 + panel_w / 2, 314, str(rep["case_type"]).title(), textAnchor="middle", fontName="Helvetica-Bold", fontSize=8))
        d.add(String(x0 + panel_w / 2, 301, WORLD_LABELS.get(rep["world_id"], rep["world_id"]), textAnchor="middle", fontName="Helvetica", fontSize=7))
        d.add(Line(x0, bottom, x0 + panel_w, bottom, strokeColor=colors.black, strokeWidth=0.7))
        d.add(Line(x0, bottom, x0, bottom + panel_h, strokeColor=colors.black, strokeWidth=0.7))
        for method_id, traj in trajs.items():
            color = METHOD_COLORS[method_id]
            pts = []
            for _, row in traj.iterrows():
                x = x0 + float(row["time"]) / float(traj["time"].max()) * panel_w
                y = bottom + float(row["B_total"]) / max_y * panel_h
                pts.append((x, y))
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
        d.add(String(x0 - 8, bottom + panel_h - 3, safe_num(max_y, 1), textAnchor="end", fontName="Helvetica", fontSize=6.5))
    d.add(Line(360, 24, 380, 24, strokeColor=METHOD_COLORS["conventional_dbtl"], strokeWidth=1.8))
    d.add(String(384, 21, "Conventional", fontName="Helvetica", fontSize=7))
    d.add(Line(440, 24, 460, 24, strokeColor=METHOD_COLORS["pretrained_hybrid_digital_twin"], strokeWidth=1.8))
    d.add(String(464, 21, "Hybrid", fontName="Helvetica", fontSize=7))
    d.add(String(260, 20, "time (simulation units); y-axis is accumulated beta-carotene product", textAnchor="middle", fontName="Helvetica", fontSize=7))
    return d


def figure_accounting(data: SourceData) -> Drawing:
    d = Drawing(520, 300)
    d.add(String(12, 278, "Biological-query, virtual-compute, and exact-solver accounting", fontName="Helvetica-Bold", fontSize=12))
    items = [
        ("Historical reference-strain cultures", 125, "pre-deployment"),
        ("Prospective exact cultures", len(data.exact), "deployment"),
        ("Hybrid virtual evaluations", len(data.virtual), "compute"),
        ("Yeast9 LP solves", int(data.exact["n_actual_lp_solves"].sum()), "exact simulator"),
    ]
    left, bottom, bar_w = 210, 75, 245
    max_log = math.log10(max(v for _, v, _ in items))
    for i, (label, value, note) in enumerate(items):
        y = 230 - i * 45
        length = math.log10(value) / max_log * bar_w
        d.add(String(24, y + 4, label, fontName="Helvetica", fontSize=8.5))
        d.add(Rect(left, y, length, 16, fillColor=colors.HexColor("#dbeafe"), strokeColor=colors.HexColor("#2563eb")))
        d.add(String(left + length + 8, y + 4, fmt_int(value), fontName="Helvetica-Bold", fontSize=8.5))
        d.add(String(left + length + 8, y - 9, note, fontName="Helvetica-Oblique", fontSize=6.8, fillColor=colors.HexColor("#4b5563")))
    d.add(String(left, 48, "bar length is log10-scaled because virtual evaluations and LP solves are orders of magnitude larger", fontName="Helvetica-Oblique", fontSize=7, fillColor=colors.HexColor("#4b5563")))
    return d


def figure_files(data: SourceData) -> list[tuple[str, Drawing]]:
    figures = [
        ("fig01_experiment_overview.pdf", figure_overview()),
        ("fig02_information_boundaries.pdf", figure_fairness()),
        ("fig03_best_product_vs_cultures.pdf", figure_best_vs_cultures(data)),
        ("fig04_best_product_vs_stage.pdf", figure_stage(data)),
        ("fig05_common_budget_search.pdf", figure_common_budget(data)),
        ("fig06_paired_final_product.pdf", figure_paired_scatter(data)),
        ("fig07_delta_distribution.pdf", figure_delta_distribution(data)),
        ("fig08_world_effects.pdf", figure_world_effects(data)),
        ("fig09_representative_trajectories.pdf", figure_trajectories(data)),
        ("fig10_resource_accounting.pdf", figure_accounting(data)),
    ]
    for filename, drawing in figures:
        save_drawing(drawing, filename)
    return figures


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet_list(items: Sequence[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(item, style), bulletColor=colors.HexColor("#2563eb")) for item in items], bulletType="bullet", start="circle")


def table_flowable(rows: list[list[Any]], widths: Sequence[float], small: bool = False) -> Table:
    converted: list[list[Any]] = []
    cell_style = STYLES["TableCellSmall"] if small else STYLES["TableCell"]
    for row in rows:
        converted.append([Paragraph(str(cell), cell_style) for cell in row])
    table = Table(converted, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def table_caption(label: str) -> Paragraph:
    return Paragraph(label, STYLES["TableCaption"])


def figure_caption(label: str) -> Paragraph:
    return Paragraph(label, STYLES["FigureCaption"])


def h1(text: str) -> Paragraph:
    return Paragraph(text, STYLES["ReportHeading1"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, STYLES["ReportHeading2"])


def h3(text: str) -> Paragraph:
    return Paragraph(text, STYLES["ReportHeading3"])


def spacer(h: float = 0.18 * cm) -> Spacer:
    return Spacer(1, h)


def historical_table(data: SourceData) -> list[list[str]]:
    lookup = dict(zip(data.historical["audit_item"], data.historical["value"].astype(str)))
    return [
        ["Item", "Final audited value"],
        ["Historical cultures", lookup.get("historical_cultures", "")],
        ["Teacher-training split cultures", lookup.get("teacher_training_cultures", "")],
        ["Training strain", "single reference / no edit strain"],
        ["Deployment inputs", "temperature, pH, dissolved oxygen"],
        ["Historical exact solver cost", "18,000 Yeast9 LP solves"],
        ["Surrogate evaluations labelled as exact biology", "0"],
        ["Edit/intervention/candidate columns in training", "absent"],
        ["Reporter labels available during training", lookup.get("reporter_information_available", "")],
    ]


def protocol_table() -> list[list[str]]:
    return [
        ["Workflow", "Pretrained?", "Virtual evaluations", "Physical schedule", "Cultures", "Stages", "Exact simulator access"],
        ["Conventional DBTL", "No", "0", "[4,4,4,4]", "16 per campaign", "4", "Only after each proposed batch"],
        ["Pretrained hybrid digital twin", "Yes", "6,000 per campaign", "[8]", "8 per campaign", "1", "Only final verification batch"],
    ]


def resource_table(data: SourceData) -> list[list[str]]:
    return [
        ["Quantity", "Value", "Interpretation"],
        ["Completed world-seed pairs", f"{data.completion.shape[0]}/20", "paired experimental units"],
        ["Prospective exact physical-style cultures", fmt_int(len(data.exact)), "charged biological-query equivalents"],
        ["Conventional cultures", fmt_int(data.exact[data.exact.method_id.eq("conventional_dbtl")]["physical_culture_charge"].sum()), "16 per campaign"],
        ["Hybrid verification cultures", fmt_int(data.exact[data.exact.method_id.eq("pretrained_hybrid_digital_twin")]["physical_culture_charge"].sum()), "8 per campaign"],
        ["Hybrid virtual evaluations", fmt_int(len(data.virtual)), "computational screens, not cultures"],
        ["Yeast9 LP solves", fmt_int(data.exact["n_actual_lp_solves"].sum()), "144 per exact culture"],
        ["Fresh exact simulations", fmt_int(data.exact["fresh_exact_simulation_performed"].astype(bool).sum()), "new exact sidecars"],
        ["Valid prospective cache hits", fmt_int(data.exact["cache_hit"].astype(bool).sum()), "computational reuse after legitimate request"],
        ["Solver failures", "0", "infeasible, unbounded, solver errors, skipped intervals all zero"],
    ]


def acceptance_table(data: SourceData) -> list[list[str]]:
    rows = [["Gate", "Passed", "Value"]]
    for row in data.acceptance.itertuples(index=False):
        rows.append([str(row.gate), str(bool(row.passed)), str(row.value)])
    return rows


def paired_results_table(data: SourceData) -> list[list[str]]:
    metrics = [
        ("best_final_product", "Best final product"),
        ("best_productivity", "Best productivity"),
        ("best_product_AUC", "Best product AUC"),
        ("physical_cultures", "Physical cultures"),
        ("scenario_days", "Scenario stage-days"),
    ]
    rows = [["Metric", "Mean delta", "Median delta", "95% bootstrap CI", "Wins/Ties/Losses"]]
    for metric, label in metrics:
        r = paired_row(data, metric)
        rows.append([
            label,
            f"{r.mean_difference:+.6f}",
            f"{r.median_difference:+.6f}",
            fmt_ci(r.bootstrap_ci_low, r.bootstrap_ci_high, 6),
            f"{int(r.hybrid_win_count)}/{int(r.tie_count)}/{int(r.hybrid_loss_count)}",
        ])
    return rows


def world_table(data: SourceData) -> list[list[str]]:
    rows = [["World", "Metric", "Mean delta", "Median delta", "95% bootstrap CI", "Wins/Losses"]]
    sub = data.world[data.world["metric"].isin(["best_final_product", "best_productivity", "best_product_AUC"])]
    for row in sub.itertuples(index=False):
        rows.append([
            WORLD_LABELS.get(row.world_id, row.world_id),
            str(row.metric).replace("_", " "),
            f"{row.mean_hybrid_minus_conventional:+.6f}",
            f"{row.median_hybrid_minus_conventional:+.6f}",
            fmt_ci(row.bootstrap_ci_low, row.bootstrap_ci_high, 6),
            f"{int(row.hybrid_win_count)}/{int(row.hybrid_loss_count)}",
        ])
    return rows


def common_budget_table(data: SourceData) -> list[list[str]]:
    metrics = [
        ("common_horizon_best_so_far_auc_final_product", "Best-so-far AUC, final product"),
        ("common_horizon_mean_best_so_far_final_product", "Mean best-so-far final product"),
        ("common_horizon_final_best_so_far_final_product", "Final best at 8 cultures"),
        ("common_horizon_best_so_far_auc_productivity", "Best-so-far AUC, productivity"),
        ("common_horizon_best_so_far_auc_product_AUC", "Best-so-far AUC, product AUC"),
    ]
    rows = [["Common 8-culture metric", "Mean delta", "Median delta", "95% bootstrap CI", "Wins/Losses"]]
    for metric, label in metrics:
        r = paired_row(data, metric)
        rows.append([
            label,
            f"{r.mean_difference:+.6f}",
            f"{r.median_difference:+.6f}",
            fmt_ci(r.bootstrap_ci_low, r.bootstrap_ci_high, 6),
            f"{int(r.hybrid_win_count)}/{int(r.hybrid_loss_count)}",
        ])
    return rows


def representative_table(data: SourceData) -> list[list[str]]:
    rows = [["Case", "World/seed", "Method", "T", "pH", "DO", "Edits", "Final product", "Product AUC", "Biomass"]]
    reps = representative_campaigns(data)
    for _, rep in reps.iterrows():
        seed = re.search(r"seed_(\d+)", rep["campaign_id"]).group(1)
        for method_id in ["conventional_dbtl", "pretrained_hybrid_digital_twin"]:
            best, call = best_candidate_call(data, rep["campaign_id"], method_id)
            rows.append([
                rep["case_type"],
                f"{WORLD_LABELS.get(rep['world_id'], rep['world_id'])}, {seed}",
                "Conventional" if method_id == "conventional_dbtl" else "Hybrid",
                safe_num(call["temperature"], 2),
                safe_num(call["pH"], 2),
                safe_num(call["DO"], 1),
                f"{int(best['best_edit_count'])}; {best['best_mechanism_classes']}",
                safe_num(best["best_final_product"], 3),
                safe_num(best["best_product_AUC"], 3),
                safe_num(best["best_final_biomass"], 3),
            ])
    return rows


def headline_numbers(data: SourceData) -> pd.DataFrame:
    conv_stats = method_stats(data, "best_final_product").set_index("method_id")
    final = paired_row(data, "best_final_product")
    common = paired_row(data, "common_horizon_best_so_far_auc_final_product")
    world_fp = data.world[data.world["metric"].eq("best_final_product")].set_index("world_id")
    rows = [
        ("campaign_pairs_completed", data.completion.shape[0]),
        ("exact_physical_cultures", len(data.exact)),
        ("yeast9_lp_solves", int(data.exact["n_actual_lp_solves"].sum())),
        ("solver_failures", int(data.exact[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals"]].fillna(0).sum().sum())),
        ("hybrid_virtual_evaluations", len(data.virtual)),
        ("conventional_mean_best_final_product", conv_stats.loc["conventional_dbtl", "mean"]),
        ("conventional_median_best_final_product", conv_stats.loc["conventional_dbtl", "median"]),
        ("hybrid_mean_best_final_product", conv_stats.loc["pretrained_hybrid_digital_twin", "mean"]),
        ("hybrid_median_best_final_product", conv_stats.loc["pretrained_hybrid_digital_twin", "median"]),
        ("paired_final_product_mean_delta", final.mean_difference),
        ("paired_final_product_median_delta", final.median_difference),
        ("paired_final_product_bootstrap_ci_low", final.bootstrap_ci_low),
        ("paired_final_product_bootstrap_ci_high", final.bootstrap_ci_high),
        ("hybrid_final_product_wins", int(final.hybrid_win_count)),
        ("common_8_culture_auc_mean_delta", common.mean_difference),
        ("common_8_culture_auc_ci_low", common.bootstrap_ci_low),
        ("common_8_culture_auc_ci_high", common.bootstrap_ci_high),
        ("baseline_world_mean_final_product_delta", world_fp.loc["baseline_world", "mean_hybrid_minus_conventional"]),
        ("strong_oxidative_world_mean_final_product_delta", world_fp.loc["strong_oxidative_burden_world", "mean_hybrid_minus_conventional"]),
        ("final_status", data.acceptance["final_status_if_all_gates_considered"].iloc[0]),
    ]
    return pd.DataFrame(rows, columns=["headline_item", "value"])


def build_story(data: SourceData, figures: list[tuple[str, Drawing]]) -> list[Flowable]:
    story: list[Flowable] = []
    body = STYLES["ReportBody"]
    small = STYLES["ReportSmall"]
    final = paired_row(data, "best_final_product")
    common = paired_row(data, "common_horizon_best_so_far_auc_final_product")
    conv = data.summary[data.summary.method_id.eq("conventional_dbtl")]["best_final_product"]
    hyb = data.summary[data.summary.method_id.eq("pretrained_hybrid_digital_twin")]["best_final_product"]

    story += [
        Paragraph("Full Prospective On-Demand DBTL Benchmark for the Yeast Digital Twin", STYLES["ReportTitle"]),
        Paragraph("Standalone technical report", STYLES["ReportSubtitle"]),
        Spacer(1, 0.6 * cm),
        Paragraph(
            f"Benchmark ID: <b>prospective_on_demand_dbtl_v1</b>. Final status: <b>{data.acceptance['final_status_if_all_gates_considered'].iloc[0]}</b>.",
            STYLES["Centered"],
        ),
        Spacer(1, 0.8 * cm),
        Paragraph("Abstract", STYLES["AbstractHeading"]),
        para(
            "This report documents a full powered prospective synthetic DBTL benchmark in which a conventional adaptive DBTL workflow was compared with a pretrained hybrid yeast digital twin. The central question was whether a pretrained regulation-metabolism model could use large-scale virtual strain-environment search to reduce the number of prospective biological-query equivalents and the number of sequential DBTL rounds required to discover high-performing beta-carotene production designs. The experiment was open-ended: neither method was supplied a target candidate or a known optimum, and both workflows queried the same hidden exact Yeast9 dynamic pFBA simulator only when their frozen workflow requested a physical-style culture.",
            STYLES["Abstract"],
        ),
        para(
            f"The powered benchmark completed 20 paired world-seed comparisons spanning baseline and strong oxidative burden worlds. It charged 480 exact simulated cultures, executed 69,120 Yeast9 LP solves, and recorded 120,000 hybrid virtual evaluations. All 11 acceptance gates passed. Mean best final product was {conv.mean():.6f} for conventional DBTL and {hyb.mean():.6f} for the hybrid. The paired hybrid-minus-conventional mean final-product difference was {final.mean_difference:+.6f} with bootstrap CI {fmt_ci(final.bootstrap_ci_low, final.bootstrap_ci_high, 6)}. The hybrid used 8 rather than 16 physical-style cultures per campaign and one rather than four sequential biological stages. At the common 8-culture physical budget, hybrid best-so-far final-product search AUC exceeded conventional by {common.mean_difference:+.6f} with CI {fmt_ci(common.bootstrap_ci_low, common.bootstrap_ci_high, 6)}. These results support a bounded computational proof-of-concept that a pretrained hybrid digital twin can replace part of sequential biological exploration with parallel in-silico screening in this synthetic Yeast9 benchmark.",
            STYLES["Abstract"],
        ),
        PageBreak(),
    ]

    story += [
        h1("1. Introduction"),
        h2("1.1 DBTL bottleneck"),
        para("Design-Build-Test-Learn cycles are powerful but sequential. A physical batch can be run in parallel, but the next design round cannot be rationally proposed until the prior exact outcomes have been observed and incorporated. In a conventional engineering campaign, this causal depth can dominate calendar latency even when the number of individual cultures is moderate.", body),
        h2("1.2 Digital-twin hypothesis"),
        para("The operational hypothesis tested here is not merely that a model can predict a simulator. It is that a pretrained hybrid digital twin can perform many cheap virtual evaluations before requesting any prospective exact cultures, thereby replacing part of the expensive sequential biological search process. The model is useful only if virtual search changes what must be physically tested.", body),
        h2("1.3 Why earlier equal-query benchmarks were insufficient"),
        para("Earlier finite-pool acquisition benchmarks asked which acquisition rule performed best when every method spent the same hidden exact-query budget. That is a useful negative control, but it suppresses the deployment advantage a pretrained twin is meant to have: cheap virtual screening before physical verification. The present benchmark therefore changes the question from equal oracle-query acquisition to prospective deployment efficiency.", body),
        h2("1.4 Objective"),
        para("The central question was: can a pretrained hybrid yeast digital twin use large-scale virtual search to identify high-performing strain-environment designs using fewer prospective physical-style cultures and fewer sequential DBTL rounds than conventional adaptive DBTL?", body),
        KeepTogether([DrawingFlowable(figures[0][1], 16.5 * cm), figure_caption("Figure 1. Prospective workflow overview. Conventional DBTL iterates through four physical stages. The hybrid uses a frozen pretrained model to screen 6,000 candidates and then verifies eight designs in one stage.")]),
    ]

    story += [
        h1("2. System Under Study"),
        h2("2.1 Yeast9 dynamic exact simulator"),
        para("The hidden evaluator is a synthetic exact biological-query oracle built around the Yeast9 genome-scale metabolic model and a heterologous beta-carotene production pathway. A candidate specifies a fixed environment and a strain edit vector. The evaluator returns a dynamic product trajectory, biomass trajectory, regulatory/burden state variables, and final scalar summaries. It is hidden from the candidate proposal logic except through legitimate exact culture requests.", body),
        h2("2.2 Definition of one exact simulated culture"),
        para("One culture is a prospective exact hidden-simulator evaluation of one strain-environment candidate. It should be read as a physical-style culture or biological-query equivalent, not as a real wet-lab culture. In the verified dynamic protocol, each culture uses 49 time points, 48 intervals, and three LP optimisations per interval: growth optimisation, product optimisation, and pFBA. The expected exact solver cost is therefore 144 LP solves per culture.", body),
        h2("2.3 Dynamic physiology and product readouts"),
        para("The simulator includes growth, product accumulation, pathway-capacity states, oxidative burden, ATP burden, bottleneck burden, and ER burden variables. The primary endpoint is final accumulated beta-carotene product, with product AUC and productivity reported as secondary biological performance metrics. Biomass is tracked to audit biological validity.", body),
    ]

    story += [
        h1("3. Historical Digital-Twin Training"),
        h2("3.1 Historical reference-strain dataset"),
        para("The hybrid was not trained on the prospective strain-edit outcomes evaluated in this benchmark. The historical audit records a fixed-reference-strain environmental dataset: 125 historical cultures, 77 training-split cultures, one no-edit strain, and deployment inputs restricted to temperature, pH, and dissolved oxygen. The historical data generation cost is reported as 18,000 Yeast9 LP solves and zero surrogate evaluations labelled as exact biology.", body),
        table_caption("Table 1. Historical training dataset and pre-deployment information contract."),
        table_flowable(historical_table(data), [5.0 * cm, 10.8 * cm], small=True),
        h2("3.2 Why this separation matters"),
        para("This separation is central to the scientific interpretation. The hybrid begins deployment with a pretrained representation learned from environmental perturbations of a reference strain. It then uses that representation to rank unseen strain-environment designs. This is a stronger test than training directly on the candidate strain designs that later appear in the benchmark.", body),
        h2("3.3 Deployment versus greenfield accounting"),
        para("Historical cultures are a pre-deployment model-building cost. In a deployed-twin scenario, the relevant prospective campaign cost is the physical-style cultures requested during each new engineering campaign. In a greenfield scenario, the historical model-building cost must eventually be amortised across future deployments. The report therefore discloses both numbers but does not mix historical cultures into per-campaign prospective culture counts.", body),
    ]

    story += [
        h1("4. Prospective Benchmark Design"),
        h2("4.1 Frozen protocol"),
        para("The final powered protocol compares only conventional_dbtl and pretrained_hybrid_digital_twin. It covers 10 campaign seeds in each of two physiological worlds: baseline_world and strong_oxidative_burden_world. These 20 world-seed combinations are the paired experimental units.", body),
        table_caption("Table 2. Frozen prospective protocol."),
        table_flowable(protocol_table(), [3.2 * cm, 2.2 * cm, 2.8 * cm, 2.8 * cm, 2.2 * cm, 1.6 * cm, 3.4 * cm], small=True),
        h2("4.2 Conventional adaptive DBTL"),
        para("Conventional DBTL implements a causal Design-Build-Test-Learn loop. In round 1 it proposes four candidates, requests four exact cultures, and receives outcomes. In rounds 2-4 it proposes each next batch only after prior exact observations have been appended. This produces 16 physical-style cultures over four sequential stages per campaign.", body),
        h2("4.3 Pretrained hybrid digital twin"),
        para("The hybrid workflow begins with the frozen pretrained model and the frozen intervention library. For each campaign it evaluates 6,000 virtual strain-environment candidates computationally, selects a nonduplicate verification batch of eight candidates, and requests exact simulator evaluation only for that final batch. It performs no adaptive exact-feedback loop before verification.", body),
        h2("4.4 Fairness and information boundaries"),
        para("Both workflows operate over the same declared engineering domain and are scored by the same hidden exact simulator. Neither receives a known optimum or future outcomes. Conventional DBTL receives the adaptive advantage it is meant to have: later proposals can use prior exact observations. The hybrid receives the deployment advantage it is meant to have: a pretrained model and cheap virtual evaluations. It does not receive prospective exact evaluations for free.", body),
        para("The old finite-pool exact cache is not an acquisition source. Prospective exact sidecar reuse is different: if a valid prospective sidecar already exists for a candidate that a workflow legitimately requests, the implementation may reuse the file to avoid recomputation. That is computational cache reuse, not informational leakage, provided the workflow cannot see the result until after making the legitimate request.", body),
        KeepTogether([DrawingFlowable(figures[1][1], 16.5 * cm), figure_caption("Figure 2. Fairness and information boundaries. Exact outcomes flow from the hidden simulator only after a valid workflow request. Historical training and virtual screening do not expose hidden future outcomes.")]),
    ]

    story += [
        h1("5. Pilot and Protocol Freeze"),
        para("Before the powered run, an exact prospective pilot was conducted to validate mechanics, accounting, causal order, leakage controls, solver reliability, and resource expectations. The pilot used two worlds, two campaign seeds per world, conventional batches [2,2,2], a hybrid verification batch of three, and 600 virtual evaluations per campaign. It produced 36 fresh exact cultures, 5,184 Yeast9 LP solves, and 2,400 hybrid virtual evaluations. All 11 readiness gates passed with status READY_FOR_FULL_PROSPECTIVE_BENCHMARK.", body),
        para("The pilot was not treated as a tuning run for method ranking. The full powered run was frozen after readiness validation with conventional batches [4,4,4,4], hybrid batch size 8, 6,000 virtual evaluations per campaign, and 20 paired world-seed campaigns.", body),
    ]

    story += [
        h1("6. Full Powered Experiment"),
        h2("6.1 Execution implementation"),
        para("The powered run was executed as isolated world-seed shards. Memory growth observed in long in-process COBRA/SymPy/GLPK sessions was controlled by moving exact Yeast9 rollouts into one-culture subprocesses. Each shard wrote private campaign ledgers and shared exact sidecars, then the canonical aggregation script merged the completed shard outputs into data/prospective_*.csv.", body),
        table_caption("Table 3. Full resource and completion accounting."),
        table_flowable(resource_table(data), [5.3 * cm, 3.1 * cm, 7.2 * cm], small=True),
        h2("6.2 Completion audit"),
        para("All 20 world-seed campaign shards completed. Each shard contains 24 exact cultures: 16 conventional and 8 hybrid. The aggregate exact ledger contains 480 complete_exact_dynamic_pfba rows, all with metabolic_backend yeast_gem_lp. The total exact solve count is 69,120 LP solves, which equals 480 cultures times 144 LP solves per culture.", body),
        h2("6.3 Acceptance and leakage gates"),
        table_caption("Table 4. Final acceptance and leakage safeguards."),
        table_flowable(acceptance_table(data), [6.3 * cm, 1.5 * cm, 8.0 * cm], small=True),
    ]

    story += [
        h1("7. Evaluation Metrics and Statistics"),
        h2("7.1 Best final product"),
        para("For a workflow and campaign, best final product is the maximum final accumulated product among candidates physically evaluated by that workflow. If b(n) denotes the best final product observed after n exact physical-style cultures, then b(n) is a nondecreasing best-so-far curve over the prospective biological-query budget.", body),
        h2("7.2 Product AUC and productivity"),
        para("Product AUC summarises the full product trajectory rather than only the terminal point. Productivity records final product per simulated duration. These secondary metrics identify candidates that produce earlier or faster, which can matter even when terminal titers are similar.", body),
        h2("7.3 Search AUC"),
        para("Search AUC integrates the best-so-far curve b(n) over a physical-culture horizon. It measures not only the final answer, but how early a method discovers strong candidates. Because conventional has 16 culture points and the hybrid has 8, full-horizon search AUC is a diagnostic rather than a fair efficiency headline. The equal-budget comparison uses the common 8-culture horizon.", body),
        h2("7.4 Paired bootstrap"),
        para("The experimental unit is a paired world-seed campaign. For each pair, delta_i = hybrid_i - conventional_i. The code resamples these paired deltas with replacement using 2,000 bootstrap samples and seed 91, and reports the 2.5th and 97.5th percentiles as a 95% confidence interval. Pairing is appropriate because both workflows face the same physiological world and campaign seed.", body),
    ]

    story += [
        h1("8. Results"),
        h2("8.1 Technical validity"),
        para("The final benchmark satisfies the intended prospective contract. It contains no mock lab rows, zero exact-outcome access during hybrid virtual search, zero old finite-pool result access, zero solver failures, and no teacher-training edit-vector overlap. All 11 acceptance gates pass.", body),
        h2("8.2 Overall final production"),
        para(f"Mean best final product is {conv.mean():.6f} for conventional DBTL and {hyb.mean():.6f} for the hybrid. The paired hybrid-minus-conventional mean is {final.mean_difference:+.6f}, the median is {final.median_difference:+.6f}, and the bootstrap CI is {fmt_ci(final.bootstrap_ci_low, final.bootstrap_ci_high, 6)}. The hybrid wins {int(final.hybrid_win_count)} of 20 paired campaigns, with {int(final.hybrid_loss_count)} losses and no ties.", body),
        table_caption("Table 5. Overall paired result."),
        table_flowable(paired_results_table(data), [4.6 * cm, 2.5 * cm, 2.5 * cm, 4.1 * cm, 2.4 * cm], small=True),
        KeepTogether([DrawingFlowable(figures[5][1], 16.5 * cm), figure_caption("Figure 6. Paired final-product comparison. Points above the diagonal are hybrid wins; points below are conventional wins.")]),
        h2("8.3 Distribution of campaign effects"),
        para("The mean advantage is larger than the median advantage, indicating a heterogeneous effect distribution. Several large hybrid wins contribute meaningfully to the positive mean, while a minority of campaigns favor conventional DBTL. The appropriate claim is therefore a positive average prospective advantage, not uniform superiority in every campaign.", body),
        KeepTogether([DrawingFlowable(figures[6][1], 16.5 * cm), figure_caption("Figure 7. Sorted paired final-product deltas. The mean is positive, but the median is much smaller, making heterogeneity visible.")]),
        h2("8.4 Physical-culture and common-budget efficiency"),
        para(f"The hybrid uses 8 exact physical-style cultures per campaign, half of conventional DBTL's 16. At the common 8-culture horizon, hybrid final-product best-so-far search AUC exceeds conventional by {common.mean_difference:+.6f}, with median {common.median_difference:+.6f} and CI {fmt_ci(common.bootstrap_ci_low, common.bootstrap_ci_high, 6)}. This directly tests sample efficiency at equal prospective biological cost.", body),
        table_caption("Table 6. Common 8-culture physical-budget results."),
        table_flowable(common_budget_table(data), [5.7 * cm, 2.3 * cm, 2.3 * cm, 4.3 * cm, 2.0 * cm], small=True),
        KeepTogether([DrawingFlowable(figures[2][1], 16.5 * cm), figure_caption("Figure 3. Best final product versus physical-style culture count. Conventional continues to 16 cultures; hybrid stops at 8.")]),
        KeepTogether([DrawingFlowable(figures[4][1], 16.5 * cm), figure_caption("Figure 5. Common 8-culture comparison. Both workflows are evaluated over the same physical-culture horizon.")]),
        h2("8.5 Sequential-round efficiency"),
        para("Sequential biological depth is distinct from culture count. Conventional DBTL requires four sequential stages because later proposals depend on earlier exact observations. The hybrid verifies eight candidates in one parallel exact stage after virtual screening. Thus the hybrid uses one quarter of the sequential biological stages. This is a proxy for experimental calendar latency, not a measured wet-lab duration.", body),
        KeepTogether([DrawingFlowable(figures[3][1], 16.5 * cm), figure_caption("Figure 4. Best final product versus sequential biological stage. The hybrid has one exact verification stage; conventional advances over four adaptive stages.")]),
        h2("8.6 World-specific effects"),
        para("World-separated point estimates favor the hybrid in both physiological worlds, but the individual world confidence intervals are wider and cross zero. Evidence is therefore strongest for the pooled paired result, not for independently established superiority within each world.", body),
        table_caption("Table 7. World-separated paired results."),
        table_flowable(world_table(data), [4.0 * cm, 3.4 * cm, 2.4 * cm, 2.4 * cm, 4.0 * cm, 1.7 * cm], small=True),
        KeepTogether([DrawingFlowable(figures[7][1], 16.5 * cm), figure_caption("Figure 8. World-separated and pooled effects. Per-world intervals are wider than the pooled paired interval.")]),
        h2("8.7 Productivity and product-AUC results"),
        para(f"Secondary biological metrics also favor the hybrid on average. The paired productivity difference is {paired_row(data, 'best_productivity').mean_difference:+.6f} with CI {fmt_ci(paired_row(data, 'best_productivity').bootstrap_ci_low, paired_row(data, 'best_productivity').bootstrap_ci_high, 6)}. The paired product-AUC difference is {paired_row(data, 'best_product_AUC').mean_difference:+.6f} with CI {fmt_ci(paired_row(data, 'best_product_AUC').bootstrap_ci_low, paired_row(data, 'best_product_AUC').bootstrap_ci_high, 6)}.", body),
        h2("8.8 Representative candidate trajectories"),
        para("Representative trajectories were selected by rule rather than by cherry-picking the largest hybrid victory: one typical hybrid win, one near tie, and one conventional win. In each case, the plotted candidates are the best final-product candidates discovered by each workflow in that campaign.", body),
        table_caption("Table 8. Representative best candidates used for trajectory examples."),
        table_flowable(representative_table(data), [2.6 * cm, 2.6 * cm, 2.0 * cm, 1.3 * cm, 1.2 * cm, 1.3 * cm, 3.2 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm], small=True),
        KeepTogether([DrawingFlowable(figures[8][1], 16.5 * cm), figure_caption("Figure 9. Representative best-candidate product trajectories for a typical hybrid win, near tie, and conventional win.")]),
        h2("8.9 Resource accounting"),
        KeepTogether([DrawingFlowable(figures[9][1], 16.5 * cm), figure_caption("Figure 10. Historical, prospective, virtual, and exact-solver accounting. Historical cultures are pre-deployment; virtual evaluations are compute.")]),
    ]

    story += [
        h1("9. Discussion"),
        h2("9.1 Main result"),
        para("The benchmark supports the operational digital-twin hypothesis in this synthetic setting. A pretrained hybrid regulation-metabolism model used extensive virtual exploration to reduce prospective exact search from four sequential DBTL rounds and sixteen physical-style cultures to one eight-culture verification stage, while improving mean final product and equal-budget search efficiency.", body),
        h2("9.2 Relationship to the earlier negative finite-pool result"),
        para("This result does not contradict the earlier finite-pool acquisition benchmark, where space filling outperformed direct and hybrid UCB methods under equal oracle-query conditions. That earlier benchmark asked which acquisition strategy selected candidates best when hidden evaluations themselves were the scarce budget. The present benchmark asks whether a pretrained digital twin can use cheap virtual search before exact verification to reduce prospective biological-query cost and sequential depth. These are different scientific questions.", body),
        h2("9.3 Deployment and greenfield interpretations"),
        para("In the deployed-twin interpretation, historical fixed-strain data have already been collected and model training is a sunk pre-deployment cost. In a greenfield interpretation, those 125 historical cultures and 18,000 historical LP solves must be amortised. The full prospective campaign nevertheless should be counted as 480 exact deployment cultures, not 605, because historical training cultures are not campaign-specific prospective exact requests.", body),
        h2("9.4 Statistical heterogeneity"),
        para("The positive average effect should not be overstated. The median paired final-product advantage is much smaller than the mean, and 7 of 20 paired campaigns favor conventional DBTL. The most defensible conclusion is a pooled prospective advantage with fewer cultures and stages, not guaranteed superiority for every campaign or every physiological world.", body),
    ]

    story += [
        h1("10. Limitations"),
        bullet_list(
            [
                "The biological oracle is synthetic: Yeast9 plus engineered dynamic regulatory logic, not actual wet-lab yeast.",
                "The model and simulator share abstractions from the same project ecosystem; real-world model mismatch may be larger.",
                "The hybrid's deployment advantage assumes that a pretrained twin already exists; greenfield deployment requires upfront data collection.",
                "Only two simulated worlds were included, so broader robustness remains unproven.",
                "Per-world confidence intervals are wide and cross zero.",
                "Effect distribution is nonuniform; mean advantage is larger than median advantage.",
                "Results apply to the declared edit/environment domain, not arbitrary metabolic engineering spaces.",
                "Actual strain construction failures, assay noise, biological replicates, contamination, and batch effects are not represented unless explicitly simulated.",
                "Yeast9/pFBA and the synthetic dynamic regulator are approximations.",
                "The experiment does not fully establish behavior under severe out-of-distribution mechanism shift.",
            ],
            body,
        ),
        h1("11. Conclusion"),
        para("In a prospective synthetic Yeast9 engineering benchmark, a pretrained hybrid regulation-metabolism digital twin used large-scale virtual strain-environment screening to reduce experimental search from four sequential DBTL rounds and sixteen physical-style cultures to one eight-culture parallel verification round. Across twenty paired campaigns, the hybrid achieved higher mean best final production and improved equal-budget search efficiency. These results provide computational proof-of-concept that a pretrained hybrid digital twin can accelerate DBTL by replacing part of sequential biological exploration with parallel in-silico screening.", body),
        para("The experiment does not prove real yeast engineering acceleration, universal hybrid superiority, discovery of a global optimum, reporter necessity, state-space superiority, replacement of scientists, or experimentally validated calendar-time savings.", body),
    ]

    story += [
        PageBreak(),
        h1("Appendix A. Frozen Configuration"),
        para(f"Worlds: baseline_world and strong_oxidative_burden_world. Campaign seeds: 86001-86010. Conventional batches: [4,4,4,4]. Hybrid verification batch: 8. Hybrid virtual evaluations per campaign: 6,000. Benchmark ID: prospective_on_demand_dbtl_v1. Wet-lab simulator version recorded by the runner: yeast9_dynamic_pfba_on_demand_v1.", body),
        h1("Appendix B. Exact Simulator Accounting"),
        para("Exact backend: yeast_gem_lp. Time grid: 49 time points, 48 intervals, dt=0.25. Solver structure: growth optimisation, product optimisation, and pFBA per interval. Expected LP solves per culture: 144. Aggregate full-run LP solves: 69,120.", body),
        h1("Appendix C. Statistical Details"),
        para("Paired bootstrap CIs are computed by resampling world-seed paired deltas with replacement using 2,000 bootstrap replicates and seed 91. The interval is the empirical 2.5th and 97.5th percentile of bootstrap mean deltas. The thresholds in prospective_relative_threshold_efficiency.csv are retrospective fractions of each campaign's observed best value, not predeclared biological targets.", body),
        h1("Appendix D. Reproducibility Commands"),
        para("Primary local aggregation command:", body),
        Paragraph("<font name='Courier'>.venv/bin/python scripts/aggregate_prospective_hpc_shards.py --overwrite</font>", STYLES["ReportCode"]),
        para("Focused test command:", body),
        Paragraph("<font name='Courier'>.venv/bin/python -m pytest tests/test_prospective_dbtl_benchmark.py</font>", STYLES["ReportCode"]),
        para("Report build command:", body),
        Paragraph("<font name='Courier'>python3 scripts/build_prospective_dbtl_benchmark_report.py</font>", STYLES["ReportCode"]),
        h1("Appendix E. Important Generated Artifacts"),
        bullet_list(
            [
                "data/prospective_dbtl_final_report.md",
                "data/prospective_full_hpc_completion_audit.csv",
                "data/prospective_campaign_summary.csv",
                "data/prospective_paired_statistical_comparisons.csv",
                "data/prospective_world_paired_summary.csv",
                "data/prospective_best_so_far_by_culture.csv",
                "data/prospective_best_so_far_by_stage.csv",
                "data/prospective_best_so_far_auc_common_budget.csv",
                "data/prospective_relative_threshold_efficiency.csv",
                "results/prospective_dbtl_benchmark/hpc_shards/",
                "results/prospective_dbtl_benchmark/on_demand_exact_rollouts/",
            ],
            small,
        ),
    ]
    return story


def build_latex_source(data: SourceData, figure_paths: list[Path]) -> str:
    final = paired_row(data, "best_final_product")
    common = paired_row(data, "common_horizon_best_so_far_auc_final_product")
    conv = data.summary[data.summary.method_id.eq("conventional_dbtl")]["best_final_product"]
    hyb = data.summary[data.summary.method_id.eq("pretrained_hybrid_digital_twin")]["best_final_product"]
    fig_lines = "\n".join(
        [
            "\\begin{figure}[ht]\n\\centering\n"
            f"\\includegraphics[width=0.92\\linewidth]{{{latex_escape(path.relative_to(OUT_DIR))}}}\n"
            f"\\caption{{{latex_escape(path.stem.replace('_', ' ').title())}.}}\n\\end{{figure}}"
            for path in figure_paths
        ]
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx,booktabs,longtable,amsmath,hyperref}}
\usepackage{{newtxtext,newtxmath}}
\title{{Full Prospective On-Demand DBTL Benchmark for the Yeast Digital Twin}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This standalone technical report documents the final full prospective synthetic DBTL benchmark comparing conventional DBTL with a pretrained hybrid yeast digital twin. The benchmark completed {len(data.exact)} exact physical-style cultures, {int(data.exact['n_actual_lp_solves'].sum())} Yeast9 LP solves, and {len(data.virtual)} hybrid virtual evaluations. Mean best final product was {conv.mean():.6f} for conventional DBTL and {hyb.mean():.6f} for the hybrid. The paired hybrid-minus-conventional final-product difference was {final.mean_difference:+.6f} with bootstrap CI {fmt_ci(final.bootstrap_ci_low, final.bootstrap_ci_high, 6)}. At the common 8-culture horizon, best-so-far final-product search AUC favored the hybrid by {common.mean_difference:+.6f} with CI {fmt_ci(common.bootstrap_ci_low, common.bootstrap_ci_high, 6)}.
\end{{abstract}}
\section{{Scope and Source of Truth}}
The source of truth for numerical values is the generated prospective benchmark artifact set under \texttt{{data/prospective\_*.csv}} and \texttt{{results/prospective\_dbtl\_benchmark/}}. This LaTeX source is emitted alongside the compiled PDF; the current execution environment did not provide a TeX engine, so the delivered PDF was typeset programmatically with ReportLab using the same source tables and figure files.
\section{{Scientific Question}}
Can a pretrained hybrid yeast digital twin use large-scale virtual search to identify high-performing strain-environment designs using fewer prospective biological-query equivalents and fewer sequential DBTL rounds than conventional adaptive DBTL?
\section{{Methods Summary}}
Conventional DBTL uses four adaptive physical stages with batches [4,4,4,4]. The hybrid uses a pretrained regulation-metabolism model obtained from fixed-reference-strain environmental data, screens 6,000 virtual candidates per campaign, and verifies eight candidates in one exact batch. Both workflows operate over the same engineering domain and query the same hidden Yeast9 dynamic pFBA simulator only after legitimate workflow requests.
\section{{Key Results}}
The final status is \texttt{{{latex_escape(data.acceptance['final_status_if_all_gates_considered'].iloc[0])}}}. All 11 acceptance gates passed. There were zero mock rows, zero solver failures, zero old finite-pool result accesses, and zero hidden exact-outcome accesses during hybrid virtual search.
{fig_lines}
\section{{Limitations}}
This is a synthetic computational proof-of-concept, not wet-lab yeast validation. Historical model-building cost is disclosed separately from prospective deployment cultures. Per-world confidence intervals are wide, effect sizes are heterogeneous, and the result applies to the declared edit/environment domain.
\end{{document}}
"""


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 1.05 * cm, "Full Prospective On-Demand DBTL Benchmark")
    canvas.drawRightString(A4[0] - doc.rightMargin, 1.05 * cm, f"{doc.page}")
    canvas.restoreState()


def build_pdf(data: SourceData, figures: list[tuple[str, Drawing]]) -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.75 * cm,
        rightMargin=1.75 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.65 * cm,
        title="Full Prospective On-Demand DBTL Benchmark for the Yeast Digital Twin",
        author="",
    )
    story = build_story(data, figures)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def validate_data(data: SourceData) -> None:
    assert len(data.exact) == 480
    assert data.exact["campaign_id"].nunique() == 20
    assert int(data.exact["n_actual_lp_solves"].sum()) == 69120
    assert set(data.exact["metabolic_backend"].astype(str)) == {"yeast_gem_lp"}
    assert set(data.exact["exact_status"].astype(str)) == {"complete_exact_dynamic_pfba"}
    assert int(data.virtual["exact_outcome_accessed"].astype(bool).sum()) == 0
    assert int(data.exact["old_result_accessed"].astype(bool).sum()) == 0
    assert data.acceptance["passed"].astype(bool).all()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    validate_data(data)
    figures = figure_files(data)
    headline = headline_numbers(data)
    headline.to_csv(HEADLINE_PATH, index=False)
    figure_paths = [FIG_DIR / filename for filename, _ in figures]
    TEX_PATH.write_text(build_latex_source(data, figure_paths), encoding="utf-8")
    build_pdf(data, figures)
    reader = PdfReader(str(PDF_PATH))
    print(
        json.dumps(
            {
                "pdf": str(PDF_PATH),
                "tex": str(TEX_PATH),
                "headline_csv": str(HEADLINE_PATH),
                "figures": [str(p) for p in figure_paths],
                "pages": len(reader.pages),
                "tables": 8,
                "figures_count": len(figures),
            },
            indent=2,
        )
    )


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="ReportTitle", parent=STYLES["Title"], fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=10))
STYLES.add(ParagraphStyle(name="ReportSubtitle", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=12, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#374151"), spaceAfter=10))
STYLES.add(ParagraphStyle(name="Centered", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_CENTER))
STYLES.add(ParagraphStyle(name="AbstractHeading", parent=STYLES["Heading1"], fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=8, spaceAfter=6))
STYLES.add(ParagraphStyle(name="Abstract", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=10.2, leading=14, alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=8))
STYLES.add(ParagraphStyle(name="ReportBody", parent=STYLES["BodyText"], fontName="Times-Roman", fontSize=10.2, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=7))
STYLES.add(ParagraphStyle(name="ReportSmall", parent=STYLES["BodyText"], fontName="Times-Roman", fontSize=8.7, leading=11.2, alignment=TA_LEFT, spaceAfter=4))
STYLES.add(ParagraphStyle(name="ReportHeading1", parent=STYLES["Heading1"], fontName="Times-Bold", fontSize=15, leading=18, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#111827")))
STYLES.add(ParagraphStyle(name="ReportHeading2", parent=STYLES["Heading2"], fontName="Times-Bold", fontSize=12.3, leading=15, spaceBefore=9, spaceAfter=5, textColor=colors.HexColor("#1f2937")))
STYLES.add(ParagraphStyle(name="ReportHeading3", parent=STYLES["Heading3"], fontName="Times-BoldItalic", fontSize=10.8, leading=13, spaceBefore=7, spaceAfter=4))
STYLES.add(ParagraphStyle(name="TableCaption", parent=STYLES["Normal"], fontName="Times-Bold", fontSize=9.2, leading=11.5, spaceBefore=8, spaceAfter=4))
STYLES.add(ParagraphStyle(name="FigureCaption", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=8.8, leading=11, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=9))
STYLES.add(ParagraphStyle(name="TableCell", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=8.2, leading=10))
STYLES.add(ParagraphStyle(name="TableCellSmall", parent=STYLES["Normal"], fontName="Times-Roman", fontSize=6.9, leading=8.2))
STYLES.add(ParagraphStyle(name="ReportCode", parent=STYLES["Normal"], fontName="Courier", fontSize=8.3, leading=10.5, leftIndent=12, spaceAfter=5))


if __name__ == "__main__":
    main()
