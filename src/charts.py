"""Altair charts shared by the app pages. Each function takes plain data and returns a chart."""
from __future__ import annotations

import altair as alt
import pandas as pd

# Validated categorical slots (dataviz reference palette, light mode), plus app accents.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RISK = "#d1495b"          # app primary: "on the call list" / raises risk
MUTED = "#a8a7a2"         # de-emphasised marks: baselines, "not called"
INK_2 = "#52514e"         # secondary text


def _pct(title: str, domain=(0, 1)) -> dict:
    return dict(title=title, axis=alt.Axis(format="%"), scale=alt.Scale(domain=list(domain)))


def risk_histogram(risk: pd.Series, threshold: float) -> alt.Chart:
    """Spread of scores, with everyone at or above the threshold highlighted."""
    df = pd.DataFrame({"risk": risk})
    df["group"] = (df["risk"] >= threshold).map({True: "On call list", False: "Not called"})
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, binSpacing=2).encode(
        x=alt.X("risk:Q", bin=alt.Bin(extent=[0, 1], step=0.05), **_pct("Churn risk")),
        y=alt.Y("count():Q", title="Customers"),
        color=alt.Color("group:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["On call list", "Not called"], range=[RISK, MUTED])),
        tooltip=[alt.Tooltip("group:N", title=" "), alt.Tooltip("count():Q", title="Customers")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=INK_2, strokeWidth=1.5).encode(
        x=alt.X("t:Q", **_pct("Churn risk")), tooltip=[alt.Tooltip("t:Q", title="Threshold", format=".0%")])
    return (bars + rule).properties(height=200)


def what_if(now: float, after: float, threshold: float) -> alt.Chart:
    """Dumbbell: the customer's risk now and with the offer, on one 0 to 100% scale."""
    df = pd.DataFrame({"when": ["Now", "With offer"], "risk": [now, after], "row": ["", ""]})
    base = alt.Chart(df).encode(y=alt.Y("row:N", title=None, axis=None),
                                x=alt.X("risk:Q", **_pct("Churn risk")))
    line = base.mark_line(color=MUTED, strokeWidth=2)
    dots = base.mark_point(filled=True, size=160, opacity=1, stroke="white", strokeWidth=2).encode(
        color=alt.Color("when:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Now", "With offer"], range=[RISK, BLUE])),
        tooltip=["when:N", alt.Tooltip("risk:Q", format=".0%")])
    labels = base.mark_text(dy=-16, color=INK_2, fontSize=12).encode(text=alt.Text("risk:Q", format=".0%"))
    t = pd.DataFrame({"risk": [threshold], "label": ["call-list threshold"]})
    rule = alt.Chart(t).mark_rule(color=INK_2, strokeWidth=1).encode(
        x=alt.X("risk:Q", **_pct("Churn risk")),
        tooltip=[alt.Tooltip("risk:Q", title="Call-list threshold", format=".0%")])
    rule_text = alt.Chart(t).mark_text(align="right", dx=-4, baseline="top", y=2, color=INK_2,
                                       fontSize=11).encode(x="risk:Q", text="label:N")
    return (rule + rule_text + line + dots + labels).properties(height=90)


def gain_chart(gain: pd.DataFrame, mark_at: float = 0.3) -> alt.Chart:
    """Share of churners reached vs. share of customers called, model vs. random calls."""
    long = gain.melt("share_called", var_name="series", value_name="reached")
    long["series"] = long["series"].map({"model": "Model ranking", "random": "Random calls"})
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Model ranking", "Random calls"], range=[BLUE, MUTED]))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("share_called:Q", **_pct("Share of customers called")),
        y=alt.Y("reached:Q", **_pct("Share of churners reached")),
        color=color)
    hover = alt.selection_point(fields=["share_called"], nearest=True, on="pointerover", empty=False)
    points = alt.Chart(gain).mark_rule(color=INK_2, strokeWidth=1).encode(
        x="share_called:Q", opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("share_called:Q", title="Called", format=".0%"),
                 alt.Tooltip("model:Q", title="Churners reached (model)", format=".0%"),
                 alt.Tooltip("random:Q", title="Churners reached (random)", format=".0%")],
    ).add_params(hover)
    m = gain.iloc[(gain["share_called"] - mark_at).abs().argmin()]
    mark = pd.DataFrame({"x": [m["share_called"]], "y": [m["model"]],
                         "label": [f"Top {m['share_called']:.0%} reaches {m['model']:.0%} of churners"]})
    dot = alt.Chart(mark).mark_point(filled=True, size=90, color=BLUE, stroke="white", strokeWidth=2,
                                     opacity=1).encode(x="x:Q", y="y:Q")
    text = alt.Chart(mark).mark_text(align="left", dx=10, dy=12, color=INK_2, fontSize=12).encode(
        x="x:Q", y="y:Q", text="label:N")
    return (lines + points + dot + text).properties(height=320)


def threshold_chart(table: pd.DataFrame, threshold: float) -> alt.Chart:
    """Precision, recall and list size as the threshold moves, with the current setting marked."""
    names = {"precision": "Called who'd really leave", "recall": "Churners reached", "share_called": "Customers called"}
    long = table.melt("threshold", list(names), var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(names)
    order = list(names.values())
    color = alt.Color("metric:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=order, range=[BLUE, ORANGE, AQUA]))
    dom = [float(table["threshold"].min()), float(table["threshold"].max())]
    x = alt.X("threshold:Q", title="Call-list threshold", axis=alt.Axis(format="%"),
              scale=alt.Scale(domain=dom, zero=False, nice=False))
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30, filled=True)).encode(
        x=x, y=alt.Y("value:Q", **_pct("")), color=color,
        tooltip=[alt.Tooltip("threshold:Q", format=".0%"), "metric:N", alt.Tooltip("value:Q", format=".0%")])
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold], "label": ["current"]})).mark_rule(
        color=INK_2, strokeWidth=1.5).encode(
        x=x, tooltip=[alt.Tooltip("threshold:Q", title="Current threshold", format=".0%")])
    rule_text = alt.Chart(pd.DataFrame({"threshold": [threshold], "label": ["current"]})).mark_text(
        align="left", dx=4, baseline="top", y=2, color=INK_2, fontSize=11).encode(x=x, text="label:N")
    return (rule + rule_text + lines).properties(height=320)


def model_comparison(cv: pd.DataFrame, chosen: str) -> alt.Chart:
    """Cross-validated ROC AUC per candidate (± 1 sd), chosen model highlighted."""
    df = cv.assign(lo=cv["roc_auc"] - cv["roc_auc_sd"].fillna(0), hi=cv["roc_auc"] + cv["roc_auc_sd"].fillna(0),
                   pick=(cv["model"] == chosen).map({True: "Chosen", False: "Other"}))
    y = alt.Y("model:N", sort=alt.EncodingSortField("roc_auc", order="descending"), title=None)
    x_scale = alt.Scale(domain=[0.45, 0.9], zero=False, nice=False)
    err = alt.Chart(df).mark_rule(strokeWidth=2, color=MUTED).encode(
        y=y, x=alt.X("lo:Q", scale=x_scale, title="ROC AUC (5-fold CV, training set)"), x2="hi:Q")
    dots = alt.Chart(df).mark_point(filled=True, size=120, opacity=1, stroke="white", strokeWidth=2).encode(
        y=y, x=alt.X("roc_auc:Q", scale=x_scale),
        color=alt.Color("pick:N", legend=None, scale=alt.Scale(domain=["Chosen", "Other"], range=[BLUE, MUTED])),
        tooltip=["model:N", alt.Tooltip("roc_auc:Q", title="ROC AUC", format=".3f"),
                 alt.Tooltip("roc_auc_sd:Q", title="± sd", format=".3f"),
                 alt.Tooltip("pr_auc:Q", title="PR AUC", format=".3f")])
    values = alt.Chart(df).mark_text(align="left", dx=12, color=INK_2, fontSize=11).encode(
        y=y, x=alt.X("hi:Q", scale=x_scale), text=alt.Text("roc_auc:Q", format=".2f"))
    coin = alt.Chart(pd.DataFrame({"x": [0.5], "label": ["coin flip"]}))
    coin_rule = coin.mark_rule(color=MUTED, strokeWidth=1).encode(x=alt.X("x:Q", scale=x_scale))
    coin_text = coin.mark_text(align="left", dx=4, baseline="top", y=2, color=INK_2, fontSize=11).encode(
        x=alt.X("x:Q", scale=x_scale), text="label:N")
    return (coin_rule + coin_text + err + dots + values).properties(height=230)


def calibration_chart(bins: pd.DataFrame) -> alt.Chart:
    """Average score in each bin vs. how many of those customers really left."""
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeWidth=1.5).encode(
        x=alt.X("x:Q", **_pct("Model score (average in bin)")), y=alt.Y("y:Q", **_pct("Customers who really left")))
    diag_label = alt.Chart(pd.DataFrame({"x": [0.72], "y": [0.78], "t": ["score = reality"]})).mark_text(
        color=INK_2, fontSize=11, angle=0, align="right").encode(x="x:Q", y="y:Q", text="t:N")
    line = alt.Chart(bins).mark_line(color=BLUE, strokeWidth=2).encode(x="predicted:Q", y="actual:Q")
    dots = alt.Chart(bins).mark_point(filled=True, color=BLUE, size=70, opacity=1, stroke="white",
                                      strokeWidth=2).encode(
        x="predicted:Q", y="actual:Q",
        tooltip=[alt.Tooltip("predicted:Q", title="Average score", format=".0%"),
                 alt.Tooltip("actual:Q", title="Really left", format=".0%"),
                 alt.Tooltip("n:Q", title="Customers")])
    return (diag + diag_label + line + dots).properties(height=320)


def segment_chart(seg: pd.DataFrame) -> alt.Chart:
    """Real churn rate vs. share put on the call list, per segment."""
    names = {"churn_rate": "Really churned", "share_flagged": "Put on call list"}
    long = seg.melt(["segment", "customers"], list(names), var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(names)
    return alt.Chart(long).mark_bar(cornerRadiusEnd=3, height={"band": 0.8}).encode(
        y=alt.Y("segment:N", title=None),
        yOffset=alt.YOffset("metric:N", sort=list(names.values())),
        x=alt.X("value:Q", **_pct("Share of segment")),
        color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(names.values()), range=[BLUE, ORANGE])),
        tooltip=["segment:N", "metric:N", alt.Tooltip("value:Q", format=".0%"), "customers:Q"],
    ).properties(height=alt.Step(16))
