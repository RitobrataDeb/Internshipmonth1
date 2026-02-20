import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────
BG         = "#0A0E17"
PANEL      = "#0D1320"
BORDER     = "#1A2D45"
TEXT       = "#C9D1E3"
MUTED      = "#4A6080"
ACCENT     = "#00D4FF"
ACCENT2    = "#FFB347"
ACCENT3    = "#C084FC"
SUCCESS    = "#2ECC71"
DANGER     = "#E63946"

FONT       = "IBM Plex Mono, Courier New, monospace"
TITLE_FONT = "DM Serif Display, Georgia, serif"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PANEL,
        plot_bgcolor=BG,
        font=dict(family=FONT, color=TEXT, size=11),
        title_font=dict(family=FONT, color=TEXT, size=13),
        colorway=[ACCENT, ACCENT2, ACCENT3, SUCCESS, DANGER,
                  "#FF6B9D", "#A8E6CF", "#FFD93D"],
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT, size=11)),
        margin=dict(l=50, r=20, t=50, b=50),
    )
)

SCATTER_COLORS = [ACCENT, ACCENT2, ACCENT3, SUCCESS, DANGER,
                  "#FF6B9D", "#A8E6CF", "#FFD93D"]

# ─────────────────────────────────────────────────────────
#  BUILT-IN DATASETS
# ─────────────────────────────────────────────────────────
def load_dataset(name: str) -> pd.DataFrame:
    if name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "cancer":
        d = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["target"] = [d.target_names[t] for t in d.target]
    return df


DATASETS = {
    "iris":   "Iris Flowers (150 × 5)",
    "wine":   "Wine Recognition (178 × 14)",
    "cancer": "Breast Cancer (569 × 31)",
}

# ─────────────────────────────────────────────────────────
#  STATS HELPERS
# ─────────────────────────────────────────────────────────
def pearson(xs, ys):
    if len(xs) < 3:
        return 0.0
    r, _ = stats.pearsonr(xs, ys)
    return r


def describe_series(s: pd.Series) -> dict:
    desc = s.describe()
    return {
        "n":      int(desc["count"]),
        "mean":   f"{desc['mean']:.3f}",
        "std":    f"{desc['std']:.3f}",
        "min":    f"{desc['min']:.3f}",
        "25%":    f"{desc['25%']:.3f}",
        "median": f"{desc['50%']:.3f}",
        "75%":    f"{desc['75%']:.3f}",
        "max":    f"{desc['max']:.3f}",
    }


# ─────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Feature Lab",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Shared store for the active dataframe (as JSON)
_df_cache: dict[str, pd.DataFrame] = {"current": load_dataset("iris")}


# ─────────────────────────────────────────────────────────
#  LAYOUT HELPERS
# ─────────────────────────────────────────────────────────
def label(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize": "9px", "letterSpacing": "0.12em",
        "textTransform": "uppercase", "color": MUTED,
        "fontWeight": "700", "marginBottom": "6px",
    })


def card(children, style=None):
    base = {
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding": "18px 20px",
    }
    base.update(style or {})
    return html.Div(children, style=base)


def stat_box(lbl, val_id):
    return html.Div([
        html.Div(lbl, style={"fontSize": "9px", "color": MUTED,
                              "letterSpacing": "0.1em", "textTransform": "uppercase"}),
        html.Div("—", id=val_id, style={
            "fontSize": "18px", "fontWeight": "700",
            "color": ACCENT, "marginTop": "2px",
        }),
    ], style={
        "background": BG, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "10px 14px",
    })


def dropdown(id_, options, value, multi=False):
    return dcc.Dropdown(
        id=id_, options=options, value=value, multi=multi,
        clearable=False,
        style={"fontFamily": FONT, "fontSize": "12px"},
        className="dark-dropdown",
    )


# ─────────────────────────────────────────────────────────
#  INJECT CSS  (Dash loads everything in ./assets/ auto)
# ─────────────────────────────────────────────────────────
import os
os.makedirs("assets", exist_ok=True)
with open("assets/custom.css", "w") as _f:
    _f.write("""
/* Dark dropdown overrides */
.dark-dropdown .Select-control {
    background: #0A0E17 !important;
    border: 1px solid #1A2D45 !important;
    color: #C9D1E3 !important;
}
.dark-dropdown .Select-menu-outer {
    background: #0D1320 !important;
    border: 1px solid #1A2D45 !important;
}
.dark-dropdown .Select-option {
    background: #0D1320 !important;
    color: #C9D1E3 !important;
}
.dark-dropdown .Select-option:hover,
.dark-dropdown .Select-option.is-focused {
    background: #111827 !important;
}
.dark-dropdown .Select-value-label { color: #C9D1E3 !important; }
.dark-dropdown .Select-arrow { border-top-color: #00D4FF !important; }
.dark-dropdown input { color: #C9D1E3 !important; background: transparent !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E17; }
::-webkit-scrollbar-thumb { background: #1A2D45; border-radius: 3px; }
""")

# ─────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────
app.layout = html.Div([

    # Google Fonts
    html.Link(rel="stylesheet", href=(
        "https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Mono:wght@400;500;700"
        "&family=DM+Serif+Display&display=swap"
    )),

    # ── HEADER ──────────────────────────────────────────
    html.Div([
        html.Div("◈", style={
            "width": "38px", "height": "38px",
            "background": "linear-gradient(135deg, #00D4FF, #0077AA)",
            "borderRadius": "9px", "display": "flex",
            "alignItems": "center", "justifyContent": "center",
            "fontSize": "18px", "boxShadow": f"0 0 20px {ACCENT}44",
            "flexShrink": "0",
        }),
        html.Div([
            html.Div("Feature Lab", style={
                "fontSize": "15px", "fontWeight": "700",
                "letterSpacing": "0.08em", "color": "#E8EDF5",
                "textTransform": "uppercase",
            }),
            html.Div("Dataset Relationship Explorer", style={
                "fontSize": "11px", "color": MUTED, "letterSpacing": "0.05em",
            }),
        ]),
        html.Div(id="obs-badge", style={
            "marginLeft": "auto",
            "background": "#0D1F35",
            "border": f"1px solid #1E3A5F",
            "borderRadius": "20px",
            "padding": "4px 14px",
            "fontSize": "11px", "color": ACCENT,
            "letterSpacing": "0.05em",
        }),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "14px",
        "padding": "16px 28px",
        "background": "#0D1320",
        "borderBottom": f"1px solid {BORDER}",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # ── BODY ────────────────────────────────────────────
    html.Div([

        # ── SIDEBAR ─────────────────────────────────────
        html.Div([

            # Dataset picker
            card([
                label("Dataset"),
                dropdown("ds-picker", [{"label": v, "value": k} for k, v in DATASETS.items()], "iris"),
            ], {"marginBottom": "14px"}),

            # Scatter controls
            card([
                label("Scatter Plot"),
                html.Div("X Axis", style={"fontSize": "10px", "color": MUTED, "marginBottom": "4px"}),
                dropdown("x-feat", [], None),
                html.Div("Y Axis", style={"fontSize": "10px", "color": MUTED,
                                          "marginTop": "8px", "marginBottom": "4px"}),
                dropdown("y-feat", [], None),
                html.Div("Colour By", style={"fontSize": "10px", "color": MUTED,
                                             "marginTop": "8px", "marginBottom": "4px"}),
                dropdown("hue-feat", [], None),
                html.Div([
                    dcc.Checklist(
                        id="show-reg",
                        options=[{"label": "  Regression line", "value": "yes"}],
                        value=["yes"],
                        style={"fontSize": "11px", "color": TEXT, "marginTop": "10px"},
                    ),
                    dcc.Checklist(
                        id="show-marginal",
                        options=[{"label": "  Marginal histograms", "value": "yes"}],
                        value=["yes"],
                        style={"fontSize": "11px", "color": TEXT, "marginTop": "6px"},
                    ),
                ]),
            ], {"marginBottom": "14px"}),

            # Histogram controls
            card([
                label("Histogram"),
                dropdown("hist-feat", [], None),
                html.Div("Bins", style={"fontSize": "10px", "color": MUTED,
                                        "marginTop": "10px", "marginBottom": "4px"}),
                dcc.Slider(id="bins-slider", min=5, max=60, step=1, value=20,
                           marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Div("Overlay KDE", style={"fontSize": "10px", "color": MUTED,
                                               "marginTop": "10px"}),
                dcc.Checklist(id="show-kde",
                              options=[{"label": "  Show KDE curve", "value": "yes"}],
                              value=["yes"],
                              style={"fontSize": "11px", "color": TEXT, "marginTop": "4px"}),
            ], {"marginBottom": "14px"}),

            # Pearson r readout
            card([
                label("Pearson  r"),
                html.Div(id="pearson-r", style={
                    "fontSize": "30px", "fontWeight": "700",
                    "color": ACCENT, "letterSpacing": "0.02em",
                }),
                html.Div(id="pearson-desc", style={
                    "fontSize": "11px", "color": MUTED, "marginTop": "4px",
                }),
            ]),

        ], style={
            "width": "240px", "flexShrink": "0",
            "display": "flex", "flexDirection": "column",
            "gap": "0",
        }),

        # ── MAIN GRID ────────────────────────────────────
        html.Div([

            # Row 1: Scatter + Corr matrix
            html.Div([
                html.Div(
                    dcc.Graph(id="scatter-plot", config={"displayModeBar": False},
                              style={"height": "440px"}),
                    style={"flex": "1", "minWidth": "0",
                           "background": PANEL, "border": f"1px solid {BORDER}",
                           "borderRadius": "10px", "overflow": "hidden"},
                ),
                html.Div(
                    dcc.Graph(id="corr-heatmap", config={"displayModeBar": False},
                              style={"height": "440px"}),
                    style={"width": "340px", "flexShrink": "0",
                           "background": PANEL, "border": f"1px solid {BORDER}",
                           "borderRadius": "10px", "overflow": "hidden"},
                ),
            ], style={"display": "flex", "gap": "14px", "marginBottom": "14px"}),

            # Row 2: Histogram + Box + Stats
            html.Div([
                html.Div(
                    dcc.Graph(id="hist-plot", config={"displayModeBar": False},
                              style={"height": "280px"}),
                    style={"flex": "1", "minWidth": "0",
                           "background": PANEL, "border": f"1px solid {BORDER}",
                           "borderRadius": "10px", "overflow": "hidden"},
                ),
                html.Div(
                    dcc.Graph(id="box-plot", config={"displayModeBar": False},
                              style={"height": "280px"}),
                    style={"flex": "1", "minWidth": "0",
                           "background": PANEL, "border": f"1px solid {BORDER}",
                           "borderRadius": "10px", "overflow": "hidden"},
                ),
                card([
                    label("Summary Statistics"),
                    html.Div([
                        stat_box("n",      "stat-n"),
                        stat_box("Mean",   "stat-mean"),
                        stat_box("Std Dev","stat-std"),
                        stat_box("Median", "stat-med"),
                        stat_box("Min",    "stat-min"),
                        stat_box("Max",    "stat-max"),
                    ], style={
                        "display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "8px",
                    }),
                ], {"width": "260px", "flexShrink": "0",
                    "display": "flex", "flexDirection": "column"}),
            ], style={"display": "flex", "gap": "14px"}),

        ], style={"flex": "1", "minWidth": "0"}),

    ], style={
        "display": "flex", "gap": "14px",
        "padding": "16px 20px",
        "minHeight": "calc(100vh - 70px)",
        "boxSizing": "border-box",
        "alignItems": "flex-start",
    }),

], style={
    "fontFamily": FONT,
    "background": BG,
    "color": TEXT,
    "minHeight": "100vh",
})


# ─────────────────────────────────────────────────────────
#  CALLBACKS — populate dropdowns when dataset changes
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("x-feat",    "options"),
    Output("x-feat",    "value"),
    Output("y-feat",    "options"),
    Output("y-feat",    "value"),
    Output("hist-feat", "options"),
    Output("hist-feat", "value"),
    Output("hue-feat",  "options"),
    Output("hue-feat",  "value"),
    Output("obs-badge", "children"),
    Input("ds-picker",  "value"),
)
def update_dataset(ds_name):
    df = load_dataset(ds_name)
    _df_cache["current"] = df

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    num_opts = [{"label": c, "value": c} for c in num_cols]
    all_opts = [{"label": c, "value": c} for c in all_cols]

    x0 = num_cols[0]
    y0 = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    hue0 = "target" if "target" in all_cols else (all_cols[-1])

    badge = f"⬤  {len(df):,} obs · {len(num_cols)} numeric features"
    return (num_opts, x0, num_opts, y0,
            num_opts, x0, all_opts, hue0, badge)


# ─────────────────────────────────────────────────────────
#  SCATTER PLOT
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("scatter-plot", "figure"),
    Output("pearson-r",    "children"),
    Output("pearson-r",    "style"),
    Output("pearson-desc", "children"),
    Input("x-feat",       "value"),
    Input("y-feat",       "value"),
    Input("hue-feat",     "value"),
    Input("show-reg",     "value"),
    Input("show-marginal","value"),
    Input("ds-picker",    "value"),
)
def update_scatter(x_col, y_col, hue_col, show_reg, show_marginal, _ds):
    df = _df_cache["current"]
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return go.Figure(), "—", {}, ""

    groups = df[hue_col].unique() if hue_col and hue_col in df.columns else [None]
    use_marginal = bool(show_marginal)

    if use_marginal:
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.82, 0.18],
            row_heights=[0.18, 0.82],
            shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.01, vertical_spacing=0.01,
        )
        scatter_row, scatter_col = 2, 1
    else:
        fig = go.Figure()
        scatter_row, scatter_col = None, None

    # Per-group scatter traces
    for i, grp in enumerate(groups):
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        mask  = df[hue_col] == grp if grp is not None else pd.Series([True] * len(df))
        sub   = df[mask]

        marker = dict(
            color=color, size=7, opacity=0.72,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        )
        hover = [
            f"<b>{hue_col}:</b> {grp}<br>"
            f"<b>{x_col}:</b> %{{x:.3f}}<br>"
            f"<b>{y_col}:</b> %{{y:.3f}}"
        ] * len(sub)

        sc = go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="markers", name=str(grp),
            marker=marker,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        )

        if use_marginal:
            fig.add_trace(sc, row=scatter_row, col=scatter_col)
            # top marginal histogram (x)
            fig.add_trace(go.Histogram(
                x=sub[x_col], name=str(grp), showlegend=False,
                marker_color=color, opacity=0.6, nbinsx=25,
            ), row=1, col=1)
            # right marginal histogram (y)
            fig.add_trace(go.Histogram(
                y=sub[y_col], name=str(grp), showlegend=False,
                marker_color=color, opacity=0.6, nbinsy=25,
            ), row=2, col=2)
        else:
            fig.add_trace(sc)

    # Regression line
    if show_reg:
        xs = df[x_col].values
        ys = df[y_col].values
        slope, intercept, *_ = stats.linregress(xs, ys)
        xr = np.linspace(xs.min(), xs.max(), 200)
        yr = slope * xr + intercept
        reg = go.Scatter(
            x=xr, y=yr, mode="lines", name="Regression",
            line=dict(color=ACCENT2, width=2, dash="dash"),
            hoverinfo="skip",
        )
        if use_marginal:
            fig.add_trace(reg, row=scatter_row, col=scatter_col)
        else:
            fig.add_trace(reg)

    # Pearson r
    r = pearson(df[x_col].values, df[y_col].values)
    r_color = ACCENT if r > 0.5 else (DANGER if r < -0.5 else ACCENT2)
    r_desc  = (
        f"{'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'} "
        f"{'positive' if r >= 0 else 'negative'} correlation"
    )
    r_style = {
        "fontSize": "30px", "fontWeight": "700",
        "color": r_color, "letterSpacing": "0.02em",
        "textShadow": f"0 0 20px {r_color}88",
    }

    # Style
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=f"<b>{x_col}</b>  vs  <b>{y_col}</b>", font=dict(size=13, color=TEXT)),
        xaxis_title=x_col, yaxis_title=y_col,
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="closest",
        margin=dict(l=55, r=15, t=55, b=45),
    )
    if use_marginal:
        for r_ in [1, 2]:
            for c_ in [1, 2]:
                fig.update_xaxes(showgrid=True, gridcolor=BORDER, row=r_, col=c_)
                fig.update_yaxes(showgrid=True, gridcolor=BORDER, row=r_, col=c_)

    return fig, f"{r:.3f}", r_style, r_desc


# ─────────────────────────────────────────────────────────
#  CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("corr-heatmap", "figure"),
    Input("ds-picker",     "value"),
    Input("x-feat",        "value"),
    Input("y-feat",        "value"),
)
def update_corr(ds, x_col, y_col):
    df = _df_cache["current"]
    num_df = df.select_dtypes(include=np.number)
    # cap at 12 features to stay readable
    if len(num_df.columns) > 12:
        num_df = num_df.iloc[:, :12]
    corr = num_df.corr().round(2)
    cols = corr.columns.tolist()

    # mask upper triangle
    z = corr.values.copy().astype(float)
    for i in range(len(z)):
        for j in range(i + 1, len(z[0])):
            z[i][j] = None

    # highlight selected pair
    shapes = []
    if x_col in cols and y_col in cols and x_col != y_col:
        xi, yi = cols.index(x_col), cols.index(y_col)
        rx, ry = (xi, yi) if xi > yi else (yi, xi)
        shapes = [dict(
            type="rect",
            x0=rx - 0.5, x1=rx + 0.5,
            y0=ry - 0.5, y1=ry + 0.5,
            line=dict(color=ACCENT2, width=2),
            fillcolor="rgba(0,0,0,0)",
        )]

    text = [[f"{v:.2f}" if v is not None else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        text=text, texttemplate="%{text}",
        colorscale=[
            [0.0,  "#E63946"], [0.3, "#7A1020"],
            [0.5,  BG],
            [0.7,  "#004466"], [1.0, "#00D4FF"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(
            tickfont=dict(color=MUTED, size=10),
            outlinecolor=BORDER,
            title=dict(text="r", font=dict(color=MUTED, size=11)),
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="<b>Correlation Matrix</b>", font=dict(size=13, color=TEXT)),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9, color=MUTED), side="bottom"),
        yaxis=dict(tickfont=dict(size=9, color=MUTED), autorange="reversed"),
        shapes=shapes,
        margin=dict(l=100, r=15, t=55, b=100),
    )
    return fig


# ─────────────────────────────────────────────────────────
#  HISTOGRAM
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("hist-plot",  "figure"),
    Output("stat-n",     "children"),
    Output("stat-mean",  "children"),
    Output("stat-std",   "children"),
    Output("stat-med",   "children"),
    Output("stat-min",   "children"),
    Output("stat-max",   "children"),
    Input("hist-feat",   "value"),
    Input("bins-slider", "value"),
    Input("show-kde",    "value"),
    Input("hue-feat",    "value"),
    Input("ds-picker",   "value"),
)
def update_hist(col, bins, show_kde, hue_col, _ds):
    df = _df_cache["current"]
    fallback = ("—",) * 6
    if not col or col not in df.columns:
        return go.Figure(), "—", *fallback

    s = describe_series(df[col])
    fig = go.Figure()

    groups = df[hue_col].unique() if hue_col and hue_col in df.columns else [None]
    for i, grp in enumerate(groups):
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        sub   = df[df[hue_col] == grp][col] if grp is not None else df[col]
        name  = str(grp) if grp is not None else col

        fig.add_trace(go.Histogram(
            x=sub, name=name, nbinsx=bins,
            marker_color=color, opacity=0.65,
            hovertemplate=f"<b>{name}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

        if show_kde:
            kde = stats.gaussian_kde(sub.dropna())
            xr  = np.linspace(sub.min(), sub.max(), 300)
            # scale KDE to histogram counts
            bin_width = (sub.max() - sub.min()) / bins
            yr = kde(xr) * len(sub) * bin_width
            fig.add_trace(go.Scatter(
                x=xr, y=yr, mode="lines", name=f"{name} KDE",
                line=dict(color=color, width=2.5),
                hoverinfo="skip",
            ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=f"<b>Distribution</b> — {col}", font=dict(size=13, color=TEXT)),
        xaxis_title=col, yaxis_title="Count",
        barmode="overlay",
        margin=dict(l=50, r=15, t=50, b=45),
    )

    return (fig,
            str(s["n"]), s["mean"], s["std"],
            s["median"], s["min"], s["max"])


# ─────────────────────────────────────────────────────────
#  BOX / VIOLIN PLOT
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("box-plot",  "figure"),
    Input("hist-feat",  "value"),
    Input("hue-feat",   "value"),
    Input("ds-picker",  "value"),
)
def update_box(col, hue_col, _ds):
    df = _df_cache["current"]
    if not col or col not in df.columns:
        return go.Figure()

    fig = go.Figure()
    groups = df[hue_col].unique() if hue_col and hue_col in df.columns else [None]

    for i, grp in enumerate(groups):
        color = SCATTER_COLORS[i % len(SCATTER_COLORS)]
        sub   = df[df[hue_col] == grp][col] if grp is not None else df[col]
        name  = str(grp) if grp is not None else col

        fig.add_trace(go.Violin(
            y=sub, name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.65,
            line_color=color,
            marker=dict(color=color, size=3, opacity=0.5),
            points="outliers",
            hoverinfo="y+name",
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=f"<b>Distribution by Group</b> — {col}", font=dict(size=13, color=TEXT)),
        yaxis_title=col,
        violingap=0.15, violinmode="group",
        margin=dict(l=50, r=15, t=50, b=45),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("  🔬  Feature Lab  —  Dataset Relationship Explorer")
    print("=" * 54)
    print("  Open in browser:  http://127.0.0.1:8050")
    print("  Stop server:      Ctrl + C")
    print("=" * 54 + "\n")
    app.run(debug=False, port=8050)
