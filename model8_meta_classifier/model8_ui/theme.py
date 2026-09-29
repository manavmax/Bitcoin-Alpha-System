"""Presentation layer for the Model 8 dashboard.

Design tokens and styling helpers only -- no data access, app state or business
logic. Everything visual (colour, spacing, radius, typography, chart chrome) is
defined once here so call sites never hard-code one-off values.

Tokens are mirrored in `.streamlit/config.toml` (the declarative half of the
theme). Change a value here and change the same value there.
"""

from __future__ import annotations

# =====================================================
# COLOUR
# =====================================================
# One restrained accent (amber), a cool-tinted graphite neutral ramp (never
# pure grey) and four semantic hues. Contrast ratios against the surfaces they
# sit on are all >= 4.5:1 for body text and >= 3:1 for UI / large text.
COLORS = {
    # surfaces -- near-black terminal, panels split by hairlines, no shadows
    "canvas": "#000000",
    "surface": "#0D0D0F",
    "surface_raised": "#141416",
    "surface_hover": "#161618",
    # strokes -- 1px rules only
    "border": "#2A2A2E",
    "border_strong": "#45454C",
    # type -- white / gray / dim, on black
    "text": "#F2F2F2",         # ~17:1 on canvas
    "text_muted": "#A6A6A6",   #  ~8:1 on surface
    "text_subtle": "#7C7C7C",  # ~4.6:1 on surface
    # accent -- terminal amber. Active state, focus, strategy line only
    "accent": "#FF9E2C",
    "accent_text": "#FFB35C",
    "accent_soft": "rgba(255, 158, 44, 0.10)",
    # semantics
    "success": "#00D97A",
    "warning": "#FFB300",
    "danger": "#FF453A",
    "info": "#4D9DE0",
    "violet": "#B79CFF",
    # directional -- terminal green / red, always paired with ▲▼ markers
    "long": "#00D97A",
    "short": "#FF4D4D",
    "benchmark": "#8E8E93",
    # chart chrome
    "grid": "rgba(255, 255, 255, 0.08)",
    "axis": "#333338",
}

# =====================================================
# TYPOGRAPHY
# =====================================================
# Terminal rule: sans for chrome, mono for data. System stacks only --
# no webfont, no new dependency.
FONT_STACK = (
    '"Helvetica Neue", Helvetica, Arial, "Segoe UI", sans-serif'
)
MONO_STACK = (
    '"SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, '
    '"Liberation Mono", monospace'
)
# When True, KPI/metric values render in the mono stack (Bloomberg numbers).
USE_MONO_NUMERALS = True

# =====================================================
# SPACING / RADIUS / MOTION
# =====================================================
# Radius 0 everywhere. A terminal has corners, not pillows. Single transition
# speed only for state changes; nothing decorative animates.
SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "20px", "6": "24px"}
RADIUS = {"control": "0px", "card": "0px", "pill": "0px"}
DURATION = "100ms"
EASING = "linear"

# =====================================================
# CHART PALETTES
# =====================================================
CATEGORICAL = [
    COLORS["info"],
    COLORS["long"],
    COLORS["accent"],
    COLORS["violet"],
    COLORS["short"],
]
# Volatility regime -- dim to bright white ramp on black.
SEQUENTIAL = ["#1A1A1C", "#2E2E33", "#4A4A52", "#6E6E78", "#9A9AA3", "#CFCFD4"]
# Macro regime -- short red -> neutral gray -> long green.
DIVERGING = ["#FF4D4D", "#A35A5A", "#8E8E93", "#4E9E7E", "#00D97A"]


# =====================================================
# LAYOUT TOKENS
# =====================================================
# NOTE: _root_vars() emits these three directly (page-max/gutter/panel-pad).
# PAGE_MAX / GUTTER / PANEL_PAD are the single source of truth -- the config
# comment in .streamlit/config.toml mirrors them for documentation only.
PAGE_MAX, GUTTER, PANEL_PAD = "1600px", "12px", "8px"

# Chart heights -- dense terminal grid, compact panels.
H_HERO, H_MAIN, H_SUB, H_MINI = 460, 320, 260, 200


def _root_vars() -> str:
    """Emit the token layer as CSS custom properties."""
    lines = [":root {"]
    for key, value in COLORS.items():
        lines.append(f"  --c-{key.replace('_', '-')}: {value};")
    for key, value in SPACE.items():
        lines.append(f"  --sp-{key}: {value};")
    for key, value in RADIUS.items():
        lines.append(f"  --r-{key}: {value};")
    lines.append("  --page-max: 1600px;")
    lines.append("  --gutter: 12px;")
    lines.append("  --panel-pad: 8px;")
    lines.append(f"  --font-sans: {FONT_STACK};")
    lines.append(f"  --font-mono: {MONO_STACK};")
    lines.append(f"  --duration: {DURATION};")
    lines.append(f"  --ease: {EASING};")
    lines.append("}")
    return "\n".join(lines)


_STYLESHEET = """
/* ================================================================ frame */
/* Terminal grid: full-bleed black, 12px gutters, zero decoration. */
.block-container {
  max-width: var(--page-max);
  padding-top: var(--sp-3);
  padding-bottom: var(--sp-6);
  padding-left: var(--gutter);
  padding-right: var(--gutter);
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stVerticalBlock"] { gap: var(--gutter); }
[data-testid="stHorizontalBlock"] { gap: var(--gutter); }

/* ============================================================ top strip */
/* Thin command bar under the header: amber ticker-style brand. */
.cq-topstrip {
  display: flex;
  align-items: baseline;
  gap: var(--sp-3);
  padding: var(--sp-2) 0;
  border-bottom: 1px solid var(--c-border-strong);
}
.cq-topstrip .cq-brand {
  font-family: var(--font-sans);
  font-size: 0.8125rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--c-accent);
}
.cq-topstrip .cq-session {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--c-text-subtle);
}

/* ================================================================ panels */
/* Every block is a flat panel: black fill, 1px hairline, square corners.
   Amber top rule marks the lead panel. No shadows, no hover glow. */
.cq-card {
  height: 100%;
  padding: var(--panel-pad);
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: 0;
  box-shadow: none;
}
.cq-card--decision { border-top: 3px solid var(--c-accent); }
.cq-card--long { border-top: 3px solid var(--c-long); }
.cq-card--short { border-top: 3px solid var(--c-short); }

/* Field label: 10px mono, dim, uppercase, tracked. Value: mono numerals. */
.cq-kpi-title {
  margin-bottom: 2px;
  font-family: var(--font-mono);
  font-size: 0.625rem;
  font-weight: 400;
  line-height: 1rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--c-text-subtle);
}
.cq-kpi-value {
  font-family: var(--font-mono);
  font-size: 1.375rem;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: 0;
  color: var(--c-text);
  font-variant-numeric: tabular-nums;
}
.cq-card--decision .cq-kpi-value { font-size: 1.625rem; }
.cq-kpi-value--long { color: var(--c-long); }
.cq-kpi-value--short { color: var(--c-short); }
.cq-kpi-sub {
  margin-top: 2px;
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  line-height: 1rem;
  color: var(--c-text-subtle);
  font-variant-numeric: tabular-nums;
}

/* ================================================================ fields */
/* st.metric renders as a quote field: label above, mono number below. */
[data-testid="stMetric"] {
  padding: var(--panel-pad);
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: 0;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--font-mono);
  font-size: 0.625rem;
  font-weight: 400;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--c-text-subtle);
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono);
  font-weight: 700;
  letter-spacing: 0;
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetricDelta"] p {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

/* ================================================================== tabs */
/* Function-key row: uppercase mono labels, boxed active tab in amber. */
[data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--c-border-strong);
}
[data-baseweb="tab"] {
  padding: var(--sp-2) var(--sp-3);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 400;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--c-text-muted);
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 0;
}
[data-baseweb="tab"]:hover { color: var(--c-text); }
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--c-accent);
  font-weight: 700;
  background: var(--c-surface);
  border-color: var(--c-border-strong);
}
[data-baseweb="tab-highlight"] { display: none; }
[data-baseweb="tab-border"] { background-color: transparent; }
[data-testid="stTabContent"] { padding-top: var(--gutter); }

/* ============================================================ data grids */
/* Grids: hairline rows, mono numerals, dim uppercase header row. */
[data-testid="stDataFrame"] {
  border: 1px solid var(--c-border);
  border-radius: 0;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] [role="columnheader"] {
  font-family: var(--font-mono);
  font-size: 0.625rem;
  font-weight: 400;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--c-text-subtle);
}

/* ================================================================ charts */
/* Figures sit directly on black panels with a 1px frame. No padding. */
[data-testid="stPlotlyChart"] {
  padding: 0;
  background: #000000;
  border: 1px solid var(--c-border);
  border-radius: 0;
  overflow: hidden;
}

/* ================================================================== type */
/* Section heads: small amber rules, not big marketing headings. */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans);
  letter-spacing: 0;
}
h2 {
  font-size: 1rem !important;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--c-text);
  border-bottom: 1px solid var(--c-border-strong);
  padding-bottom: var(--sp-2);
  margin-bottom: var(--sp-3);
}
h3 {
  font-size: 0.8125rem !important;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--c-accent);
  margin-bottom: var(--sp-2);
}
h4 {
  font-size: 0.75rem !important;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--c-text-muted);
}
[data-testid="stCaptionContainer"] p {
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  line-height: 1rem;
  color: var(--c-text-subtle);
}
[data-testid="stWidgetLabel"] p {
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  font-weight: 400;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--c-text-muted);
}

/* ================================================================ sidebar */
/* Left rail = function panel: black, hard right rule, amber section tags. */
[data-testid="stSidebar"] {
  background: #000000;
  border-right: 1px solid var(--c-border-strong);
}
[data-testid="stSidebar"] h2 {
  margin-bottom: var(--sp-2);
  font-family: var(--font-mono);
  font-size: 0.6875rem !important;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--c-accent);
  border-bottom: 1px solid var(--c-border);
  padding-bottom: var(--sp-2);
}
[data-testid="stSidebar"] hr {
  margin: var(--sp-3) 0;
  border: none;
  border-top: 1px solid var(--c-border);
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-size: 0.6875rem;
  line-height: 1rem;
  color: var(--c-text-subtle);
}

/* =============================================================== controls */
/* Square inputs with hairline borders. Primary = solid amber block. */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  min-height: 32px;
  padding: 0 var(--sp-3);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-radius: 0;
  background: var(--c-surface);
  border: 1px solid var(--c-border-strong);
  color: var(--c-text);
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--c-accent);
  color: var(--c-accent);
}
[data-testid="stButton"] button[kind="primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {
  background: var(--c-accent);
  border-color: var(--c-accent);
  color: #000000;
}
input, select, textarea {
  border-radius: 0 !important;
  font-family: var(--font-mono) !important;
}
[data-baseweb="select"] > div {
  border-radius: 0 !important;
  background: var(--c-surface) !important;
  border-color: var(--c-border-strong) !important;
  font-family: var(--font-mono) !important;
}
[data-testid="stAlert"] { border-radius: 0; }

/* =========================================================== focus rings */
/* Terminal focus = 1px amber outline, square. */
:where(a, button, [role="tab"], [role="slider"], input, select, textarea,
       [tabindex]):focus-visible {
  outline: 1px solid var(--c-accent);
  outline-offset: 1px;
}

/* ============================================================ scrollbars */
* {
  scrollbar-width: thin;
  scrollbar-color: var(--c-border-strong) #000000;
}

/* ================================================================ motion */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}
* { transition-duration: var(--duration); }

/* ============================================================ responsive */
@media (max-width: 768px) {
  .block-container {
    padding-left: var(--sp-2);
    padding-right: var(--sp-2);
  }
  .cq-card { padding: var(--sp-2); }
  .cq-kpi-value { font-size: 1.125rem; }
  .cq-card--decision .cq-kpi-value { font-size: 1.375rem; }
}
@media (max-width: 480px) {
  /* touch targets >= 44px */
  [data-baseweb="tab"] { min-height: 44px; }
  [data-testid="stButton"] button,
  [data-testid="stDownloadButton"] button { min-height: 44px; }
}
"""

CSS = "<style>\n" + _root_vars() + "\n" + _STYLESHEET + "\n</style>"

# =====================================================
# CHART CHROME
# =====================================================
# Terminal charts: black panels, hairline axes, white tick labels, mono font.
# No rounded anything, no transparency, no glow.
_BASE_XAXIS = {
    "showgrid": False,
    "zeroline": False,
    "linecolor": COLORS["axis"],
    "ticks": "outside",
    "tickcolor": COLORS["axis"],
    "ticklen": 3,
    "tickfont": {"family": MONO_STACK, "size": 10, "color": COLORS["text_muted"]},
}
_BASE_YAXIS = {
    "showgrid": True,
    "gridcolor": COLORS["grid"],
    "gridwidth": 1,
    "zeroline": False,
    "linecolor": COLORS["axis"],
    "ticks": "",
    "tickfont": {"family": MONO_STACK, "size": 10, "color": COLORS["text_muted"]},
}
_BASE_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#000000",
    "plot_bgcolor": "#000000",
    "font": {"family": MONO_STACK, "size": 11, "color": COLORS["text_muted"]},
    "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(0,0,0,0)",
        "font": {"family": MONO_STACK, "size": 10, "color": COLORS["text_muted"]},
    },
    "hoverlabel": {
        "bgcolor": "#000000",
        "bordercolor": COLORS["border_strong"],
        "font": {"family": MONO_STACK, "size": 11, "color": COLORS["text"]},
    },
}


def chart_layout(height: int = H_SUB, **overrides) -> dict:
    """Build an `update_layout(**...)` dict from the shared chart tokens.

    Nested dicts are deep-copied one level so callers can tweak an axis
    without restating the whole chrome.
    """
    import copy

    layout = copy.deepcopy(_BASE_LAYOUT)
    layout["height"] = height
    layout["xaxis"] = copy.deepcopy(_BASE_XAXIS)
    layout["yaxis"] = copy.deepcopy(_BASE_YAXIS)

    for key in ("xaxis", "yaxis", "legend", "hoverlabel", "margin"):
        if key in overrides:
            layout[key].update(overrides.pop(key))

    layout.update(overrides)
    return layout


# =====================================================
# CANDLE STYLE -- one shared spec, Bloomberg line weights
# =====================================================
# Hollow candles read on black without any glow: filled body for down days,
# thin wicks, flat line width. Markers keep ▲▼ shape + colour.
CANDLE = {
    "increasing_fillcolor": "#000000",
    "increasing_line_color": COLORS["long"],
    "increasing_line_width": 1,
    "decreasing_fillcolor": COLORS["short"],
    "decreasing_line_color": COLORS["short"],
    "decreasing_line_width": 1,
    "whiskerwidth": 1,
}
MARKER_LONG = {"symbol": "triangle-up", "size": 8, "color": COLORS["long"]}
MARKER_SHORT = {"symbol": "triangle-down", "size": 8, "color": COLORS["short"]}


# =====================================================
# ENTRY POINTS
# =====================================================
def inject_css(st) -> None:
    """Inject the token stylesheet into a Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)


def apply_theme(st):
    """Theme entry point (kept for backwards compatibility)."""
    st.set_page_config(
        page_title="Model 8 — Bitcoin Intelligence",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_css(st)


