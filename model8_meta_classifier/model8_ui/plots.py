import plotly.graph_objects as go

import theme


def da_coverage_plot(points):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[p["coverage"] for p in points],
        y=[p["directional_accuracy"] for p in points],
        mode="markers+lines",
        marker=dict(size=7, color=theme.COLORS["accent"]),
        line=dict(color=theme.COLORS["accent"], width=1),
        name="DA vs Coverage"
    ))

    fig.update_layout(
        **theme.chart_layout(
            height=theme.H_MAIN,
            title=dict(
                text="Directional Accuracy vs Coverage",
                font=dict(family=theme.MONO_STACK, size=12, color=theme.COLORS["text"]),
            ),
            xaxis=dict(title="Coverage"),
            yaxis=dict(title="Directional Accuracy"),
            margin=dict(l=8, r=8, t=40, b=8),
        )
    )

    return fig
