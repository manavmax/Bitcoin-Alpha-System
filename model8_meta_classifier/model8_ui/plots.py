import plotly.graph_objects as go

def da_coverage_plot(points):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[p["coverage"] for p in points],
        y=[p["directional_accuracy"] for p in points],
        mode="markers+lines",
        marker=dict(size=10),
        name="DA vs Coverage"
    ))

    fig.update_layout(
        title="Directional Accuracy vs Coverage",
        xaxis_title="Coverage",
        yaxis_title="Directional Accuracy",
        template="plotly_dark"
    )

    return fig
