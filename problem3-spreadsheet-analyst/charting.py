from __future__ import annotations

import plotly.express as px

def choose_chart(result, plan):
    requested = plan.get("chart", "auto")
    if requested != "auto":
        return requested
    if "__time__" in result.columns:
        return "line"
    return "bar"

def build_chart(result, plan, question):
    if result.empty or len(result) == 1 or "value" not in result.columns:
        return None
    chart = choose_chart(result, plan)
    dims = [c for c in result.columns if c != "value"]
    x = dims[0]

    if chart == "line":
        fig = px.line(result, x=x, y="value", markers=True, title=question)
    elif chart == "pie" and len(result) <= 8:
        fig = px.pie(result, names=x, values="value", title=question)
    else:
        fig = px.bar(result, x=x, y="value", title=question)

    fig.update_layout(
        title_x=0,
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis_title=str(x).replace("__time__", "Time"),
        yaxis_title="Value",
    )
    return fig
