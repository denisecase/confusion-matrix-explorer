# app_raise_the_bar.py
"""Explore how changing the decision threshold (the vertical T line) affects the confusion matrix and related metrics (sensitivity, specificity, etc.) for a binary classification problem."""

import numpy as np
import pandas as pd
import plotly.express as px  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from plotly.graph_objects import Figure  # pyright: ignore[reportMissingTypeStubs]
from shiny import reactive, render
from shiny.express import input as shiny_input
from shiny.express import ui
from shinywidgets import render_plotly

from confusion_matrix_explorer.utils_confusion import (
    ConfusionCounts,
    ConfusionMetrics,
    compute_confusion_counts,
    compute_metrics,
    format_counts_table,
    generate_synthetic_scores,
)

# -----------------------------
# Global data (synthetic scores)
# -----------------------------

# Teaching, not research, just generate once at startup.
BASE_DF: pd.DataFrame = generate_synthetic_scores(
    n_pos=200,
    n_neg=200,
    pos_mean=1.5,
    neg_mean=0.0,
    pos_std=1.0,
    neg_std=1.0,
    random_state=42,
)


# -----------------------------
# UI
# -----------------------------

ui.page_opts(title="Confusion Matrix Explorer")

ui.markdown(
    """
    - **Left** of the T Line is "Test negative" | **Right** of the T Line is "Test positive".
    - **Raising the bar** (moving threshold right):
      - makes it harder to call someone positive: sensitivity goes down, **specificity goes up**.
    - **Lowering the bar** (moving threshold left):
      - makes it easier to call someone positive: **sensitivity goes up**, specificity goes down.

    """
)

with ui.sidebar():
    ui.h4("Threshold and Distribution Settings")

    ui.input_slider(
        "threshold",
        "Decision threshold (raise the bar)",
        min=float(BASE_DF["score"].min()),
        max=float(BASE_DF["score"].max()),
        value=float(BASE_DF["score"].mean()),
        step=0.1,
    )

    ui.input_checkbox(
        "show_density",
        "Normalize histograms (density)",
        value=True,
    )

    ui.input_slider(
        "bins",
        "Number of bins",
        min=10,
        max=60,
        value=30,
        step=5,
    )

    ui.markdown(
        """
        - Move the decision threshold to **raise** or **lower** the bar.
        - Watch how TP, FP, FN, TN and metrics change.
        """
    )


@reactive.calc
def current_threshold() -> float:
    """Get the current decision threshold value from the slider input.

    Returns:
    float
        The current threshold value selected by the user.
    """
    return float(shiny_input.threshold())


with ui.layout_columns(col_widths=(6, 6)):
    with ui.card():
        ui.h4("Score Distributions and Threshold")

        @render_plotly
        def score_histogram() -> Figure:
            """Overlaid histograms of scores for Disease Present vs Absent.

            Creates overlaid histograms with a vertical line at the current threshold.
            """
            threshold = current_threshold()
            density = bool(shiny_input.show_density())
            bins = int(shiny_input.bins())

            # Copy to avoid accidental mutation
            df = BASE_DF.copy()

            # Map label to a friendly name
            mapping: dict[int, str] = {1: "Disease Present", 0: "Disease Absent"}
            df["group"] = df["label"].replace(mapping)

            fig = px.histogram(
                df,
                x="score",
                color="group",
                nbins=bins,
                histnorm="probability" if density else None,
                barmode="group",  # group for side-by-side bars, overlay for overlapping
                opacity=0.6,
                labels={"score": "Test score", "group": "Group"},
            )

            # Add a vertical line for the threshold
            fig.add_vline(
                x=threshold,
                line_width=2,
                line_dash="dash",
            )

            fig.update_layout(
                title={
                    "text": "Score distributions with decision threshold",
                    "y": 0.99,  # move title up (0–1 scale)
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                margin={"t": 80, "b": 40, "l": 40, "r": 40},  # add top/bottom/side margins
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1.0,
                },
            )

            return fig

    with ui.card():
        ui.h4("Confusion Matrix and Metrics")

        @reactive.calc
        def current_counts_and_metrics() -> tuple[ConfusionCounts, ConfusionMetrics]:
            """Calculate confusion matrix counts and metrics for the current threshold.

            Returns:
            tuple[ConfusionCounts, ConfusionMetrics]
                A tuple containing the confusion matrix counts and computed metrics.
            """
            threshold = current_threshold()
            scores = BASE_DF["score"].to_numpy()
            labels = BASE_DF["label"].to_numpy()

            counts = compute_confusion_counts(scores, labels, threshold)
            metrics = compute_metrics(counts)
            return counts, metrics

        @render.table
        def confusion_table():
            """Generate a formatted confusion matrix table for display.

            Returns:
                Formatted confusion matrix table showing TP, FP, FN, TN counts.
            """
            counts, _ = current_counts_and_metrics()
            return format_counts_table(counts)

        @render.ui
        def metrics_table() -> ui.HTML:
            """Generate an HTML table for metrics display."""
            _, metrics = current_counts_and_metrics()

            def pct(x: float) -> str:
                if np.isnan(x):
                    return "N/A"
                return f"{x * 100:.1f}%"

            return ui.HTML(f"""
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; font-weight: bold;">Sensitivity (TPR)</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.sensitivity)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; font-weight: bold;">Specificity (TNR)</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.specificity)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; font-weight: bold;">Precision (PPV)</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.precision)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; font-weight: bold;">Negative PV (NPV)</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.npv)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px; font-weight: bold;">Accuracy</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.accuracy)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Prevalence</td>
                        <td style="padding: 8px; text-align: right;">{pct(metrics.prevalence)}</td>
                    </tr>
                </table>
            """)


with ui.card():
    ui.h4("Confusion Heatmap")

    @render_plotly
    def confusion_heatmap() -> Figure:
        """Generate an interactive confusion matrix heatmap."""
        counts, _ = current_counts_and_metrics()

        # Create matrix for heatmap
        matrix = np.array([[counts.tn, counts.fp], [counts.fn, counts.tp]])

        # Annotations for each cell
        annotations = []
        labels = [["TN", "FP"], ["FN", "TP"]]

        for i in range(2):
            for j in range(2):
                annotations.append(
                    {
                        "text": f"<b>{labels[i][j]}</b><br>{matrix[i][j]}",
                        "x": j,
                        "y": i,
                        "xref": "x",
                        "yref": "y",
                        "showarrow": False,
                        "font": {
                            "size": 16,
                            "color": "white" if matrix[i][j] > matrix.max() / 2 else "black",
                        },
                    }
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale="Greens",  # Or 'Blues' or 'Greens' for correct, 'Reds' for errors
                showscale=True,
                hovertemplate="%{z} cases<extra></extra>",
            )
        )

        fig.update_layout(
            annotations=annotations,
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},
            height=400,
            margin={"l": 50, "r": 10, "t": 10, "b": 60},
        )

        return fig


with ui.card():
    ui.h4("Confusion Bubble Matrix")

    @render_plotly
    def confusion_bubble_matrix() -> Figure:
        """Generate a bubble matrix where size represents count."""
        counts, _ = current_counts_and_metrics()

        # Prepare data
        data = [
            {"x": "Pred Neg", "y": "Act Neg", "count": counts.tn, "label": "TN", "color": "green"},
            {"x": "Pred Pos", "y": "Act Neg", "count": counts.fp, "label": "FP", "color": "red"},
            {"x": "Pred Neg", "y": "Act Pos", "count": counts.fn, "label": "FN", "color": "orange"},
            {
                "x": "Pred Pos",
                "y": "Act Pos",
                "count": counts.tp,
                "label": "TP",
                "color": "darkgreen",
            },
        ]

        df_matrix = pd.DataFrame(data)

        # Scale circles - max size proportional to sqrt of count
        max_count = df_matrix["count"].max()
        df_matrix["size"] = np.sqrt(df_matrix["count"] / max_count) * 100

        fig = px.scatter(
            df_matrix,
            x="x",
            y="y",
            size="size",
            color="color",
            text="label",
            hover_data={"count": True, "size": False, "color": False},
            color_discrete_map={
                "green": "#2ca02c",
                "darkgreen": "#006400",
                "red": "#d62728",
                "orange": "#ff7f0e",
            },
        )

        # Add count labels
        fig.update_traces(
            texttemplate="<b>%{text}</b><br>%{customdata[0]}",
            textposition="middle center",
            marker={"line": {"width": 2, "color": "white"}},
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Prediction",
            yaxis_title="Actual",
            yaxis={"autorange": "reversed"},
            height=350,
        )

        return fig


with ui.card():
    ui.h4("Enhanced Confusion Matrix")

    @render_plotly
    def enhanced_confusion_matrix() -> Figure:
        """Multi-encoding confusion matrix with size, opacity, and color."""
        counts, _ = current_counts_and_metrics()

        # Create the data structure
        matrix_data = [
            {
                "row": "Actual: No Disease",
                "col": "Predicted: No Disease",
                "value": counts.tn,
                "label": "TN",
                "correct": True,
            },
            {
                "row": "Actual: No Disease",
                "col": "Predicted: Disease",
                "value": counts.fp,
                "label": "FP",
                "correct": False,
            },
            {
                "row": "Actual: Disease",
                "col": "Predicted: No Disease",
                "value": counts.fn,
                "label": "FN",
                "correct": False,
            },
            {
                "row": "Actual: Disease",
                "col": "Predicted: Disease",
                "value": counts.tp,
                "label": "TP",
                "correct": True,
            },
        ]

        df = pd.DataFrame(matrix_data)

        # Calculate proportions for visual encoding
        total = df["value"].sum()
        df["proportion"] = df["value"] / total
        df["opacity"] = 0.3 + (df["value"] / df["value"].max()) * 0.7

        # Create grid positions
        positions = {
            ("Actual: No Disease", "Predicted: No Disease"): (0, 0),
            ("Actual: No Disease", "Predicted: Disease"): (1, 0),
            ("Actual: Disease", "Predicted: No Disease"): (0, 1),
            ("Actual: Disease", "Predicted: Disease"): (1, 1),
        }

        fig = go.Figure()

        for _, row in df.iterrows():
            row_label = str(row["row"])
            col_label = str(row["col"])
            x, y = positions[(row_label, col_label)]

            # Size based on count
            marker_size = 20 + (row["proportion"] * 180)

            # Color based on correct/incorrect
            is_correct = bool(row["correct"])
            if is_correct:
                color = f"rgba(46, 160, 44, {row['opacity']:.2f})"  # greenish
            else:
                color = f"rgba(214, 39, 40, {row['opacity']:.2f})"  # reddish
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker={
                        "size": marker_size,
                        "color": color,
                        "line": {"width": 2, "color": "white"},
                    },
                    text=f"<b>{row['label']}</b><br>{row['value']}<br>({row['proportion'] * 100:.1f}%)",
                    textposition="middle center",
                    textfont={
                        "size": 12 + row["proportion"] * 20,
                        "color": "white" if row["opacity"] > 0.6 else "black",
                    },
                    hovertemplate=f"{row['label']}: {row['value']} cases<br>"
                    + f"Proportion: {row['proportion'] * 100:.1f}%<extra></extra>",
                    showlegend=False,
                )
            )

        fig.update_layout(
            xaxis={
                "tickvals": [0, 1],
                "ticktext": ["Predicted: No Disease", "Predicted: Disease"],
                "range": [-0.5, 1.5],
                "zeroline": False,
            },
            yaxis={
                "tickvals": [0, 1],
                "ticktext": ["Actual: No Disease", "Actual: Disease"],
                "range": [-0.5, 1.5],
                "autorange": "reversed",
                "zeroline": False,
            },
            height=400,
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            margin={"l": 150, "r": 50, "t": 50, "b": 100},
        )

        # Add grid lines
        fig.add_shape(
            type="line", x0=-0.5, x1=1.5, y0=0.5, y1=0.5, line={"color": "gray", "width": 1}
        )
        fig.add_shape(
            type="line", x0=0.5, x1=0.5, y0=-0.5, y1=1.5, line={"color": "gray", "width": 1}
        )

        return fig


with ui.card():
    ui.h4("Confusion Waffle Chart")

    @render_plotly
    def confusion_waffle() -> Figure:
        """Create a waffle chart representation of the confusion matrix."""
        counts, _ = current_counts_and_metrics()

        # Create unit squares for each count
        total = counts.tp + counts.tn + counts.fp + counts.fn

        # Create a 20x20 grid (400 squares total)
        grid_size = 20
        scale_factor = (grid_size * grid_size) / total

        # Scale counts to grid
        scaled_tp = int(counts.tp * scale_factor)
        scaled_tn = int(counts.tn * scale_factor)
        scaled_fp = int(counts.fp * scale_factor)
        scaled_fn = int(counts.fn * scale_factor)

        # Adjust for rounding
        diff = (grid_size * grid_size) - (scaled_tp + scaled_tn + scaled_fp + scaled_fn)
        scaled_tp += diff

        # Create grid data
        categories = (
            ["TP"] * scaled_tp + ["TN"] * scaled_tn + ["FP"] * scaled_fp + ["FN"] * scaled_fn
        )

        # Shuffle for better visualization (optional)
        # np.random.shuffle(categories)

        # Create coordinates
        data = []
        for i, cat in enumerate(categories):
            x = i % grid_size
            y = i // grid_size
            data.append({"x": x, "y": y, "category": cat})

        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="category",
            color_discrete_map={"TP": "#2ca02c", "TN": "#1f77b4", "FP": "#d62728", "FN": "#ff7f0e"},
            hover_data={"category": True},
        )

        fig.update_traces(marker={"size": 15, "symbol": "square"})

        fig.update_layout(
            showlegend=True,
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=400,
            plot_bgcolor="white",
        )

        return fig
