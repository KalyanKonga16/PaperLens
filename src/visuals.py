import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_weighted_word_cloud(keyword_metrics: list[dict]):
    if not keyword_metrics:
        return None

    frequencies = {
        item["keyword"]: max(float(item.get("evidence_score", 1)), 1)
        for item in keyword_metrics
    }

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap="viridis",
        collocations=False,
        prefer_horizontal=1.0
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def create_category_donut(keyword_metrics: list[dict]):
    if not keyword_metrics:
        return None

    df = pd.DataFrame(keyword_metrics)
    category_df = (
        df.groupby("category", as_index=False)
        .agg(
            keyword_count=("keyword", "count"),
            total_evidence=("evidence_score", "sum")
        )
        .sort_values("total_evidence", ascending=False)
    )

    fig = px.pie(
        category_df,
        names="category",
        values="total_evidence",
        hole=0.45,
        title="Keyword Category Distribution (by Evidence)",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def create_radial_evidence_chart(keyword_metrics: list[dict]):
    if not keyword_metrics:
        return None

    df = pd.DataFrame(keyword_metrics)
    df = df.sort_values("evidence_score", ascending=True)

    fig = px.bar_polar(
        df,
        r="evidence_score",
        theta="keyword",
        color="evidence_score",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Keyword Evidence Score"
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    return fig


def create_top_keywords_bar(keyword_metrics: list[dict], top_n: int = 10):
    """
    Creates a horizontal bar chart of top N keywords by evidence score.
    Much more actionable than histogram.
    """
    if not keyword_metrics:
        return None

    df = pd.DataFrame(keyword_metrics)
    df = df.nlargest(top_n, "evidence_score").sort_values("evidence_score")

    fig = px.bar(
        df,
        x="evidence_score",
        y="keyword",
        orientation="h",
        title=f"Top {top_n} Keywords by Evidence Score",
        color="evidence_score",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={
            "evidence_score": "Evidence Score (0-100)",
            "keyword": "Keyword"
        }
    )

    fig.update_layout(
        xaxis_title="Evidence Score",
        yaxis_title="Keyword",
        showlegend=False
    )

    return fig


def create_keyword_metrics_table(keyword_metrics: list[dict]):
    """
    Creates a clean table showing only essential metrics.
    Removes Page Coverage, Context Support, Pages Found On.
    """
    if not keyword_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(keyword_metrics)
    
    # Select only essential columns
    columns = ["keyword", "category", "evidence_score", "occurrences"]
    df = df[columns]

    df = df.rename(columns={
        "keyword": "Keyword",
        "category": "Category",
        "evidence_score": "Evidence Score",
        "occurrences": "Occurrences"
    })

    df.index = range(1, len(df) + 1)
    return df
