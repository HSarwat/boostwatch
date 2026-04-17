"""Self-contained HTML report generator for Boostwatch training runs."""

from __future__ import annotations

import base64
import datetime
import io
from typing import Any, List, Optional

import matplotlib.pyplot as plt

from . import charts as _charts
from ._helpers import _get_framework, _resolve_feature_names

_DEFAULT_INCLUDE = [
    "header",
    "learning_curve",
    "tree_complexity",
    "feature_stats",
    "feature_heatmap",
    "leaf_distribution",
    "split_depth",
]

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f2f5;
  color: #1a1a1a;
  padding: 24px;
}
.container { max-width: 1200px; margin: 0 auto; }
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  border-radius: 10px;
  padding: 28px 36px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
header h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
.meta { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 20px; }
.meta-item { font-size: 0.82rem; color: #9db3cc; }
.meta-item b { color: #c8ddef; }
details {
  background: #fff;
  border-radius: 10px;
  margin-bottom: 16px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.07);
  overflow: hidden;
}
summary.section-title {
  padding: 16px 24px;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
  border-left: 4px solid #4a90d9;
  display: flex;
  align-items: center;
  gap: 8px;
}
summary.section-title::before { content: "\\25B6"; font-size: 0.7em; color: #4a90d9; }
details[open] summary.section-title::before { content: "\\25BC"; }
summary.section-title:hover { background: #f5f9ff; }
.section-body {
  padding: 24px;
  border-top: 1px solid #eef1f5;
}
img {
  display: block;
  margin: 0 auto;
  border-radius: 6px;
  max-width: 100%;
  height: auto;
}
.unavailable {
  padding: 16px;
  background: #fafafa;
  border-radius: 6px;
  color: #888;
  font-size: 0.85rem;
  font-style: italic;
}
"""


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a Figure to a base64-encoded PNG string and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return data


def _img_tag(b64: str) -> str:
    return '<img src="data:image/png;base64,{}" alt="chart">'.format(b64)


def _section(title: str, content: str, open: bool = False) -> str:
    open_attr = " open" if open else ""
    return (
        "\n<details{}>"
        "\n  <summary class=\"section-title\">{}</summary>"
        "\n  <div class=\"section-body\">{}</div>"
        "\n</details>".format(open_attr, title, content)
    )


def _safe_section(title: str, fn, open: bool = False) -> str:
    """Call fn() to get content HTML; return a note section on any exception."""
    try:
        content = fn()
        return _section(title, content, open=open)
    except Exception as exc:
        note = '<p class="unavailable">Not available: {}</p>'.format(exc)
        return _section(title, note)


def generate_report(
    logs: List[Any],
    feature_names: Optional[List[str]] = None,
    title: str = "Boostwatch Training Report",
    output_path: Optional[str] = None,
    include: Optional[List[str]] = None,
) -> str:
    """Generate a self-contained HTML training report.

    Each section is a collapsible ``<details>`` block. Charts are embedded as
    base64 PNG — no external dependencies, fully shareable as a single file.

    Args:
        logs: List of ``IterationLog`` objects from any framework observer.
        feature_names: Optional list of feature names. Falls back to names
            embedded in logs if available.
        title: Report title shown in the header and browser tab.
        output_path: If provided, the HTML is also written to this file path.
        include: List of section keys to render. Omit a key to hide that
            section. Valid keys:
            ``"header"``, ``"learning_curve"``, ``"tree_complexity"``,
            ``"feature_stats"``, ``"feature_heatmap"``,
            ``"leaf_distribution"``, ``"split_depth"``.
            Defaults to all sections.

    Returns:
        The complete HTML document as a string.
    """
    if include is None:
        include = list(_DEFAULT_INCLUDE)
    include_set = set(include)

    names = _resolve_feature_names(logs, feature_names)
    framework = _get_framework(logs) or "unknown"
    n_iters = len(logs)
    n_features = len(names) if names else "?"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    header_html = (
        "\n<header>"
        "\n  <h1>{title}</h1>"
        "\n  <div class=\"meta\">"
        "\n    <div class=\"meta-item\"><b>Framework</b> {fw}</div>"
        "\n    <div class=\"meta-item\"><b>Iterations</b> {iters}</div>"
        "\n    <div class=\"meta-item\"><b>Features</b> {feats}</div>"
        "\n    <div class=\"meta-item\"><b>Generated</b> {ts}</div>"
        "\n  </div>"
        "\n</header>"
    ).format(title=title, fw=framework, iters=n_iters, feats=n_features, ts=timestamp)

    sections_html = ""

    if "learning_curve" in include_set:
        def _learning_curve():
            fig = _charts.plot_learning_curve(logs)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Learning Curve", _learning_curve, open=True)

    if "tree_complexity" in include_set:
        def _tree_complexity():
            fig = _charts.plot_tree_complexity(logs)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Tree Complexity", _tree_complexity)

    if "feature_stats" in include_set and names:
        def _feature_stats():
            from ..analysis.feature_stats import compute_feature_stats
            stats = compute_feature_stats(logs, names)
            fig = _charts.plot_feature_stats(stats, names)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Feature Importance", _feature_stats)

    if "feature_heatmap" in include_set:
        def _feature_heatmap():
            fig = _charts.plot_feature_heatmap(logs)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Feature Heatmap", _feature_heatmap)

    if "leaf_distribution" in include_set:
        def _leaf_dist():
            fig = _charts.plot_leaf_distribution(logs)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Leaf Distribution", _leaf_dist)

    if "split_depth" in include_set:
        def _split_depth():
            fig = _charts.plot_split_depth_dist(logs)
            return _img_tag(_fig_to_base64(fig))
        sections_html += _safe_section("Split Depth Distribution", _split_depth)

    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>{title}</title>\n"
        "  <style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"container\">\n"
        "    {header}\n"
        "    {sections}\n"
        "  </div>\n"
        "</body>\n"
        "</html>"
    ).format(title=title, css=_CSS, header=header_html, sections=sections_html)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


def display_report(
    logs: List[Any],
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Display a self-contained HTML report inline in a Jupyter notebook.

    Args:
        logs: List of ``IterationLog`` objects from any framework observer.
        feature_names: Optional list of feature names.
        **kwargs: Forwarded to :func:`generate_report`.
    """
    from IPython.display import HTML, display
    html = generate_report(logs, feature_names=feature_names, **kwargs)
    display(HTML(html))
