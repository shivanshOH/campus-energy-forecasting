"""Report generation that refuses to label demo results as research results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from campus_forecast.exceptions import DataUnavailableError


def generate_html_report(record: dict[str, Any], output_path: str | Path) -> Path:
    if record.get("data_is_demo", False):
        raise DataUnavailableError(
            "Synthetic DEMO DATA cannot be included in a research-performance report"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    title = html.escape(str(record.get("model_name", "Campus energy experiment")))
    metrics = record.get("metrics", {})
    metric_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#172033}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d6dbe5;padding:.5rem;text-align:left}}
.note{{background:#f2f6fb;padding:1rem;border-left:4px solid #2b6cb0}}</style></head>
<body><h1>{title}</h1>
<p class="note">This report contains persisted experiment metadata and measured model outputs. Prediction intervals are empirical calibrated intervals, not guarantees. What-if scenarios are model-based and non-causal.</p>
<h2>Experiment configuration</h2><pre>{html.escape(str(record))}</pre>
<h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>Limitations</h2><ul><li>Results are valid only for the supplied dataset coverage and split.</li>
<li>Future observed occupancy and weather are excluded from operational configurations.</li>
<li>Exact DECODE reproduction is not claimed where the paper leaves implementation details unspecified.</li></ul>
</body></html>"""
    destination.write_text(body, encoding="utf-8")
    return destination
