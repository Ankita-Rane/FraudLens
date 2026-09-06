"""Generate the editable, implemented-only diagrams.net architecture."""

from __future__ import annotations

import argparse
import base64
import subprocess
import urllib.parse
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "FRAUD_INVESTIGATION_FRAMEWORK_ARCHITECTURE.drawio"

NAVY = "#102A43"
BLUE = "#E7F0FA"
BLUE_STROKE = "#3273A8"
GREEN = "#E7F6EC"
GREEN_STROKE = "#398454"
PURPLE = "#F0EAF8"
PURPLE_STROKE = "#76549B"
RED = "#FCEBEA"
RED_STROKE = "#AE514B"
AMBER = "#FFF4D6"
AMBER_STROKE = "#AD791A"
GRAY = "#F4F6F8"
GRAY_STROKE = "#718096"


def node_style(fill: str, stroke: str, *, shape: str = "rounded") -> str:
    shape_style = "rounded=1;" if shape == "rounded" else f"shape={shape};"
    return (
        f"{shape_style}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        f"strokeWidth=2;fontColor={NAVY};fontFamily=Helvetica;fontSize=13;"
        "align=center;verticalAlign=middle;spacing=7;shadow=0;"
    )


STYLES = {
    "ui": node_style(BLUE, BLUE_STROKE),
    "api": node_style(BLUE, BLUE_STROKE),
    "data": node_style(GREEN, GREEN_STROKE, shape="cylinder3"),
    "pipeline": node_style(GREEN, GREEN_STROKE),
    "reasoning": node_style(PURPLE, PURPLE_STROKE),
    "control": node_style(RED, RED_STROKE),
    "evaluation": node_style(GRAY, GRAY_STROKE),
    "human": node_style(AMBER, AMBER_STROKE, shape="mxgraph.basic.person"),
}

ZONE_STYLE = (
    "swimlane;html=1;rounded=1;startSize=38;fillColor=#FFFFFF;swimlaneFillColor=#F8FAFC;"
    f"strokeColor=#A8B6C4;strokeWidth=2;fontColor={NAVY};fontFamily=Helvetica;"
    "fontStyle=1;fontSize=16;horizontal=1;collapsible=0;container=0;"
)
TITLE_STYLE = (
    f"text;html=1;strokeColor=none;fillColor=none;fontColor={NAVY};"
    "fontFamily=Helvetica;fontStyle=1;fontSize=27;align=left;verticalAlign=middle;"
)
TEXT_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;fontColor=#486581;"
    "fontFamily=Helvetica;fontSize=12;align=left;verticalAlign=middle;"
)
NOTE_STYLE = (
    f"shape=note;whiteSpace=wrap;html=1;fillColor=#FFFBEA;strokeColor={AMBER_STROKE};"
    f"fontColor={NAVY};fontFamily=Helvetica;fontSize=11;align=left;spacing=7;"
)
EDGE = (
    f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={BLUE_STROKE};"
    "strokeWidth=2;endArrow=block;endFill=1;fontFamily=Helvetica;fontSize=10;"
    "fontColor=#274C67;labelBackgroundColor=#FFFFFF;"
)
EDGE_DATA = EDGE.replace(BLUE_STROKE, GREEN_STROKE)
EDGE_POLICY = (
    f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={PURPLE_STROKE};"
    "strokeWidth=2;dashed=1;dashPattern=5 4;endArrow=block;endFill=1;"
    "fontFamily=Helvetica;fontSize=10;fontColor=#5E4278;labelBackgroundColor=#FFFFFF;"
)
EDGE_EVAL = (
    f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={GRAY_STROKE};"
    "strokeWidth=2;dashed=1;dashPattern=4 4;endArrow=open;endFill=0;"
    "fontFamily=Helvetica;fontSize=10;fontColor=#52606D;labelBackgroundColor=#FFFFFF;"
)


@dataclass(frozen=True)
class Box:
    cell_id: str
    value: str
    x: int
    y: int
    width: int
    height: int
    kind: str = "api"


class Diagram:
    def __init__(self) -> None:
        self.model = ET.Element(
            "mxGraphModel",
            {
                "dx": "1900",
                "dy": "1300",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": "1900",
                "pageHeight": "1300",
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def vertex(self, box: Box, *, style: str | None = None) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": box.cell_id,
                "value": box.value,
                "style": style or STYLES[box.kind],
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(box.x),
                "y": str(box.y),
                "width": str(box.width),
                "height": str(box.height),
                "as": "geometry",
            },
        )

    def edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        value: str = "",
        *,
        style: str = EDGE,
    ) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": edge_id,
                "value": value,
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


def rich(title: str, details: str) -> str:
    return f"<b>{title}</b><br><font style=\"font-size:11px\">{details}</font>"


def add_header(diagram: Diagram) -> None:
    diagram.vertex(
        Box("title", "Implemented Fraud Investigation Architecture", 35, 20, 1030, 42),
        style=TITLE_STYLE,
    )
    diagram.vertex(
        Box(
            "subtitle",
            "Current repository only • executable paths, stored artifacts and implemented interfaces • generated from source audit",
            35,
            60,
            1130,
            28,
        ),
        style=TEXT_STYLE,
    )
    legends = [
        ("leg_ui", "UI / API", 1215, "ui"),
        ("leg_data", "Data / evidence", 1360, "pipeline"),
        ("leg_reason", "LLM / retrieval", 1530, "reasoning"),
        ("leg_control", "Safety / evaluation", 1700, "control"),
    ]
    for cell_id, value, x, kind in legends:
        diagram.vertex(Box(cell_id, f"<b>{value}</b>", x, 42, 135, 30, kind))


def build_diagram() -> Diagram:
    d = Diagram()
    add_header(d)

    # Four current-state zones. These are visual containers, not runtime components.
    d.vertex(Box("zone_ui", "1  RESEARCH &amp; OPERATIONS CONSOLE", 35, 115, 300, 940), style=ZONE_STYLE)
    d.vertex(Box("zone_service", "2  API &amp; INVESTIGATION SERVICE", 360, 115, 705, 940), style=ZONE_STYLE)
    d.vertex(Box("zone_evidence", "3  DATA, EVIDENCE &amp; RETRIEVAL", 1090, 115, 775, 565), style=ZONE_STYLE)
    d.vertex(Box("zone_eval", "4  STORED RUNS, AUDIT &amp; EVALUATION", 1090, 705, 775, 350), style=ZONE_STYLE)

    # Research and operations console.
    d.vertex(Box("analyst", rich("Researcher / analyst", "external human actor"), 90, 220, 190, 80, "human"))
    d.vertex(Box("browser", rich("Seven-page browser console", "workflow • EDA • experiments<br>alerts • evidence • safety"), 75, 390, 220, 85, "ui"))
    d.vertex(Box("node", rich("Node web server", "web/server.mjs<br>static UI + API proxy"), 75, 575, 220, 95, "api"))
    d.vertex(
        Box(
            "ui_note",
            "The browser submits named allowlisted operations. It displays saved evidence, job state and bounded Jira analytics; it receives no shell or credentials.",
            60,
            790,
            245,
            105,
        ),
        style=NOTE_STYLE,
    )

    # API and service orchestration.
    d.vertex(Box("fastapi", rich("FastAPI boundary", "src/api.py<br>jobs • studies • EDA • alerts • Jira"), 405, 190, 220, 100, "api"))
    d.vertex(Box("service", rich("InvestigationService", "src/abc_service.py<br>strategy selection + orchestration"), 690, 190, 310, 100, "api"))
    d.vertex(Box("scope", rich("Evidence scoping", "src/model_evidence_retrieval.py<br>question + transaction evidence"), 550, 345, 300, 90, "pipeline"))
    d.vertex(Box("config_a", rich("Configuration A", "BaselineStrategy<br>direct explanation; no retrieval"), 400, 505, 190, 100, "reasoning"))
    d.vertex(Box("config_b", rich("Configuration B", "RetrievalStrategy<br>retrieved public-source context"), 620, 505, 190, 100, "reasoning"))
    d.vertex(Box("config_c", rich("Configuration C", "GuardedStrategy<br>retrieval + enforced controls"), 840, 505, 190, 100, "control"))
    d.vertex(Box("generator", rich("Text generator", "LocalQwenProvider when available<br>deterministic demo fallback otherwise"), 465, 695, 260, 100, "reasoning"))
    d.vertex(Box("controls", rich("C output controls", "src/safety.py + src/llm_backend.py<br>input checks • score gate • citations • schema"), 770, 695, 260, 105, "control"))
    d.vertex(Box("response", rich("Structured investigation response", "answer + evidence/citations + telemetry"), 570, 885, 320, 90, "api"))

    # Current datasets and generated evidence.
    d.vertex(Box("ulb_artifacts", rich("ULB model artifacts", "artifacts/ulb<br>pipeline • metrics • predictions"), 1130, 190, 205, 90, "data"))
    d.vertex(Box("sparkov_artifacts", rich("Sparkov model artifacts", "artifacts/sparkov<br>pipeline • metrics • predictions • attributions"), 1130, 325, 205, 100, "data"))
    d.vertex(Box("builder", rich("Unified evidence builder", "tools/build_unified_evidence.py<br>preserves dataset-specific features"), 1390, 245, 220, 100, "pipeline"))
    d.vertex(Box("unified", rich("Unified model evidence", "artifacts/unified/unified_model_evidence.json<br>actual label excluded from LLM context"), 1650, 245, 185, 110, "data"))
    d.vertex(Box("samples", rich("Runtime samples", "datasets/sparkov/runtime_transaction_samples.csv"), 1130, 500, 205, 80, "data"))
    d.vertex(Box("policies", rich("Frozen public-source corpus", "datasets/policy_sources<br>HTML snapshots + source registry"), 1390, 480, 220, 95, "data"))
    d.vertex(Box("retriever", rich("TF-IDF retriever", "src/retrieval.py<br>load • chunk • rank"), 1650, 480, 185, 95, "reasoning"))
    d.vertex(
        Box(
            "evidence_note",
            "ULB and Sparkov remain separate model pipelines. Their outputs meet only through the versioned evidence structure; raw feature spaces are not merged.",
            1370,
            600,
            445,
            58,
        ),
        style=NOTE_STYLE,
    )

    # Run storage, downstream operational prototype, and evaluation endpoints.
    d.vertex(Box("runs", rich("ABC run store", "artifacts/abc_runs/*.json<br>immutable run persisted first"), 1120, 740, 205, 80, "data"))
    d.vertex(Box("jira_trigger", rich("Post-C event trigger", "src/jira_trigger.py<br>alert-ID dedupe • dataset routing • audit"), 1370, 735, 220, 90, "control"))
    d.vertex(Box("jira_cloud", rich("Jira Cloud sandbox", "ULB/Sparkov Epics<br>one Story per detector evidence ID"), 1630, 735, 205, 90, "human"))
    d.vertex(Box("metrics", rich("Automatic evaluation", "src/abc_evaluation.py<br>paired metrics + consistency"), 1120, 850, 205, 85, "evaluation"))
    d.vertex(Box("annotations", rich("Human annotation records", "src/annotations.py<br>tasks • claim units • agreement"), 1370, 850, 220, 85, "evaluation"))
    d.vertex(Box("jira_monitor", rich("Bounded Jira monitor", "open ≥7 days • overdue • due soon<br>API-lifespan scheduler"), 1630, 850, 205, 85, "control"))
    d.vertex(Box("manifest", rich("Input + UI registries", "input hashes • study instances<br>SQLite job state"), 1120, 965, 205, 65, "data"))
    d.vertex(Box("audit", rich("Artifact audit", "src/artifact_audit.py<br>existence + recorded hashes"), 1370, 950, 220, 80, "evaluation"))
    d.vertex(Box("tests", rich("Automated tests", "tests/<br>including trigger, dedupe and monitor"), 1630, 950, 205, 80, "evaluation"))

    # Runtime request/response flow.
    d.edge("e_analyst_browser", "analyst", "browser", "question + transaction")
    d.edge("e_browser_node", "browser", "node", "HTTP/JSON")
    d.edge("e_node_api", "node", "fastapi", "HTTP /api/v1")
    d.edge("e_api_service", "fastapi", "service", "validated request")
    d.edge("e_service_scope", "service", "scope", "evidence lookup")
    d.edge("e_scope_a", "scope", "config_a", "scoped evidence")
    d.edge("e_scope_b", "scope", "config_b", "scoped evidence")
    d.edge("e_scope_c", "scope", "config_c", "scoped evidence")
    d.edge("e_a_generator", "config_a", "generator", "prompt")
    d.edge("e_b_generator", "config_b", "generator", "grounded prompt")
    d.edge("e_c_generator", "config_c", "generator", "controlled prompt")
    d.edge("e_generator_response", "generator", "response", "A/B output")
    d.edge("e_generator_controls", "generator", "controls", "C candidate output")
    d.edge("e_controls_response", "controls", "response", "accepted/blocked result")
    d.edge("e_response_api", "response", "fastapi", "response contract")
    d.edge("e_api_node", "fastapi", "node", "JSON response")

    # Evidence and retrieval flow.
    d.edge("e_ulb_builder", "ulb_artifacts", "builder", "dataset-tagged evidence", style=EDGE_DATA)
    d.edge("e_sparkov_builder", "sparkov_artifacts", "builder", "dataset-tagged evidence", style=EDGE_DATA)
    d.edge("e_builder_unified", "builder", "unified", "JSON evidence bundle", style=EDGE_DATA)
    d.edge("e_unified_scope", "unified", "scope", "evidence_id lookup", style=EDGE_DATA)
    d.edge("e_samples_service", "samples", "service", "sample transactions", style=EDGE_DATA)
    d.edge("e_policy_retriever", "policies", "retriever", "frozen documents", style=EDGE_POLICY)
    d.edge("e_retriever_b", "retriever", "config_b", "ranked chunks + source IDs", style=EDGE_POLICY)
    d.edge("e_retriever_c", "retriever", "config_c", "ranked chunks + source IDs", style=EDGE_POLICY)

    # Persistence and evaluation flow.
    d.edge("e_response_runs", "response", "runs", "audit record", style=EDGE_EVAL)
    d.edge("e_runs_jira_trigger", "runs", "jira_trigger", "operational C persisted event", style=EDGE_EVAL)
    d.edge("e_jira_trigger_cloud", "jira_trigger", "jira_cloud", "allowlisted create/find", style=EDGE_EVAL)
    d.edge("e_jira_cloud_monitor", "jira_cloud", "jira_monitor", "open Epic children", style=EDGE_EVAL)
    d.edge("e_jira_monitor_cloud", "jira_monitor", "jira_cloud", "bounded labels/comments", style=EDGE_POLICY)
    d.edge("e_runs_metrics", "runs", "metrics", "stored outputs", style=EDGE_EVAL)
    d.edge("e_runs_annotations", "runs", "annotations", "blinded task payload", style=EDGE_EVAL)
    d.edge("e_manifest_audit", "manifest", "audit", "declared inputs + state", style=EDGE_EVAL)
    d.edge("e_metrics_api", "metrics", "fastapi", "evaluation endpoints", style=EDGE_EVAL)
    d.edge("e_annotations_api", "annotations", "fastapi", "annotation endpoints", style=EDGE_EVAL)
    d.edge("e_audit_api", "audit", "fastapi", "artifact audit endpoint", style=EDGE_EVAL)
    return d


def build_xml() -> bytes:
    diagram = build_diagram()
    modified = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": modified,
            "agent": "thesis_code implemented architecture generator",
            "version": "24.7.17",
            "type": "device",
            "compressed": "false",
            "pages": "1",
        },
    )
    page = ET.SubElement(mxfile, "diagram", {"id": "implemented-architecture", "name": "Implemented Architecture"})
    page.append(diagram.model)
    ET.indent(mxfile, space="  ")
    return ET.tostring(mxfile, encoding="utf-8", xml_declaration=True)


def validate(xml_bytes: bytes) -> tuple[int, int]:
    text = xml_bytes.decode("utf-8").lower()
    forbidden = ("planned", "ieee-cis", "production gap", "data:image/png", "<image")
    present = [term for term in forbidden if term in text]
    if present:
        raise ValueError(f"Non-current or raster content found: {present}")
    root = ET.fromstring(xml_bytes)
    pages = root.findall("diagram")
    if len(pages) != 1:
        raise ValueError(f"Expected one diagram page, found {len(pages)}")
    vertices = root.findall(".//mxCell[@vertex='1']")
    edges = root.findall(".//mxCell[@edge='1']")
    required = (
        "src/api.py",
        "src/abc_service.py",
        "src/retrieval.py",
        "src/safety.py",
        "src/abc_evaluation.py",
        "artifacts/unified/unified_model_evidence.json",
    )
    missing = [term for term in required if term not in xml_bytes.decode("utf-8")]
    if missing:
        raise ValueError(f"Required implemented components missing: {missing}")
    if len(vertices) < 35 or len(edges) < 30:
        raise ValueError(f"Diagram unexpectedly sparse: {len(vertices)} vertices, {len(edges)} edges")
    return len(vertices), len(edges)


def diagrams_net_url(xml_bytes: bytes, title: str) -> str:
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(xml_bytes) + compressor.flush()
    encoded = urllib.parse.quote(base64.b64encode(compressed).decode("ascii"), safe="")
    return f"https://app.diagrams.net/?title={urllib.parse.quote(title)}#R{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open-chrome", action="store_true", help="Open the editable diagram in Google Chrome")
    args = parser.parse_args()

    xml_bytes = build_xml()
    vertices, edges = validate(xml_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(xml_bytes)
    print(f"Wrote {args.output} (1 page, {vertices} vertices, {edges} connectors)")

    if args.open_chrome:
        url = diagrams_net_url(xml_bytes, args.output.name)
        subprocess.run(["open", "-a", "Google Chrome", url], check=True)


if __name__ == "__main__":
    main()
