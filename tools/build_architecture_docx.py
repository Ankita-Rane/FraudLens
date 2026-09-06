"""Build the thesis architecture as native editable Word tables and text.

The generator writes WordprocessingML directly with the Python standard library. The
post-build validation rejects embedded media and confirms that native Word tables are
present, so the architecture is not a flattened PNG or screenshot.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Fraud Investigation Framework Architecture</title>
<style>
  @page { size: A4 landscape; margin: 12mm; }
  body { font-family: Arial, Helvetica, sans-serif; color: #172033; font-size: 9.5pt; line-height: 1.26; }
  h1 { color: #12355b; font-size: 23pt; margin: 0 0 5pt 0; }
  h2 { color: #12355b; font-size: 15pt; border-bottom: 2px solid #2f6b9a; padding-bottom: 3pt; margin-top: 18pt; }
  h3 { color: #1d4d6f; font-size: 11.5pt; margin: 12pt 0 4pt 0; }
  p { margin: 3pt 0 6pt 0; }
  ul, ol { margin: 3pt 0 7pt 18pt; }
  li { margin-bottom: 2pt; }
  table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; }
  th { background: #12355b; color: white; font-weight: bold; padding: 5pt; border: 1px solid #73869a; }
  td { padding: 5pt; border: 1px solid #9aa9b8; vertical-align: top; }
  .cover { background: #eef4f8; border-left: 8px solid #12355b; padding: 18pt; margin-bottom: 14pt; }
  .status { background: #e8f5e9; border: 1px solid #78a67b; padding: 7pt; }
  .warning { background: #fff5dc; border: 1px solid #c8932f; padding: 7pt; }
  .boundary { background: #2b3e50; color: white; font-weight: bold; text-align: center; }
  .zone { background: #dce8f2; color: #12355b; font-weight: bold; width: 10%; }
  .live { background: #e4f3e8; border: 2px solid #3b7d52; }
  .prototype { background: #e9f0fa; border: 2px solid #416e9c; }
  .planned { background: #fff3d6; border: 2px dashed #b47b18; }
  .external { background: #f2e9f7; border: 2px dotted #76518e; }
  .control { background: #fbe8e7; border: 2px solid #a94b45; }
  .arrow { border: none; text-align: center; vertical-align: middle; font-size: 17pt; color: #305c7d; width: 4%; }
  .small { font-size: 8.2pt; color: #34495e; }
  .tag { font-weight: bold; letter-spacing: .3pt; }
  .center { text-align: center; }
  .page { page-break-before: always; }
  .tight td { padding: 3.5pt; }
  .num { width: 4%; text-align: center; font-weight: bold; }
  .source { font-size: 8.5pt; word-break: break-all; }
</style>
</head>
<body>

<div class="cover">
  <h1>Credit-Card Fraud Investigation Framework</h1>
  <h3>Industry-grade logical architecture and independent real-transaction validation design</h3>
  <p><b>Repository:</b> thesis_code &nbsp; | &nbsp; <b>Architecture baseline:</b> 28 August 2026</p>
  <p><b>Document form:</b> Native editable Word tables and text. No PNG, screenshot, or raster architecture is embedded.</p>
</div>

<div class="status"><b>Scope truth:</b> ULB and Sparkov detection, source-labelled evidence fusion, A/B/C orchestration, FastAPI, Node.js UI, local Qwen integration, public-source retrieval, Configuration C controls, telemetry, and evaluation scaffolding exist as a research prototype. IEEE-CIS external validation and production bank controls are planned. Configuration C is a guarded assistant, not yet a tool-using agent.</div>

<h2>1. What the framework is building</h2>
<p>The system asks a controlled research question: when the <i>same fraud-model evidence and analyst question</i> are passed through three explanation configurations, do retrieval grounding and enforceable guardrails improve explanation reliability and governance at an acceptable operational cost?</p>

<table>
  <tr><th>Configuration</th><th>Intentional treatment</th><th>What must remain identical</th></tr>
  <tr><td><b>A — direct baseline</b></td><td>Model evidence → local LLM; no policy retrieval; no enforced C controls.</td><td rowspan="3">Frozen transaction/evidence, question, LLM checkpoint, prompt-contract version, decoding parameters, run count, and evaluation protocol.</td></tr>
  <tr><td><b>B — retrieval augmented</b></td><td>Same model evidence + frozen public-source policy retrieval → local LLM; no explicit C enforcement.</td></tr>
  <tr><td><b>C — guarded RAG</b></td><td>Same evidence and retrieval as B + input blocking, model-score generation gate, strict structure, citation allow-list, claim-level citations, scope constraints, and traces.</td></tr>
</table>

<p><b>Candidate project contribution, pending chapter review:</b> a controlled, source-labelled A/B/C investigation layer that separates heterogeneous fraud detectors from explanation generation and evaluates claim reliability, governance, and operational behavior without leaking offline labels. This is not yet a defensible global novelty claim.</p>

<h2 class="page">2. Editable logical architecture</h2>
<p class="small"><b>Legend:</b> green = implemented; blue = research prototype; amber dashed = planned; purple dotted = external dependency/source; red = enforced control. Solid arrows describe runtime data flow. Evaluation/control flow is labelled explicitly.</p>

<table class="tight">
  <tr><td class="boundary" colspan="9">TRUST BOUNDARY 1 — RESEARCH DATA ZONE (local files, frozen hashes, no raw-row public redistribution)</td></tr>
  <tr>
    <td class="zone">Sources</td>
    <td class="live" colspan="2"><span class="tag">[LIVE] ULB / Worldline</span><br>Real, anonymized benchmark<br><span class="small">CSV: Time, Amount, V1–V28, Class</span></td>
    <td class="live" colspan="2"><span class="tag">[LIVE] Sparkov</span><br>Synthetic behavioral benchmark<br><span class="small">Train/test CSV; interpretable attributes</span></td>
    <td class="external" colspan="3"><span class="tag">[EXTERNAL—CONDITIONAL] IEEE-CIS / Vesta</span><br>Real e-commerce payment transactions<br><span class="small">Kaggle-controlled; rules and thesis-use permission must be confirmed</span></td>
  </tr>
  <tr>
    <td class="zone">Ingest</td>
    <td class="arrow">↓</td><td class="live"><b>ULB validator</b><br>schema, duplicates, chronology, hash</td>
    <td class="arrow">↓</td><td class="live"><b>Sparkov validator</b><br>schema, identity-field exclusion, temporal split, hash</td>
    <td class="arrow">↓</td><td class="planned" colspan="2"><b>External quarantine + adapter</b><br>terms record, hash, masked schema, label isolation</td>
  </tr>
  <tr>
    <td class="zone">Detection</td>
    <td class="arrow">↓</td><td class="live"><b>ULB Random Forest pipeline</b><br>validation-selected threshold; untouched natural test</td>
    <td class="arrow">↓</td><td class="live"><b>Sparkov Random Forest pipeline</b><br>later-period holdout; grouped perturbation pilot</td>
    <td class="arrow">↓</td><td class="planned" colspan="2"><b>IEEE-CIS-specific detector</b><br>chronological labelled holdout; no forced ULB/Sparkov mapping</td>
  </tr>
  <tr><td class="boundary" colspan="9">TRUST BOUNDARY 2 — VERSIONED MODEL-EVIDENCE ZONE</td></tr>
  <tr>
    <td class="zone">Evidence</td>
    <td class="arrow">→</td><td class="live" colspan="6"><b>Federated evidence contract</b><br>dataset_id • model/version • record_id • score • operating threshold • alert decision • nested dataset-specific features • attribution method/limitations • evidence_id • artifact hashes<br><span class="small">Offline actual label is excluded before the LLM boundary.</span></td><td class="arrow">→</td>
  </tr>
  <tr><td class="boundary" colspan="9">TRUST BOUNDARY 3 — EXPLANATION AND POLICY ZONE</td></tr>
  <tr>
    <td class="zone">Policy</td>
    <td class="external" colspan="2"><b>Frozen public-source corpus</b><br>CFPB • Federal Reserve • NIST • BIS<br><span class="small">Provenance registry + content hashes</span></td>
    <td class="arrow">→</td><td class="prototype"><b>TF-IDF retriever</b><br>transaction-aware privacy-minimised query<br>top-k evidence IDs/ranks/similarity</td>
    <td class="arrow">→</td><td class="prototype" colspan="2"><b>B and C only</b><br>retrieval evidence added to prompt; A receives none</td>
  </tr>
  <tr>
    <td class="zone">Orchestration</td>
    <td class="prototype" colspan="2"><b>A strategy</b><br>direct natural-language answer</td>
    <td class="prototype" colspan="2"><b>B strategy</b><br>retrieval-grounded natural-language answer</td>
    <td class="control" colspan="3"><b>C guarded strategy</b><br>pre-generation score gate → strict JSON → evidence/citation allow-list → per-claim citation structure → scope validation</td>
  </tr>
  <tr>
    <td class="zone">Generation</td>
    <td class="arrow">↘</td><td class="external" colspan="5"><b>Local Qwen provider (Qwen/Qwen3-0.6B)</b><br>lazy Hugging Face load; shared generation interface; token/latency/resource capture<br><span class="small">Deterministic demo engine is for software demonstration only and cannot be reported as an LLM experiment.</span></td><td class="arrow">↙</td>
  </tr>
  <tr><td class="boundary" colspan="9">TRUST BOUNDARY 4 — APPLICATION, HUMAN DECISION AND AUDIT ZONE</td></tr>
  <tr>
    <td class="zone">Application</td>
    <td class="live" colspan="2"><b>FastAPI investigation service</b><br>REST/JSON /api/v1; controlled comparison and annotation endpoints</td>
    <td class="arrow">⇄</td><td class="live"><b>Node.js proxy + chatbot UI</b><br>same case/question to A/B/C; no direct LLM access</td>
    <td class="arrow">⇄</td><td class="control" colspan="2"><b>Human investigator/reviewer</b><br>decision authority; blinded claim assessment; no automatic adverse action</td>
  </tr>
  <tr>
    <td class="zone">Audit + evaluation</td>
    <td class="live" colspan="2"><b>Immutable JSON run records</b><br>prompt hash, evidence IDs, model/engine, status, latency, tokens, cost assumptions, resources</td>
    <td class="arrow">→</td><td class="prototype"><b>Paired evaluator</b><br>A/B/C completeness, repeated consistency, claim units, reviewer agreement</td>
    <td class="arrow">→</td><td class="planned" colspan="2"><b>Thesis inference gate</b><br>effect estimates + confidence intervals + limitations; no conclusion until frozen Qwen runs and blinded labels exist</td>
  </tr>
</table>

<div class="warning"><b>Non-negotiable interpretation:</b> the LLM explains model evidence and policy context. It does not independently establish that a transaction is fraudulent, and policy citations do not by themselves prove that each claim is entailed.</div>

<h2 class="page">3. How components communicate</h2>
<table>
  <tr><th>#</th><th>Producer → consumer</th><th>Interface / protocol</th><th>Payload and control</th><th>Failure behavior</th></tr>
  <tr><td class="num">1</td><td>ULB/Sparkov CSV → modelling notebooks</td><td>Local filesystem; pandas CSV; SHA-256 manifest</td><td>Validated schema, split metadata, naturally imbalanced holdout</td><td>Fail closed on missing target/schema or manifest mismatch</td></tr>
  <tr><td class="num">2</td><td>Detector pipeline → evidence builder</td><td>joblib pipeline + CSV/JSON metrics/predictions</td><td>Score, threshold, alert, nested feature evidence, model identity</td><td>Reject missing provenance or incompatible contract</td></tr>
  <tr><td class="num">3</td><td>Evidence builder → InvestigationService</td><td>Versioned JSON / JSONL; Python repository protocol</td><td>Source-labelled federated bundle; no cross-schema feature equivalence</td><td>No generation when required evidence is absent</td></tr>
  <tr><td class="num">4</td><td>Node browser → Node proxy → FastAPI</td><td>HTTP REST + JSON on localhost</td><td>configuration or comparison, sample_id, frozen question, engine, controls</td><td>4xx for validation; 502 if Python API is unavailable</td></tr>
  <tr><td class="num">5</td><td>Policy corpus → retriever → B/C</td><td>Frozen HTML + provenance JSON; TF-IDF top-k</td><td>Evidence ID, exact chunk, source/version/hash, rank, similarity</td><td>B/C fail when approved corpus is empty; A is isolated from policy</td></tr>
  <tr><td class="num">6</td><td>Strategy → Qwen provider</td><td>In-process TextGenerator protocol; Hugging Face generate</td><td>Prompt contract, compact evidence, max_new_tokens</td><td>Unavailable dependency is surfaced; demo result is tagged separately</td></tr>
  <tr><td class="num">7</td><td>C output → validator</td><td>JSON parse + deterministic programmatic checks</td><td>Exact keys, risk enum, null self-confidence, citations and per-claim allow-list</td><td>Block invalid output; do not silently repair it</td></tr>
  <tr><td class="num">8</td><td>Service → run repository</td><td>Atomic local JSON write</td><td>Answer, evidence, checks/violations, prompt hash, engine, telemetry</td><td>Refuse overwrite of an existing request ID</td></tr>
  <tr><td class="num">9</td><td>Run repository → evaluator/UI</td><td>REST evaluation endpoints + JSON artifacts</td><td>Automatic metrics, claim-unit tasks, blinded annotations, agreement</td><td>Human-dependent metrics remain null until eligible annotations exist</td></tr>
  <tr><td class="num">10</td><td>Independent dataset → shared explanation layer</td><td>Planned dataset adapter emitting the same evidence contract</td><td>Independent detector score and dataset-specific fields; label kept separately</td><td>Never force-map IEEE-CIS fields into ULB or Sparkov features</td></tr>
</table>

<h3>Runtime sequence for one chatbot comparison</h3>
<table class="tight">
  <tr><td class="num">1</td><td>Analyst selects a frozen transaction sample and asks one investigation question.</td><td class="arrow">→</td><td class="num">2</td><td>Node posts one comparison request; FastAPI counterbalances A/B/C run order.</td></tr>
  <tr><td class="num">3</td><td>Service loads the same source-labelled model evidence for all conditions and excludes the evaluation label.</td><td class="arrow">→</td><td class="num">4</td><td>A receives model evidence only; B/C receive the same frozen retrieval results.</td></tr>
  <tr><td class="num">5</td><td>The same Qwen provider generates each condition. C may suppress low-score cases or block unsafe/invalid output.</td><td class="arrow">→</td><td class="num">6</td><td>Every run is stored with evidence IDs, hashes, timing, token and resource telemetry.</td></tr>
  <tr><td class="num">7</td><td>UI renders answers and control outcomes; the user does not see ground truth during explanation review.</td><td class="arrow">→</td><td class="num">8</td><td>Evaluator later joins labels and blinded claim annotations, then estimates A/B/C differences.</td></tr>
</table>

<h2 class="page">4. Independent real-transaction validation lane</h2>
<h3>Selected Kaggle candidate: IEEE-CIS Fraud Detection (conditional)</h3>
<p>The official Kaggle dataset describes online-transaction fraud with binary target <b>isFraud</b>, separate transaction and identity files joined by <b>TransactionID</b>, payment-card fields, device/identity fields, and masked engineered variables. In the official host discussion, Vesta states that the rows are real rather than synthetic and describes chargeback-linked label creation. The same discussion acknowledges possible false negatives when fraud is not reported.</p>

<div class="warning"><b>Access gate:</b> Kaggle marks the data “Subject to Competition Rules.” Before thesis use, the researcher must accept and preserve the applicable terms and confirm they permit the intended academic experiment. Raw data and row-level extracts must not be committed or redistributed without explicit permission. This architecture identifies a candidate; it does not override Kaggle’s terms.</div>

<h3>Why it is outside the development datasets</h3>
<table>
  <tr><th>Property</th><th>ULB</th><th>Sparkov</th><th>IEEE-CIS candidate</th></tr>
  <tr><td>Generation</td><td>Real, anonymized benchmark</td><td>Synthetic</td><td>Host-confirmed real e-commerce payment records</td></tr>
  <tr><td>Provider/process</td><td>ULB/Worldline benchmark</td><td>Sparkov simulator</td><td>Vesta fraud-protection ecosystem</td></tr>
  <tr><td>Features</td><td>Time, Amount, V1–V28</td><td>Behavioral, merchant, geography, demographics</td><td>Payment card, product, device/identity, masked relations</td></tr>
  <tr><td>Role</td><td>Development/test benchmark</td><td>Development/later holdout</td><td>Locked external validity stress test; never merged into training</td></tr>
  <tr><td>Label exposure to LLM</td><td colspan="3" class="center"><b>Never.</b> Ground truth is evaluator-only for every source.</td></tr>
</table>

<h3>End-to-end real-sample experiment</h3>
<ol>
  <li><b>Authorize and freeze:</b> accept applicable Kaggle terms, record acquisition date, hash <code>train_transaction.csv</code> and optional identity file, and keep them in the ignored local directory.</li>
  <li><b>Quarantine and validate:</b> test schema, missingness, duplicate TransactionID values, chronological field semantics, label balance, and train/holdout boundaries.</li>
  <li><b>Train independently:</b> build an IEEE-CIS-specific preprocessing/model pipeline. Never merge its rows or force its features into ULB/Sparkov models.</li>
  <li><b>Lock a labelled time holdout:</b> use only the labelled training data for local evaluation because Kaggle does not provide test labels. Select the detector and threshold without touching the final period.</li>
  <li><b>Create a prespecified case panel:</b> sample true positives, false positives, false negatives and true negatives across score bands. Store prompt-safe fields separately from evaluator labels.</li>
  <li><b>Normalize evidence, not features:</b> emit dataset_id, model/version, score, threshold, alert, nested IEEE-CIS features, attribution limits, and stable evidence IDs.</li>
  <li><b>Freeze the comparison:</b> send identical cases and questions to A/B/C with the same Qwen checkpoint and decoding settings. B and C use the same retrieved chunks.</li>
  <li><b>Evaluate after generation:</b> rejoin labels only in the evaluator. Score detection separately from claim reliability, citation correctness/coverage, guardrail behavior, consistency, latency, cost and resources.</li>
  <li><b>Blind human assessment:</b> randomize output order, hide configuration identity, freeze atomic claim units, use at least two reviewers, and report agreement.</li>
  <li><b>Conclude conservatively:</b> report per-dataset estimates and uncertainty. A result that improves governance while increasing latency or suppressing useful cases is a trade-off, not an unconditional win.</li>
</ol>

<p><b>Prepared utility:</b> <code>tools/prepare_ieee_cis_external_panel.py</code> creates separate prompt and evaluator-label files. Without detector predictions it deliberately marks the output as a draft; with frozen predictions it can form TP/FP/FN/TN strata.</p>

<h2 class="page">5. Controls, metrics and conclusion gates</h2>
<table>
  <tr><th>Layer</th><th>Implemented / prototype controls</th><th>Highest-priority production gap</th></tr>
  <tr><td>Data</td><td>Input manifest and hashes; schema checks; label separation; dataset identity retained</td><td>Data catalog/lineage service, encrypted secrets/objects, retention/deletion policy, access audit</td></tr>
  <tr><td>Detection</td><td>Leakage-safe split/resampling; held-out metrics; model artifacts and thresholds</td><td>Model registry, reproducible build image, drift/quality monitors, calibrated probabilities and robust external validation</td></tr>
  <tr><td>Retrieval</td><td>Frozen official-source corpus; provenance, chunk hashes, top-k and similarity telemetry</td><td>Institution-approved policy set, semantic/lexical hybrid retrieval, access control, gold-set sign-off and drift/version governance</td></tr>
  <tr><td>LLM / C</td><td>Input pattern controls, model-score gate, strict JSON, citation allow-list, per-claim structure, scope disclosure</td><td>Semantic claim-entailment checks, calibrated abstention, adversarial test suite, prompt/model registry, safe bounded tools if “agentic” is claimed</td></tr>
  <tr><td>Application</td><td>FastAPI validation, fixed CORS origins, Node proxy, immutable local run files</td><td>Authentication/RBAC, TLS, rate limiting, durable database/object store, queue/retries/idempotency, centralized logs and alerting</td></tr>
  <tr><td>Human governance</td><td>Ground-truth withholding, blinded claim task design, reviewer agreement endpoint</td><td>Reviewer training, adjudication, escalation SLA, documented decision authority and prohibited automatic actions</td></tr>
</table>

<h3>Measurement model</h3>
<table class="tight">
  <tr><th>Family</th><th>Measures</th><th>Valid inference</th></tr>
  <tr><td>Detection</td><td>PR-AUC, precision, recall, F2, calibration, FP/FN, threshold behavior</td><td>How well each dataset-specific detector ranks/classifies its own locked holdout</td></tr>
  <tr><td>Explanation</td><td>Claim hallucination rate, citation correctness, policy grounding completeness, claim coverage</td><td>Requires frozen atomic claims and blinded human evidence assessment</td></tr>
  <tr><td>Governance</td><td>Guardrail violations, trace completeness, repeated explanation consistency, suppression/blocking outcomes</td><td>Whether controls are enforced and auditable; not whether every claim is true</td></tr>
  <tr><td>Operations</td><td>End-to-end/generation/retrieval latency, token-estimated cost, CPU/memory and later GPU telemetry</td><td>Resource and performance trade-offs under the frozen experiment environment</td></tr>
</table>

<h3>Minimum evidence before a thesis conclusion</h3>
<ul>
  <li>All A/B/C cells are paired for the same frozen cases/questions and use real Qwen generation rather than the demo engine.</li>
  <li>Required human annotation metrics are non-null, reviewer agreement is reported, and exclusions are documented.</li>
  <li>Retrieval quality is evaluated on an approved gold set; citation presence is not substituted for correctness.</li>
  <li>At least one independent real-data result is reported only after lawful access and a dataset-specific pipeline.</li>
  <li>Effect estimates and uncertainty are reported alongside failures, suppression rates, latency/cost and resource trade-offs.</li>
</ul>

<h2 class="page">6. Sources and architecture decisions</h2>
<table>
  <tr><th>Source / artifact</th><th>Use in this document</th></tr>
  <tr><td class="source"><a href="https://www.kaggle.com/competitions/ieee-fraud-detection/data">Kaggle IEEE-CIS data page</a></td><td>Target, transaction/identity file structure, join key, feature families, file availability, and competition-rule licensing status.</td></tr>
  <tr><td class="source"><a href="https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/101203">Official Vesta host data description</a></td><td>Real/not-synthetic statement, feature interpretation, label process, and acknowledged unreported-fraud limitation.</td></tr>
  <tr><td><code>src/abc_service.py</code>, <code>src/llm_backend.py</code>, <code>src/api.py</code></td><td>Current orchestration, A/B/C contracts, C enforcement, run traces, paired API and annotation interfaces.</td></tr>
  <tr><td><code>src/retrieval.py</code>, <code>datasets/policy_sources/</code></td><td>Current frozen local policy ingestion and TF-IDF retrieval boundary.</td></tr>
  <tr><td><code>web/server.mjs</code> and <code>web/public/</code></td><td>Node proxy/UI boundary and lack of direct browser-to-LLM coupling.</td></tr>
  <tr><td><code>datasets/input_artifacts.json</code></td><td>Current immutable-input and hash policy.</td></tr>
</table>

<p class="small">Architecture type: logical/research reference architecture. It is intentionally honest about prototype status and does not imply PCI DSS certification, bank deployment approval, calibrated LLM confidence, autonomous fraud decisions, or verified global novelty.</p>

</body>
</html>
"""


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def xml_bytes(element: ET.Element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


@dataclass
class HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["HtmlNode | str"] = field(default_factory=list)


class ArchitectureHtmlParser(HTMLParser):
    VOID_TAGS = {"br", "meta", "link", "hr", "img", "input"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("root")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = HtmlNode(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def classes(node: HtmlNode) -> set[str]:
    return set(node.attrs.get("class", "").split())


def descendants(node: HtmlNode, tag: str) -> list[HtmlNode]:
    found: list[HtmlNode] = []
    for child in node.children:
        if isinstance(child, HtmlNode):
            if child.tag == tag:
                found.append(child)
            found.extend(descendants(child, tag))
    return found


def inline_text(node: HtmlNode | str) -> str:
    if isinstance(node, str):
        return node
    if node.tag == "br":
        return "\n"
    text = "".join(inline_text(child) for child in node.children)
    if node.tag == "a" and node.attrs.get("href"):
        return f"{text} [{node.attrs['href']}]"
    return text


def clean_text(node: HtmlNode) -> str:
    raw = inline_text(node).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
    return "\n".join(line for line in lines if line)


def add_run(
    paragraph: ET.Element,
    text: str,
    *,
    bold: bool = False,
    color: str | None = None,
    size: int | None = None,
) -> None:
    run = ET.SubElement(paragraph, w("r"))
    properties = ET.SubElement(run, w("rPr"))
    if bold:
        ET.SubElement(properties, w("b"))
    if color:
        ET.SubElement(properties, w("color"), {w("val"): color})
    if size:
        ET.SubElement(properties, w("sz"), {w("val"): str(size)})
        ET.SubElement(properties, w("szCs"), {w("val"): str(size)})
    parts = text.split("\n")
    for index, part in enumerate(parts):
        if index:
            ET.SubElement(run, w("br"))
        value = ET.SubElement(run, w("t"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
        value.text = part


def add_paragraph(
    parent: ET.Element,
    text: str,
    *,
    style: str = "Normal",
    page_break: bool = False,
    bold: bool = False,
    color: str | None = None,
    size: int | None = None,
    align: str | None = None,
    shade: str | None = None,
    border_color: str | None = None,
    prefix: str = "",
) -> ET.Element:
    paragraph = ET.SubElement(parent, w("p"))
    properties = ET.SubElement(paragraph, w("pPr"))
    ET.SubElement(properties, w("pStyle"), {w("val"): style})
    if page_break:
        ET.SubElement(properties, w("pageBreakBefore"))
    if align:
        ET.SubElement(properties, w("jc"), {w("val"): align})
    if shade:
        ET.SubElement(properties, w("shd"), {w("val"): "clear", w("fill"): shade})
    if border_color:
        borders = ET.SubElement(properties, w("pBdr"))
        ET.SubElement(
            borders,
            w("left"),
            {w("val"): "single", w("sz"): "18", w("space"): "5", w("color"): border_color},
        )
    add_run(paragraph, prefix + text, bold=bold, color=color, size=size)
    return paragraph


CELL_COLORS = {
    "boundary": "2B3E50",
    "zone": "DCE8F2",
    "live": "E4F3E8",
    "prototype": "E9F0FA",
    "planned": "FFF3D6",
    "external": "F2E9F7",
    "control": "FBE8E7",
}


def add_cell_properties(cell: ET.Element, node: HtmlNode, *, header: bool) -> tuple[str | None, bool]:
    properties = ET.SubElement(cell, w("tcPr"))
    colspan = int(node.attrs.get("colspan", "1") or "1")
    if colspan > 1:
        ET.SubElement(properties, w("gridSpan"), {w("val"): str(colspan)})
    node_classes = classes(node)
    shade = "12355B" if header else next((CELL_COLORS[name] for name in CELL_COLORS if name in node_classes), None)
    if shade:
        ET.SubElement(properties, w("shd"), {w("val"): "clear", w("fill"): shade})
    margins = ET.SubElement(properties, w("tcMar"))
    for side in ("top", "left", "bottom", "right"):
        ET.SubElement(margins, w(side), {w("w"): "90", w("type"): "dxa"})
    if "arrow" in node_classes:
        ET.SubElement(properties, w("vAlign"), {w("val"): "center"})
    return shade, header or "boundary" in node_classes or "zone" in node_classes


def add_table(parent: ET.Element, node: HtmlNode) -> None:
    table = ET.SubElement(parent, w("tbl"))
    properties = ET.SubElement(table, w("tblPr"))
    ET.SubElement(properties, w("tblStyle"), {w("val"): "TableGrid"})
    ET.SubElement(properties, w("tblW"), {w("w"): "0", w("type"): "auto"})
    ET.SubElement(properties, w("tblLayout"), {w("type"): "autofit"})
    borders = ET.SubElement(properties, w("tblBorders"))
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(borders, w(side), {w("val"): "single", w("sz"): "5", w("color"): "8DA0B3"})

    direct_rows = [child for child in node.children if isinstance(child, HtmlNode) and child.tag == "tr"]
    if not direct_rows:
        direct_rows = descendants(node, "tr")
    for row_node in direct_rows:
        row = ET.SubElement(table, w("tr"))
        cell_nodes = [
            child
            for child in row_node.children
            if isinstance(child, HtmlNode) and child.tag in {"td", "th"}
        ]
        for cell_node in cell_nodes:
            cell = ET.SubElement(row, w("tc"))
            header = cell_node.tag == "th"
            _, bold = add_cell_properties(cell, cell_node, header=header)
            node_classes = classes(cell_node)
            text_color = "FFFFFF" if header or "boundary" in node_classes else None
            alignment = "center" if {"center", "arrow", "num", "boundary"} & node_classes else None
            size = 28 if "arrow" in node_classes else (16 if "small" in node_classes else 18)
            add_paragraph(
                cell,
                clean_text(cell_node),
                style="Small" if "small" in node_classes else "Normal",
                bold=bold,
                color=text_color,
                size=size,
                align=alignment,
            )


def render_body(document_body: ET.Element, body_node: HtmlNode) -> None:
    for child in body_node.children:
        if not isinstance(child, HtmlNode):
            continue
        node_classes = classes(child)
        if child.tag == "div" and "cover" in node_classes:
            for nested in child.children:
                if isinstance(nested, HtmlNode) and nested.tag in {"h1", "h2", "h3", "p"}:
                    style = {"h1": "Title", "h2": "Heading1", "h3": "Heading2", "p": "Normal"}[nested.tag]
                    add_paragraph(document_body, clean_text(nested), style=style, shade="EEF4F8", border_color="12355B")
        elif child.tag == "div":
            shade = "E8F5E9" if "status" in node_classes else "FFF5DC" if "warning" in node_classes else None
            border = "78A67B" if "status" in node_classes else "C8932F" if "warning" in node_classes else None
            add_paragraph(document_body, clean_text(child), shade=shade, border_color=border)
        elif child.tag == "h1":
            add_paragraph(document_body, clean_text(child), style="Title")
        elif child.tag == "h2":
            add_paragraph(document_body, clean_text(child), style="Heading1", page_break="page" in node_classes)
        elif child.tag == "h3":
            add_paragraph(document_body, clean_text(child), style="Heading2")
        elif child.tag == "p":
            add_paragraph(document_body, clean_text(child), style="Small" if "small" in node_classes else "Normal")
        elif child.tag == "table":
            add_table(document_body, child)
        elif child.tag in {"ul", "ol"}:
            items = [item for item in child.children if isinstance(item, HtmlNode) and item.tag == "li"]
            for index, item in enumerate(items, start=1):
                prefix = "• " if child.tag == "ul" else f"{index}. "
                add_paragraph(document_body, clean_text(item), prefix=prefix)


def styles_xml() -> bytes:
    styles = ET.Element(w("styles"))
    defaults = ET.SubElement(styles, w("docDefaults"))
    run_default = ET.SubElement(ET.SubElement(defaults, w("rPrDefault")), w("rPr"))
    ET.SubElement(run_default, w("rFonts"), {w("ascii"): "Arial", w("hAnsi"): "Arial"})
    ET.SubElement(run_default, w("sz"), {w("val"): "19"})
    paragraph_default = ET.SubElement(ET.SubElement(defaults, w("pPrDefault")), w("pPr"))
    ET.SubElement(paragraph_default, w("spacing"), {w("after"): "90", w("line"): "250", w("lineRule"): "auto"})

    def style(style_id: str, name: str, size: int, color: str, *, bold: bool = False, before: int = 0, after: int = 100) -> None:
        element = ET.SubElement(styles, w("style"), {w("type"): "paragraph", w("styleId"): style_id})
        ET.SubElement(element, w("name"), {w("val"): name})
        paragraph_properties = ET.SubElement(element, w("pPr"))
        ET.SubElement(paragraph_properties, w("spacing"), {w("before"): str(before), w("after"): str(after)})
        run_properties = ET.SubElement(element, w("rPr"))
        ET.SubElement(run_properties, w("rFonts"), {w("ascii"): "Arial", w("hAnsi"): "Arial"})
        if bold:
            ET.SubElement(run_properties, w("b"))
        ET.SubElement(run_properties, w("color"), {w("val"): color})
        ET.SubElement(run_properties, w("sz"), {w("val"): str(size)})
        ET.SubElement(run_properties, w("szCs"), {w("val"): str(size)})

    style("Normal", "Normal", 19, "172033", after=90)
    style("Title", "Title", 46, "12355B", bold=True, after=160)
    style("Heading1", "heading 1", 30, "12355B", bold=True, before=220, after=100)
    style("Heading2", "heading 2", 23, "1D4D6F", bold=True, before=150, after=60)
    style("Small", "Small", 16, "34495E", after=60)
    table_style = ET.SubElement(styles, w("style"), {w("type"): "table", w("styleId"): "TableGrid"})
    ET.SubElement(table_style, w("name"), {w("val"): "Table Grid"})
    ET.SubElement(table_style, w("basedOn"), {w("val"): "TableNormal"})
    ET.SubElement(table_style, w("uiPriority"), {w("val"): "59"})
    return xml_bytes(styles)


def document_xml() -> bytes:
    parser = ArchitectureHtmlParser()
    parser.feed(HTML)
    body_candidates = descendants(parser.root, "body")
    if not body_candidates:
        raise RuntimeError("Architecture HTML has no body.")
    document = ET.Element(w("document"))
    body = ET.SubElement(document, w("body"))
    render_body(body, body_candidates[0])
    section = ET.SubElement(body, w("sectPr"))
    ET.SubElement(section, w("pgSz"), {w("w"): "16838", w("h"): "11906", w("orient"): "landscape"})
    ET.SubElement(
        section,
        w("pgMar"),
        {w("top"): "680", w("right"): "680", w("bottom"): "680", w("left"): "680", w("header"): "360", w("footer"): "360", w("gutter"): "0"},
    )
    return xml_bytes(document)


def package_parts() -> dict[str, bytes]:
    return {
        "[Content_Types].xml": b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        "_rels/.rels": b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        "word/_rels/document.xml.rels": b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        "word/document.xml": document_xml(),
        "word/styles.xml": styles_xml(),
        "docProps/core.xml": b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Credit-Card Fraud Investigation Framework Architecture</dc:title>
  <dc:creator>thesis_code research project</dc:creator>
  <dc:subject>Editable industry-grade logical architecture and external validation design</dc:subject>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-08-28T00:00:00Z</dcterms:created>
</cp:coreProperties>''',
        "docProps/app.xml": b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>thesis_code architecture builder</Application>
</Properties>''',
    }


def validate_docx(docx_path: Path) -> dict[str, int]:
    with zipfile.ZipFile(docx_path, "r") as archive:
        names = archive.namelist()
        document = archive.read("word/document.xml").decode("utf-8")
    media = [name for name in names if name.startswith("word/media/")]
    if media:
        raise RuntimeError(f"Architecture must not embed raster/media files: {media}")
    table_count = document.count("<w:tbl>")
    if table_count < 8:
        raise RuntimeError(f"Expected native Word tables; found only {table_count}.")
    for required in (
        "Editable logical architecture",
        "IEEE-CIS Fraud Detection",
        "TRUST BOUNDARY",
        "Configuration C",
    ):
        if required not in document:
            raise RuntimeError(f"Required architecture text missing: {required}")
    return {"native_word_tables": table_count, "embedded_media_files": len(media)}


def build(output_path: Path) -> dict[str, object]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in package_parts().items():
            archive.writestr(name, data)
    validation = validate_docx(output_path)
    return {"output": str(output_path), "bytes": output_path.stat().st_size, **validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/FRAUD_INVESTIGATION_FRAMEWORK_ARCHITECTURE.docx"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args().output))
