# FraudLens

FraudLens is a reproducible research prototype for credit-card fraud detection,
evidence retrieval, local-LLM investigation support, guardrail evaluation, and bounded
Jira integration. It contains one definitive six-condition experiment over independent
ULB and Sparkov evaluation panels.

The project is research software, not a production fraud-decision system. It does not
replace investigators, establish regulatory compliance, or demonstrate generalisation
to a bank deployment.

## Definitive experiment

Study ID: `FOLLOWUP-RETRIEVAL-20260902-02`

- 550 evaluation cases: 150 ULB and 400 Sparkov
- six conditions per case
- 3,300 total condition calls
- 55 locked batches of 10 cases and 60 calls
- one question and one execution per condition/case cell
- ULB and Sparkov analysed separately; dataset pooling is prohibited
- evaluation labels withheld from generation
- local generator: `Qwen/Qwen3-0.6B`, pinned revision
- dense encoder: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision
- no human-review experiment or human-rated outcome

The six conditions are:


| ID          | Retrieval                     | Guardrails |
| ----------- | ----------------------------- | ---------- |
| `A_direct`  | None                          | No         |
| `B_lexical` | TF-IDF lexical                | No         |
| `B_dense`   | Dense                         | No         |
| `C_lexical` | TF-IDF lexical                | Yes        |
| `C_dense`   | Dense                         | Yes        |
| `D_hybrid`  | Hybrid reciprocal-rank fusion | Yes        |


The retrieval diagnostic compares TF-IDF, pinned dense retrieval, and hybrid reciprocal-
rank fusion using researcher-authored known-item queries. It reports Precision@k,
Recall@k, MRR, nDCG@k, Top-1 accuracy, and latency. It is a retrieval-ranking diagnostic,
not evidence that generated answers are factually correct.

Candidate-response metrics are calculated only where generation occurred. Suppressed
runs are not scored as zero-quality candidate text. Released-response metrics may be
zero when nothing reached the user. These two measurement levels must not be conflated.

The V4 analysis uses full-precision p-values, per-dataset Holm adjustment as its primary
reporting family, and a global Holm sensitivity analysis. Null results are not treated
as equivalence evidence.

### Legacy implementation boundary

Some source modules, notebooks and local UI routes retain A/B/C names and a 180-call
replication reservation from the earlier development scaffold. They are retained for
implementation traceability and local integration/Jira testing; they are not the
definitive thesis experiment and must not be used as thesis result evidence. The only
definitive experiment is the 550-case, six-condition, 3,300-record study identified
above. Its locked artefacts are listed below.

## Primary evidence


| Evidence                          | Path                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Locked protocol                   | `datasets/evaluation/followup_retrieval/protocol_locked.json`                                                            |
| Frozen panel manifest             | `datasets/evaluation/followup_retrieval/followup_panel_manifest.json`                                                    |
| Dataset-fixture selector manifest | `datasets/evaluation/dataset_fixture_selection_manifest.json`                                                           |
| Evaluation-only labels            | `datasets/evaluation/followup_retrieval/followup_evaluation_labels.csv`                                                  |
| Known-item diagnostic manifest    | `datasets/evaluation/followup_retrieval/known_item_diagnostic_locked.json`                                               |
| Retrieval results                 | `datasets/evaluation/followup_retrieval/retrieval_diagnostic_results.json`                                               |
| Retrieval audit                   | `datasets/evaluation/followup_retrieval/retrieval_diagnostic_audit.json`                                                 |
| Definitive aggregate              | `artifacts/results/followup/FOLLOWUP-RETRIEVAL-20260902-02-CORRECTED-V4.json`                                            |
| Post-hoc descriptive summary      | `artifacts/results/followup/posthoc/FOLLOWUP-RETRIEVAL-20260902-02-POSTHOC-DESCRIPTIVE-V1/posthoc_summary.json`          |
| Post-hoc audit manifest           | `artifacts/results/followup/posthoc/FOLLOWUP-RETRIEVAL-20260902-02-POSTHOC-DESCRIPTIVE-V1/audit_manifest.json`           |
| Independent post-hoc verification | `artifacts/results/followup/posthoc/FOLLOWUP-RETRIEVAL-20260902-02-POSTHOC-DESCRIPTIVE-V1/independent_verification.json` |
| Post-hoc reporting boundaries     | `artifacts/results/followup/posthoc/FOLLOWUP-RETRIEVAL-20260902-02-POSTHOC-DESCRIPTIVE-V1/THESIS_REPORTING_NOTES.txt`    |
| Human-readable evidence ledger    | `Six_Condition_Evidence_Ledger.html`                                                                                     |


Current SHA-256 values:

```text
protocol     af9fbda7a2edee26cb02ce32f70e0a661d1ab681ab602a6fe5b233792f366ff5
panel        e2b388848a479c6ff932f06f2e39b2cbc72c41d67712ebb61b7a423a9054b774
known item   49b2a73ebff38d15f3390bc041cd49091b7bddb07e02eb83cbd3f238772a4ef5
retrieval    2368088d30cc3c74f0dfc92c83fdcafe20590ee1064c12cce856fac35555619c
audit        ad528ea399a8e376cc13ab545f5f83c9273f4ca94ef9642f458f7e6c9f90ed38
V4 aggregate 219092d03d75c6511896ba0df1f50e3ad38cb7e80e31c8f8537aa1983d79e65e
```

`SHA256SUMS` is the canonical checksum register for these six thesis artefacts. The
post-hoc audit manifest separately records the hashes of every file in the post-hoc
evidence package and the hashes of the code used to generate it.

The V4 aggregate, evidence ledger, and verified post-hoc descriptive package are
committed for review. The 3,300 individual raw-run JSON files are not currently
published in this repository. Therefore, a fresh clone can inspect and test the
released aggregate and post-hoc evidence but cannot independently regenerate them from
the raw calls unless the separately retained raw batch archive is supplied. Do not
claim raw-call reproducibility from the GitHub repository alone.

## Architecture and boundaries

ULB and Sparkov have different schemas and data-generating processes. They are modelled
independently and combined only as source-labelled evidence. ULB anonymised components
such as `V14` are never treated as equivalent to Sparkov features such as merchant
category or distance.

The guardrailed conditions implement score gating, citation structure, source-scope
disclosure, schema checks, input controls, output disposition, and telemetry. Citation
IDs prove provenance membership, not semantic entailment. The system does not use an
LLM confidence score as a calibrated probability.

Jira operates downstream as an optional, bounded operational integration. Jira activity
is excluded from the six-condition experiment. Automated referral/non-release fields are
computational dispositions, not measurements of analyst workload or human-review time.

## Repository contents

```text
artifacts/
  historical/                  committed historical profiles used by the UI
  results/followup/            definitive V4 aggregate
  sparkov/                     committed UI runtime model bundle
  ulb/                         committed UI runtime model bundle
  unified/                     committed source-labelled model evidence
datasets/
  evaluation/                  locked protocol and diagnostic evidence
  policy_sources/              frozen public-source corpus and registry
  sparkov/                     Sparkov inputs and frozen panels
  ulb/                         frozen ULB panels
notebooks/                     EDA, modelling, and walkthrough notebooks
schemas/                       structured response contract
src/                           API, retrieval, evaluation, statistics, and Jira code
tests/                         active automated tests
tools/                         reproducible build, audit, execution, and analysis tools
ui/                            legacy Streamlit interface
web/                           legacy Node.js integration and Jira console
```

Local thesis drafts, spreadsheets, superseded evidence, dormant extensions, agent files,
and audit working material belong under the ignored `ignore/` directory and are not part
of the GitHub research release.

## Dataset availability and public-release boundary

The public repository does not track the source dataset CSVs or the eight derived
transaction-fixture CSVs. Download the source data from the original Kaggle pages and
place each file at the exact case-sensitive repository-relative path below. Do not put
the files in the repository root or rename the Sparkov files. The same source URLs and
full source-file digests appear in the thesis, Appendix B, Table B.1.


| Dataset                                  | Kaggle source                                    | Exact local path                  | Required for fixture rebuild |
| ---------------------------------------- | ------------------------------------------------ | --------------------------------- | ---------------------------- |
| ULB credit-card fraud (real, anonymised) | `kaggle.com/datasets/mlg-ulb/creditcardfraud`    | `datasets/ULB_creditCard.csv`     | Yes                          |
| Sparkov (synthetic) — training file      | `kaggle.com/datasets/kartik2112/fraud-detection` | `datasets/sparkov/fraudTrain.csv` | No; detector retraining only |
| Sparkov (synthetic) — test file          | `kaggle.com/datasets/kartik2112/fraud-detection` | `datasets/sparkov/fraudTest.csv`  | Yes                          |


The ULB download is commonly named `creditcard.csv`; rename that downloaded file to
`ULB_creditCard.csv` when placing it under `datasets/`. From the repository root, create
the destination directories if necessary:

```bash
mkdir -p datasets/ulb datasets/sparkov
```

Verify the two fixture-rebuild inputs before use:

```bash
shasum -a 256 datasets/ULB_creditCard.csv
shasum -a 256 datasets/sparkov/fraudTest.csv
```

The expected source hashes are:

```text
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89  datasets/ULB_creditCard.csv
12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0  datasets/sparkov/fraudTest.csv
```

The rebuild tool checks these hashes automatically and stops before writing output if
either source differs. A mismatch means the downloaded file is not the source file used
by this study; do not adjust row selectors, models, or thresholds to make it pass.

After installing the Python dependencies, run the following command from the repository
root. In a fresh clone, where the generated files do not yet exist, use:

```bash
python tools/rebuild_dataset_fixtures.py --verify
```

If the eight local fixtures already exist and must be regenerated, explicitly allow
their replacement:

```bash
python tools/rebuild_dataset_fixtures.py --overwrite --verify
```

The command creates and verifies these files in place:

| Dataset | Generated path                                           | Rows |
| ------- | -------------------------------------------------------- | ---: |
| ULB     | `datasets/ulb/candidate_transaction_samples.csv`         |   30 |
| ULB     | `datasets/ulb/followup_transaction_samples.csv`          |  150 |
| ULB     | `datasets/ulb/pilot_transaction_samples.csv`             |    8 |
| ULB     | `datasets/ulb/runtime_transaction_samples.csv`           |    4 |
| Sparkov | `datasets/sparkov/candidate_transaction_samples.csv`     |   30 |
| Sparkov | `datasets/sparkov/followup_transaction_samples.csv`      |  400 |
| Sparkov | `datasets/sparkov/pilot_transaction_samples.csv`         |    8 |
| Sparkov | `datasets/sparkov/runtime_transaction_samples.csv`       |    4 |

A successful run ends with `"all_verified": true`. This means all eight generated
files—634 rows in total—match their frozen SHA-256 digests byte for byte. The expected
output hashes, row selectors, column order, formatting rules, detector paths, detector
hashes and thresholds are locked in
`datasets/evaluation/dataset_fixture_selection_manifest.json`.

That manifest contains sample identifiers and source-row selectors, not transaction
feature values. Reconstruction verifies the pinned base detectors and the dedicated ULB
follow-up detector before writing a fixture. The generated CSVs remain ignored by Git,
so rebuilding them does not add raw or derived transaction data to a commit.

The five public regulatory and standards documents that form the retrieval corpus are
listed with their canonical URLs and digests in Appendix B, Table B.2, and are obtained
the same way.

Once the datasets are in place, run the experiment locally as described under
**Definitive experiment** and **Clone and install** below. Full reproduction requires the
environment recorded in the thesis (Section 3.10): Python 3.12.14, PyTorch 2.13.0,
Transformers 5.16.0, sentence-transformers 6.0.1, scikit-learn 1.6.1, SHAP 0.52.0,
NumPy 2.3.2, pandas 2.3.1, SciPy 1.16.1, with the pinned model revisions in Appendix D,
Table D.2. Latency figures were recorded on one Apple M1 machine and will not transfer to
other hardware.

## Prerequisites

- Git and Git LFS
- Python 3.12 or a compatible version
- Node.js 20 or newer for the primary web interface
- sufficient disk space for downloaded datasets and Git LFS model pipelines
- network access for initial dependency and model downloads
- optional Apple MPS support for the recorded local execution path; CPU execution is
  possible but does not reproduce MPS latency measurements

The repository is public and can be cloned without GitHub repository access. Dataset
downloads and any third-party service credentials remain the responsibility of the
reproducing user.

## Clone and install

```bash
git lfs install
git clone https://github.com/Ankita-Rane/FraudLens.git
cd FraudLens
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the full local-LLM and dense-retrieval experiment environment, install the follow-up
requirements instead of the base file:

```bash
python -m pip install -r requirements-followup.txt
```

`requirements-followup.txt` includes `requirements-llm.txt`, which in turn includes the
base `requirements.txt`. The first uncached Qwen or dense-encoder execution downloads
the pinned model files. Local caches are intentionally not committed.

## Verify the clone

Run the automated suite. In a source-only public clone, tests requiring the downloaded
datasets, reconstructed transaction fixtures, or separately retained immutable
3,300-run archive report explicit skips rather than failures:

```bash
python -m pytest -q
```

Verify the hashes of all committed evidence and run the public post-hoc evidence audit:

```bash
shasum -a 256 -c SHA256SUMS
python tools/audit_followup_posthoc_evidence.py
```

After downloading the source datasets and rebuilding the eight transaction fixtures as
described above, run the input audit and test suite again:

```bash
python tools/audit_input_artifacts.py
python -m pytest -q
```

The locked-study post-hoc contract tests additionally require the separately retained
immutable 3,300-run archive. That archive is not distributed in the public repository;
the committed V4 aggregate, post-hoc summary, audit manifest, independent verification,
and reporting notes provide the publishable evidence record.

Check the Node.js source:

```bash
cd web
npm run check
cd ..
```

The deterministic demo engine is suitable for plumbing tests only. Its outputs are not
scientific LLM results.

## Run the interfaces

Primary Node.js console:

```bash
./run_node_ui.sh
```

Open `http://127.0.0.1:3000`. The launcher starts the FastAPI service on
`http://127.0.0.1:8000` and binds both services to the local machine.

Legacy Streamlit interface:

```bash
./run_ui.sh
```

The committed detector and evidence bundle, together with the locally reconstructed
runtime fixtures, makes the UI readiness status available after Git LFS checkout and
fixture reconstruction. It includes both detector pipelines, model manifests, test
metrics, runtime attributions, historical profiles, and unified evidence.

## Jira integration

Copy the example configuration and keep the resulting file local:

```bash
cp .env.example .env
```

Configure, as applicable:

```text
JIRA_BASE_URL
JIRA_USER_EMAIL
JIRA_API_TOKEN
JIRA_PROJECT_KEY
```

Preview and read-only UI paths can be tested without enabling writes. Live Jira actions
require valid Atlassian credentials, project permission, and all relevant approval/write
gates in `.env`. Defaults are disabled. Credentials remain server-side and must never be
committed.

The macOS launcher can read a token from Keychain. On other platforms, provide the token
through a protected environment mechanism. Successful software tests do not establish
that a user has access to an external Jira project.

## Review the existing study

1. Run the input audit and test suite.
2. Inspect the locked protocol and its recorded hashes.
3. Inspect the known-item diagnostic manifest, retrieval results, and diagnostic audit.
4. Inspect the V4 JSON as the machine-readable result source.
5. Inspect the post-hoc summary, audit manifest, independent verification, and reporting
   boundaries. These are descriptive analyses of the completed study and involved no
   new model calls or modification of V4.
6. Use `Six_Condition_Evidence_Ledger.html` as a readable projection, not as the primary
   data source.
7. Keep ULB and Sparkov findings separate.
8. Distinguish candidate metrics, released-response metrics, retrieval ranking, and
   computational disposition.

Do not infer factual correctness from citation membership, absence of effect from a
non-significant result, human workload from an automated referral flag, or production
validity from these two research datasets.

## Run a new experiment

New work must receive a new study ID and new immutable hashes. Do not overwrite the
existing protocol, diagnostic, V4 aggregate, or evidence ledger.

A safe high-level sequence is:

1. Rebuild detector artifacts only if data or modelling changes.
2. Create new frozen panels with `tools/build_followup_panels.py`.
3. Create and audit a new known-item retrieval diagnostic.
4. Build a new protocol with `tools/build_followup_protocol.py`.
5. Run a deterministic pilot using `tools/run_followup_retrieval_experiment.py`.
6. Lock model IDs, revisions, prompts, retrieval parameters, panels, labels, conditions,
   batch membership, and statistical rules before final generation.
7. Execute only the newly registered batches; retain failed attempts separately.
8. Aggregate with `tools/aggregate_followup_study.py` into a new output path.
9. Analyse with `tools/analyse_followup_conditions.py` and preserve every source hash.

Use each tool's `--help` output for its exact arguments. Final generation requires the
explicit `--confirm-generation-authorized` flag and a matching locked protocol. That
flag records authorization; it does not relax any protocol checks.

The existing helper `tools/run_remaining_followup_batches.py` is hard-coded to the
completed study and must not be reused as a new-study protocol. Create a new registered
batch plan instead.

## Rebuild supporting model evidence

The committed runtime bundle is sufficient for UI review. If the datasets, feature
engineering, split, model, threshold, or dependencies change, rerun the dataset-specific
notebooks and regenerate the derived evidence:

```bash
python tools/build_historical_profiles.py
python tools/build_unified_evidence.py --max-alerts-per-dataset 25
python tools/build_runtime_attributions.py --panel runtime
```

Do not manually edit generated metrics, pipelines, attributions, or manifests. A changed
upstream input requires a new manifest and new hashes.

## Licence

Original FraudLens source code is licensed under the MIT License. See `LICENSE`.

The MIT License does not grant rights to third-party datasets, downloaded regulatory or
standards documents, external model weights, third-party libraries, or other material
owned by external parties. Those materials remain subject to their respective owners'
licences and terms. The ULB and Sparkov source datasets are not distributed in the
current repository tree; reproducing users must obtain them from the original sources
listed above.

## Statistical and interpretation rules

- The analysis unit is one transaction within one dataset.
- ULB and Sparkov must not be pooled.
- Paired tests require matching condition/case cells.
- Full-precision p-values must be retained; never report a computed small value as
  exactly `0.0`.
- Candidate metrics exclude cases where generation never occurred.
- Released-response metrics retain the operational consequence of non-release.
- Confidence intervals and effect sizes must accompany significance tests.
- Holm family definitions and sensitivity analyses must be labelled explicitly.
- Non-significance does not demonstrate equivalence.
- Latency and resource findings are specific to the recorded local hardware and
  software environment.



## Data and security

- Never commit `.env`, tokens, credentials, confidential bank data, or unapproved policy
  documents.
- Keep evaluation-only labels out of LLM prompts and UI responses.
- Treat joblib files as trusted repository artifacts; never load untrusted serialized
  models.
- Verify dataset licences and redistribution rights before making the repository public.
- ULB is anonymised and temporally narrow; Sparkov is synthetic.
- Neither dataset establishes independent real-bank deployment performance.
- The local UI is research software and does not provide production authentication,
  authorization, monitoring, or availability guarantees.
