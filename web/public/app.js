const FAMILY = "AUTO-ABC-20260829-01";
const state = { readiness: null, workflows: [], studies: null, eda: null, samples: [], dataset: "ULB", selectedJob: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value = "") => String(value).replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
const fmt = (value, digits = 2) => value === null || value === undefined || value === "" ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const pct = (value, digits = 1) => value === null || value === undefined ? "—" : `${fmt(Number(value) * 100, digits)}%`;

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {"content-type":"application/json"}, ...options});
  const payload = await response.json().catch(() => ({detail: response.statusText}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail));
  return payload;
}
function toast(text) { const node = $("#toast"); node.textContent = text; node.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 4200); }
function showError(error) { toast(error?.message || String(error)); }
function kpi(label, value, note = "") { return `<article class="kpi"><small>${esc(label)}</small><strong>${esc(value)}</strong><span>${esc(note)}</span></article>`; }
function showScreen(id) {
  $$(".screen").forEach((node) => node.classList.toggle("active", node.id === id));
  $$(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.screen === id));
  const titles = {command:"Command Centre",workflow:"Workflow Runner",eda:"EDA & Models",experiment:"Experiment Lab",alerts:"Alerts & Jira",evidence:"Evidence & Reports",system:"System & Safety"};
  $("#screen-title").textContent = titles[id]; history.replaceState(null, "", `#${id}`);
  if (id === "workflow") loadJobs(); if (id === "alerts") loadJira(); if (id === "evidence") loadEvidence();
}
$$('.nav-item').forEach((button) => button.addEventListener('click', () => showScreen(button.dataset.screen)));
$$('.navigate').forEach((button) => button.addEventListener('click', () => showScreen(button.dataset.target)));

function svgBars(items, options = {}) {
  if (!items.length) return `<p class="note">No measured values available.</p>`;
  const width = 640, row = 44, height = items.length * row + 24, left = 150, right = 82;
  const max = options.max ?? Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label || "bar chart")}">${items.map((item, i) => {
    const y = i * row + 12, value = Number(item.value) || 0, bar = Math.max(value > 0 ? 2 : 0, (width-left-right) * value / max);
    return `<text x="0" y="${y+16}" fill="#526d78" font-size="12">${esc(item.label)}</text><rect x="${left}" y="${y}" width="${width-left-right}" height="20" rx="6" fill="#edf3f2"/><rect x="${left}" y="${y}" width="${bar}" height="20" rx="6" fill="${item.color || '#2e968d'}"/><text x="${width-4}" y="${y+15}" text-anchor="end" fill="#183246" font-size="12" font-weight="700">${esc(item.display ?? fmt(value, 3))}</text>`;
  }).join("")}</svg>`;
}

function renderReadiness() {
  const checks = state.readiness?.checks || {};
  $("#readiness-ribbon").innerHTML = Object.entries(checks).map(([key, value]) => `<span class="${value.available ? 'ready' : 'missing'}">${value.available ? '●' : '○'} ${esc(key.replaceAll('_',' '))}</span>`).join("") + `<span class="${state.readiness?.jira?.configured ? 'ready':'missing'}">${state.readiness?.jira?.configured ? '●':'○'} Jira configured</span>`;
  const instances = state.studies?.instances || [];
  $("#overview-cards").innerHTML = [
    kpi("Legacy scaffold calls", "180", "not definitive thesis evidence"),
    kpi("Registered instances", String(instances.length), "original plus reservations"),
    kpi("Verified replications", String(state.studies?.verified_replication_count || 0), "reported separately"),
    kpi("Jira project", state.readiness?.jira?.project_key || "Not configured", "credentials remain server-side"),
  ].join("");
  $("#system-grid").innerHTML = [
    ["API evidence readiness", state.readiness?.status || "unknown", "Required local artifacts"],
    ["Jira", state.readiness?.jira?.configured ? "Configured" : "Unavailable", state.readiness?.jira?.base_url_host || "No host exposed"],
    ["Credential exposure", "Blocked", "Token and email omitted"],
    ["Arbitrary commands", "Blocked", "Allowlisted jobs only"],
    ["Experiment → Jira", "Blocked", "experiment_id exclusion"],
    ["Cross-study pooling", "Blocked", "descriptive comparison only"],
  ].map(([a,b,c]) => `<article><small>${esc(a)}</small><strong>${esc(b)}</strong><small>${esc(c)}</small></article>`).join("");
}

function renderWorkflows() {
  $("#workflow-cards").innerHTML = state.workflows.map((spec) => `<article><h3>${esc(spec.title)}</h3><p>${esc(spec.purpose)}</p><footer><small>${spec.compute_heavy ? 'Compute-heavy' : 'Light'} · ${spec.external_write ? 'external write' : 'local only'}</small><button class="secondary job-start" data-job="${esc(spec.job_type)}">Run</button></footer></article>`).join("");
}
async function startJob(jobType) {
  try {
    const job = await api('/api/v1/jobs', {method:'POST', body:JSON.stringify({job_type:jobType, parameters:{}})});
    toast(`${job.title} queued as ${job.job_id.slice(0,8)}`); state.selectedJob = job.job_id; showScreen('workflow'); await loadJobs();
  } catch (error) { showError(error); }
}
document.addEventListener('click', (event) => { const button = event.target.closest('.job-start'); if (button) startJob(button.dataset.job); });

async function loadJobs() {
  try {
    const payload = await api('/api/v1/jobs');
    $("#job-list").innerHTML = payload.jobs.length ? payload.jobs.map((job) => `<article class="job-row" data-job-id="${esc(job.job_id)}"><div><strong>${esc(job.title)}</strong><small>${esc(job.job_id.slice(0,8))} · ${esc(job.created_at_utc)}</small></div><span class="status ${esc(job.status)}">${esc(job.status)}</span></article>`).join('') : '<p class="note">No UI jobs have run yet.</p>';
    $$('.job-row').forEach((row) => row.addEventListener('click', () => loadJobLog(row.dataset.jobId)));
    if (state.selectedJob) await loadJobLog(state.selectedJob);
    if (payload.jobs.some((job) => !job.terminal)) setTimeout(() => document.querySelector('#workflow.active') && loadJobs(), 1800);
  } catch (error) { showError(error); }
}
async function loadJobLog(jobId) { state.selectedJob = jobId; try { const payload = await api(`/api/v1/jobs/${jobId}/log`); $('#job-log').textContent = payload.lines.join('\n') || 'Job has not emitted output yet.'; } catch(error) { showError(error); } }

function profileFact(profile, topic) { return profile?.records?.find((row) => row.topic === topic)?.facts; }
function renderEda() {
  const data = state.eda?.datasets?.find((item) => item.dataset_id === state.dataset); if (!data) return;
  const prevalence = profileFact(data.profile, 'historical_prevalence') || {};
  const tuned = data.test_metrics.find((row) => row.threshold_rule === 'validation_tuned') || data.test_metrics[0] || {};
  $('#eda-kpis').innerHTML = [kpi('Training transactions', fmt(prevalence.transactions,0), data.profile?.scope || ''),kpi('Observed fraud',fmt(prevalence.fraud,0),pct(prevalence.fraud_rate,3)),kpi('Test average precision',fmt(tuned.average_precision,3),'selected model row'),kpi('Test recall',pct(tuned.recall,1),`precision ${pct(tuned.precision,1)}`)].join('');
  $('#class-chart').innerHTML = svgBars([{label:'Legitimate',value:prevalence.legitimate || 0,display:fmt(prevalence.legitimate,0),color:'#83bcb6'},{label:'Fraud label',value:prevalence.fraud || 0,display:fmt(prevalence.fraud,0),color:'#c85c63'}],{label:`${state.dataset} class counts`});
  $('#model-chart').innerHTML = svgBars(['average_precision','roc_auc','precision','recall','f1','mcc'].map((key) => ({label:key.replaceAll('_',' '),value:tuned[key],display:fmt(tuned[key],3)})),{max:1,label:`${state.dataset} test metrics`});
  const cols = ['model','threshold_rule','threshold','average_precision','roc_auc','precision','recall','f1','mcc'];
  $('#model-table').innerHTML = `<table><thead><tr>${cols.map((c)=>`<th>${esc(c.replaceAll('_',' '))}</th>`).join('')}</tr></thead><tbody>${data.test_metrics.map((row)=>`<tr>${cols.map((c)=>`<td>${typeof row[c] === 'number' ? fmt(row[c],4) : esc(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('#run-model-build').dataset.job = state.dataset === 'ULB' ? 'ULB_NOTEBOOK_BUILD' : 'SPARKOV_NOTEBOOK_BUILD';
}
$$('.dataset-tab').forEach((button) => button.addEventListener('click', () => { state.dataset=button.dataset.dataset; $$('.dataset-tab').forEach((b)=>b.classList.toggle('active',b===button)); renderEda(); }));
$('#run-model-build').addEventListener('click', () => startJob($('#run-model-build').dataset.job));

function populateInstances() {
  const entries = state.studies?.instances || [];
  const options = entries.map((item) => `<option value="${esc(item.instance_id)}">${esc(item.instance_id)} · ${esc(item.status)}</option>`).join('');
  ['#global-instance','#instance-one','#instance-two'].forEach((id) => $(id).innerHTML = options);
  $('#instance-two').value = entries[1]?.instance_id || entries[0]?.instance_id || 'ORIGINAL';
}
async function loadInstance(id = 'ORIGINAL') {
  try {
    const payload = await api(`/api/v1/studies/${FAMILY}/instances/${encodeURIComponent(id)}`), record = payload.record, result = payload.result;
    $('#study-status').textContent = `${record.instance_id} · ${record.role} · ${record.status} · ${record.expected_calls} expected calls · primary immutable: ${record.immutable}`;
    if (!result) { $('#experiment-results').innerHTML = `<div class="panel"><h3>${esc(record.instance_id)} is reserved</h3><p class="note">No generation/result artifact has been registered. This planned entry cannot feed thesis conclusions.</p></div>`; return; }
    const datasets = result.dataset_results || {};
    $('#experiment-results').innerHTML = Object.entries(datasets).map(([dataset, value]) => `<div><h3>${esc(dataset)} · ${value.run_count} calls</h3><div class="experiment-grid">${Object.entries(value.configuration_results || {}).map(([config,row]) => { const m=row.automatic_metric_means||{}; return `<article class="metric-card"><h3>${esc(config)}</h3><dl><div><dt>Completed</dt><dd>${fmt(row.completed_count,0)}/${fmt(row.run_count,0)}</dd></div><div><dt>Explanation fidelity</dt><dd>${fmt(m.explanation_fidelity,3)}</dd></div><div><dt>JSON valid</dt><dd>${pct(m.response_json_valid,1)}</dd></div><div><dt>Citation coverage</dt><dd>${pct(m.claim_citation_coverage,1)}</dd></div><div><dt>Referral</dt><dd>${pct(m.manual_review_referral,1)}</dd></div></dl></article>`; }).join('')}</div></div>`).join('');
  } catch(error) { showError(error); }
}
$('#instance-one').addEventListener('change', (e)=>{ $('#global-instance').value=e.target.value; loadInstance(e.target.value); });
$('#global-instance').addEventListener('change', (e)=>{ $('#instance-one').value=e.target.value; loadInstance(e.target.value); });
$('#compare-instances').addEventListener('click', async()=>{ try { const p=await api(`/api/v1/studies/${FAMILY}/compare?first=${encodeURIComponent($('#instance-one').value)}&second=${encodeURIComponent($('#instance-two').value)}`); $('#study-status').textContent=`Compatibility: ${p.strictly_compatible ? 'strict protocol match':'not compatible'} · ${p.metric_differences.length} metric differences · pooling performed: no`; if(!p.metric_differences.length) toast('One or both instances has no verified result yet.'); } catch(e){showError(e);} });
async function replication(action) { try { const count=Number($('#replication-count').value); const body=action==='queue'?{count,confirm_directory_creation:true}:{count}; const payload=await api(`/api/v1/studies/${FAMILY}/replications/${action}`,{method:'POST',body:JSON.stringify(body)}); $('#replication-preview').textContent=`${payload.requested_replications} replication(s) · ${payload.total_new_model_calls} planned calls · IDs ${payload.planned_instance_ids.join(', ')} · Qwen generation started: no`; if(action==='queue'){toast('Replication provenance directories reserved; no model calls started.'); await loadStudies();} }catch(e){showError(e);} }
$('#preview-replications').addEventListener('click',()=>replication('preview')); $('#reserve-replications').addEventListener('click',()=>replication('queue'));

function selectedSampleIds(){return $$('#alert-samples input:checked').map((node)=>node.value).slice(0,10)}
async function alertAction(submit=false){const sample_ids=selectedSampleIds();if(!sample_ids.length){toast('Select at least one sample.');return}if(submit&&!confirm(`Process ${sample_ids.length} alert(s) through Configuration C and allow eligible Jira referrals?`))return;try{const body={sample_ids,engine:'qwen',confirm_write:submit};const p=await api(`/api/v1/operational-alerts/${submit?'batch':'preview'}`,{method:'POST',body:JSON.stringify(body)});$('#alert-message').textContent=submit?`${p.processed_count} alert(s) processed. Review Jira for created or reused Stories.`:`${p.requested_count} unique alert(s); maximum ${p.maximum_possible_jira_writes} Jira write(s); none performed.`;if(submit)loadJira()}catch(e){showError(e)}}
$('#preview-alerts').addEventListener('click',()=>alertAction(false));$('#submit-alerts').addEventListener('click',()=>alertAction(true));
async function loadJira(){try{const p=await api('/api/v1/jira/analytics');if(p.status!=='available'){const missing=String(p.reason||'').includes('Missing Jira environment variables');$('#jira-kpis').innerHTML=kpi('Jira analytics','Unavailable',missing?'Restart via run_node_ui.sh and complete the secure token prompt.':(p.reason||'Check local API environment'));$('#jira-status-chart').innerHTML=`<p class="note">${missing?'The API was started without Jira credentials. Stop both local servers, rerun ./run_node_ui.sh, enter the token once when prompted, then refresh Jira.':'No live Jira snapshot available.'}</p>`;$('#jira-duration-chart').innerHTML='<p class="note">No lifecycle durations are calculated until Jira connects.</p>';$('#jira-table').innerHTML='';return }$('#jira-kpis').innerHTML=[kpi('Open cases',fmt(p.open_count,0),'current workload'),kpi('New cases',fmt(p.new_last_7_days_count,0),'created in last 7 days'),kpi('Closed',fmt(p.closed_count,0),'Done/resolved'),kpi('Mean resolution',p.mean_time_to_resolution_hours==null?'—':`${fmt(p.mean_time_to_resolution_hours,1)} h`,`${p.resolved_duration_sample_size} resolved case(s)`)].join('');$('#jira-status-chart').innerHTML=svgBars(Object.entries(p.status_counts).map(([label,value])=>({label,value,display:fmt(value,0)})),{label:'Jira status counts'});$('#jira-duration-chart').innerHTML=svgBars([{label:'Mean MTTR',value:p.mean_time_to_resolution_hours||0,display:p.mean_time_to_resolution_hours==null?'—':`${fmt(p.mean_time_to_resolution_hours,1)} h`},{label:'Median MTTR',value:p.median_time_to_resolution_hours||0,display:p.median_time_to_resolution_hours==null?'—':`${fmt(p.median_time_to_resolution_hours,1)} h`},{label:'Stale open',value:p.stale_open_count||0,display:fmt(p.stale_open_count,0),color:'#d6922d'}],{label:'Jira durations and aging'});$('#jira-table').innerHTML=`<table><thead><tr><th>Key</th><th>Summary</th><th>Status</th><th>Open age</th><th>Stale</th><th>Reopens</th></tr></thead><tbody>${p.issues.map(i=>`<tr><td>${esc(i.key)}</td><td>${esc(i.summary)}</td><td>${esc(i.status)}</td><td>${i.open_age_hours==null?'—':`${fmt(i.open_age_hours,1)} h`}</td><td>${i.stale?'Yes':'No'}</td><td>${fmt(i.reopen_count,0)}</td></tr>`).join('')}</tbody></table>`}catch(e){showError(e)}}
$('#refresh-jira').addEventListener('click',loadJira);

async function loadEvidence(){try{const p=await api('/api/v1/evidence');$('#evidence-list').innerHTML=p.entries.map((e)=>`<article class="evidence-row"><strong>${esc(e.name)}</strong><code>${esc(e.path)}</code><span class="tag">${esc(e.classification)}</span><span class="${e.available?'check':'missing'}">${e.available?'Available':'Missing'}</span></article>`).join('')}catch(e){showError(e)}}
async function loadStudies(){state.studies=await api(`/api/v1/studies/${FAMILY}/instances`);populateInstances();renderReadiness();await loadInstance($('#instance-one').value||'ORIGINAL')}
async function initialize(){try{const [ready,workflows,studies,eda,samples]=await Promise.all([api('/api/v1/readiness'),api('/api/v1/workflows'),api(`/api/v1/studies/${FAMILY}/instances`),api('/api/v1/eda-models'),api('/api/v1/samples')]);Object.assign(state,{readiness:ready,workflows:workflows.job_types,studies,eda,samples:samples.samples});$('#api-dot').classList.add('online');$('#api-status').textContent='Local API ready';renderReadiness();renderWorkflows();populateInstances();renderEda();$('#alert-samples').innerHTML=state.samples.map((s)=>`<label><input type="checkbox" value="${esc(s.sample_id)}"/>${esc(s.sample_id)} · ${esc(s.dataset_id||s.dataset||'')}</label>`).join('');await loadInstance('ORIGINAL');const route=location.hash.slice(1);if($(`#${route}`))showScreen(route)}catch(e){$('#api-status').textContent='API unavailable';showError(e)}}
$('#refresh-jobs').addEventListener('click',loadJobs); initialize();
