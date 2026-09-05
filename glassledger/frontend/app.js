/* GlassLedger workbench.
 *
 * Vanilla JS, no build step. The charts are hand-rolled inline SVG because there
 * are four of them and a charting library would be a bigger dependency than the
 * code it replaces.
 *
 * Two conventions worth stating:
 *
 * 1. Money never gets computed here. The backend sends integer paise plus a
 *    pre-formatted display string, and this file renders the string. A frontend
 *    that divides paise by 100 in JavaScript reintroduces float money at the last
 *    possible moment, which is the worst place to reintroduce it.
 *
 * 2. Colour is assigned by what a mark *is*, never by its rank in a sorted list.
 *    Bars keep their hue when a filter changes the ordering, so a reader tracking
 *    "the orange one" is not silently shown a different thing.
 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (p) => fetch(p).then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)));

const C = {
  s1: '#3987e5', s2: '#d95926', s3: '#199e70',
  good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b',
  text: '#c3c2b7', muted: '#8b8a80', grid: '#33322f', surface: '#1a1a19',
};

const svgEl = (n, attrs = {}) => {
  const e = document.createElementNS('http://www.w3.org/2000/svg', n);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

/* ---------- tooltip ---------- */
const tip = $('#tip');
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add('on');
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}
const hideTip = () => tip.classList.remove('on');

/* Every mark gets a hover layer. An SVG chart in a browser is interactive by
 * default; shipping one that isn't wastes the medium. */
function hoverable(el, html) {
  el.addEventListener('mousemove', e => showTip(e, html));
  el.addEventListener('mouseleave', hideTip);
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ---------- horizontal bar chart ----------
 * Horizontal because the categories are long text labels. A vertical bar chart
 * with "component_assignment" on the x-axis needs rotated labels, and rotated
 * labels are a readability tax paid on every read to save one layout decision. */
function barChart(mount, rows, opts = {}) {
  mount.innerHTML = '';
  if (!rows.length) { mount.innerHTML = '<div class="empty">nothing to show</div>'; return; }

  const labelW = opts.labelW ?? 186;
  const rowH = 27, gap = 7, padR = 66, padT = 4;
  const W = mount.clientWidth || 560;
  const H = padT + rows.length * (rowH + gap);
  const plotW = Math.max(60, W - labelW - padR);
  const max = Math.max(...rows.map(r => r.value), 1);

  const svg = svgEl('svg', { width: '100%', height: H, viewBox: `0 0 ${W} ${H}` });

  // Recessive gridlines: present enough to read a value against, quiet enough
  // that the bars stay the figure.
  for (let i = 0; i <= 4; i++) {
    const x = labelW + (plotW * i) / 4;
    svg.appendChild(svgEl('line', {
      x1: x, x2: x, y1: padT, y2: H - gap, class: 'gridline',
    }));
  }

  rows.forEach((r, i) => {
    const y = padT + i * (rowH + gap);
    const w = Math.max(2, (r.value / max) * plotW);
    const fill = r.color || C.s1;

    const label = svgEl('text', {
      x: labelW - 11, y: y + rowH / 2 + 4, 'text-anchor': 'end', class: 'mark-label',
    });
    label.textContent = r.label.length > 26 ? r.label.slice(0, 25) + '…' : r.label;
    svg.appendChild(label);

    // 4px rounded data-end, anchored to the baseline: the rounding lives at the
    // value end only, so the bar's start stays a hard reference line.
    const bar = svgEl('rect', {
      x: labelW, y, width: w, height: rowH, rx: 4, fill,
    });
    hoverable(bar, `<b>${esc(r.label)}</b><br>${esc(r.display ?? r.value)}` +
      (r.note ? `<br><span class="tk">${esc(r.note)}</span>` : ''));
    svg.appendChild(bar);

    // Direct label on every bar is fine here: at most ~8 rows, and it removes
    // the need to read a value off an axis.
    const val = svgEl('text', {
      x: labelW + w + 9, y: y + rowH / 2 + 4, class: 'mark-label',
    });
    val.textContent = r.display ?? r.value;
    svg.appendChild(val);
  });

  mount.appendChild(svg);
}

/* ---------- stacked proportion bar ---------- */
function stackBar(mount, segments, opts = {}) {
  mount.innerHTML = '';
  const W = mount.clientWidth || 560, H = 46, gapPx = 2;
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  const svg = svgEl('svg', { width: '100%', height: H + 34, viewBox: `0 0 ${W} ${H + 34}` });

  let x = 0;
  segments.forEach((s, i) => {
    const raw = (s.value / total) * W;
    // 2px surface-coloured gap between adjacent fills: reads as separation
    // without adding a stroke that would compete with the fill.
    const w = Math.max(0, raw - (i < segments.length - 1 ? gapPx : 0));
    if (w <= 0) { x += raw; return; }
    const rect = svgEl('rect', { x, y: 0, width: w, height: H, rx: 4, fill: s.color });
    hoverable(rect, `<b>${esc(s.label)}</b><br>${esc(s.display)}<br>` +
      `<span class="tk">${((s.value / total) * 100).toFixed(1)}% of total</span>`);
    svg.appendChild(rect);

    if (raw > 74) {
      const pct = svgEl('text', {
        x: x + w / 2, y: H / 2 + 4, 'text-anchor': 'middle',
        class: 'mark-label', fill: '#fff',
      });
      pct.textContent = ((s.value / total) * 100).toFixed(1) + '%';
      svg.appendChild(pct);
    }
    x += raw;
  });

  const leg = svgEl('g');
  svg.appendChild(leg);
  mount.appendChild(svg);

  // Legend widths are measured after the text is in the DOM, not estimated from
  // character count. The estimate was wrong for a monospace face at this size and
  // the labels collided -- and a legend that overlaps is worse than no legend,
  // because it makes two series look like one.
  let lx = 0;
  segments.forEach(s => {
    const sw = svgEl('rect', { x: lx, y: H + 15, width: 10, height: 10, rx: 3, fill: s.color });
    leg.appendChild(sw);
    const t = svgEl('text', { x: lx + 15, y: H + 24, class: 'mark-label' });
    t.textContent = `${s.label} — ${s.display}`;
    leg.appendChild(t);
    lx += 15 + t.getComputedTextLength() + 22;
  });
}

/* ---------- tiles ---------- */
function renderTiles(m) {
  const t = $('#tiles');
  const autoRate = m.matches_total
    ? (m.matches_confirmed / m.matches_total * 100).toFixed(1) : '0.0';
  const tiles = [
    { label: 'Auto-confirmed', value: m.matches_confirmed.toLocaleString(),
      sub: `<span class="accent">${autoRate}%</span> of all hypotheses · no human touched these` },
    { label: 'Awaiting a human', value: m.matches_pending_human.toLocaleString(),
      sub: `<span class="accent">${m.pending_human.display}</span> held for approval` },
    { label: 'Open exceptions', value: m.exceptions_open.toLocaleString(),
      sub: `<span class="accent">${m.exception_amount.display}</span> unexplained` },
    { label: 'Bank leg reconciled', value: m.bank_leg_reconciled.display,
      sub: `${m.journal_entries} balanced journal entries posted` },
    { label: 'Events in the log', value: m.events_applied.toLocaleString(),
      sub: `${m.transactions.toLocaleString()} transactions ingested` },
  ];
  t.innerHTML = tiles.map(x =>
    `<div class="tile"><div class="label">${x.label}</div>
     <div class="value">${x.value}</div><div class="sub">${x.sub}</div></div>`).join('');
}

function renderIntegrity(m) {
  const i = m.integrity;
  const chip = (ok, label, detail) =>
    `<span class="chip" title="${esc(detail)}"><span class="dot${ok ? '' : ' bad'}"></span>${label}</span>`;
  $('#integrity').innerHTML =
    chip(i.chain_ok, 'hash chain', i.chain_detail) +
    chip(i.ledger_balanced, 'ledger balanced', `drift ${i.ledger_drift_paise} paise`) +
    `<span class="chip" title="rebuilt twice from scratch; identical">replay ${i.replay_fingerprint.slice(0, 10)}…</span>`;
}

function renderLedger(m) {
  const rows = Object.entries(m.ledger_display);
  $('#ledger').innerHTML =
    `<table class="detail"><thead><tr><th>Account</th><th style="text-align:right">Balance</th></tr></thead><tbody>` +
    rows.map(([a, v]) => `<tr><td>${esc(a)}</td><td class="num">${esc(v)}</td></tr>`).join('') +
    `<tr><td style="color:var(--text-primary)">Net (must be zero)</td>` +
    `<td class="num" style="color:${m.integrity.ledger_balanced ? C.good : C.critical}">` +
    `${m.integrity.ledger_drift_paise} paise</td></tr>` +
    `</tbody></table>`;
}

/* ---------- dashboard ---------- */
let METRICS = null;

async function loadDash() {
  const m = await api('/api/metrics');
  METRICS = m;
  renderTiles(m);
  renderIntegrity(m);
  renderLedger(m);

  // Tier 1 vs tier 2 is a real distinction (identity vs inference), so the two
  // groups get different hues. Within a group every bar shares a hue: they are
  // the same kind of thing, and varying colour would imply a difference that
  // is not there.
  const t1 = new Set(['exact_utr_and_amount', 'exact_utr_split_sum', 'order_key_join',
                      'wash_pair_netting']);
  const algo = Object.entries(m.confirmed_by_algorithm).map(([k, v]) => ({
    label: k, value: v, display: v.toLocaleString(),
    color: t1.has(k) ? C.s1 : C.s3,
    note: t1.has(k) ? 'tier 1 — deterministic identity'
                    : 'tier 2 — optimisation + calibrated score',
  }));
  barChart($('#chart-algo'), algo);
  $('#chart-algo').insertAdjacentHTML('afterend',
    `<div class="legend">
       <span class="item"><span class="swatch" style="background:${C.s1}"></span>Tier 1 — deterministic</span>
       <span class="item"><span class="swatch" style="background:${C.s3}"></span>Tier 2 — optimisation</span>
     </div>`);

  const cats = Object.entries(m.open_exceptions_by_category).map(([k, v]) => ({
    label: k, value: v, display: v.toLocaleString(), color: C.s2,
  }));
  barChart($('#chart-cat'), cats, { labelW: 200 });

  stackBar($('#chart-money'), [
    { label: 'Reconciled', value: m.bank_leg_reconciled_paise,
      display: m.bank_leg_reconciled.display, color: C.s3 },
    { label: 'Awaiting approval', value: m.pending_human_paise,
      display: m.pending_human.display, color: C.s1 },
    { label: 'Exceptions', value: m.exception_paise,
      display: m.exception_amount.display, color: C.s2 },
  ]);
}

/* ---------- exceptions ---------- */
let EXC = [];

/* Severity, derived from money and age.
 *
 * The age term is measured against the *batch*, not the calendar. An earlier
 * version used absolute thresholds (>14 days = critical) and every item in a
 * month-long batch came out critical, which is the same as having no severity at
 * all -- a queue where everything is urgent tells a reviewer nothing. The age
 * component now keys on how old an item is relative to the oldest thing in the
 * queue, so severity ranks within the batch it is describing.
 *
 * Never hue alone: each badge carries a glyph and a word.
 */
function sevOf(e, maxAge) {
  const oldish = maxAge > 0 && e.age_days >= maxAge * 0.75;
  if (e.category === 'reversal_wash_pair') return ['good', '✓ self-resolving'];
  if (e.amount_paise >= 5000000) return ['critical', '● critical'];
  if (e.amount_paise >= 500000 || (oldish && e.amount_paise >= 100000))
    return ['serious', '◆ serious'];
  return ['warning', '▲ review'];
}

function renderExceptions() {
  const cat = $('#f-cat').value, q = $('#f-q').value.toLowerCase(), sort = $('#f-sort').value;
  let rows = EXC.filter(e =>
    (!cat || e.category === cat) &&
    (!q || e.txn_ids.join(' ').toLowerCase().includes(q) ||
      JSON.stringify(e.evidence).toLowerCase().includes(q)));

  rows.sort((a, b) => sort === 'amount' ? b.amount_paise - a.amount_paise
    : sort === 'age' ? b.age_days - a.age_days
    : b.priority - a.priority);

  const shownPaise = rows.reduce((s, e) => s + Math.abs(e.amount_paise), 0);
  $('#exc-count').textContent =
    `${rows.length} of ${EXC.length} shown · ` +
    `₹${(shownPaise / 100).toLocaleString('en-IN', { minimumFractionDigits: 2 })} outstanding`;

  const maxAge = Math.max(0, ...EXC.map(e => e.age_days));
  $('#exc-list').innerHTML = rows.length ? rows.map(e => {
    const [sev, sevLabel] = sevOf(e, maxAge);
    const dets = (e.evidence.txn_details || []);
    return `<div class="exc" data-id="${esc(e.exception_id)}">
      <div class="exc-head">
        <div class="prio">${e.priority.toFixed(1)}</div>
        <div><div class="cat">${esc(e.category.replace(/_/g, ' '))}</div>
             <div class="ids">${esc(e.txn_ids.join(', '))}</div></div>
        <div><span class="badge ${sev}">${sevLabel}</span></div>
        <div class="age">${e.age_days}d</div>
        <div class="amt">${esc(e.amount.display)}</div>
      </div>
      <div class="exc-body">
        <div class="suggest">${esc(e.suggested_action)}</div>
        ${dets.length ? `<table class="detail">
          <thead><tr><th>Transaction</th><th>Value date</th><th>Narration</th>
          <th>References found</th><th style="text-align:right">Amount</th></tr></thead>
          <tbody>${dets.map(d => `<tr>
            <td>${esc(d.txn_id)}</td><td>${esc(d.value_date)}</td>
            <td>${esc((d.narration || '').slice(0, 46))}</td>
            <td>${esc((d.ref_candidates || []).slice(0, 2).join(', ') || '—')}</td>
            <td class="num">${(d.amount_paise / 100).toLocaleString('en-IN',
              { minimumFractionDigits: 2 })}</td></tr>`).join('')}</tbody></table>` : ''}
        <div class="dim mono" style="font-size:11px;margin-top:11px">
          best hypothesis scored ${e.max_confidence.toFixed(4)} · leg ${esc(e.leg)}
          · events ${e.history.join(', ')}
        </div>
        <div class="actions">
          <button class="act primary" data-res="accept">Accept the suggestion</button>
          <button class="act" data-res="no_action">No action needed</button>
          <button class="act" data-res="escalate">Escalate</button>
          <button class="act" data-audit="${esc(e.txn_ids[0])}">Open audit trail</button>
        </div>
      </div>
    </div>`;
  }).join('') : '<div class="empty">no exceptions match this filter</div>';
}

async function loadExceptions() {
  const d = await api('/api/exceptions?status=open');
  EXC = d.items;
  const cats = [...new Set(EXC.map(e => e.category))].sort();
  $('#f-cat').innerHTML = '<option value="">All categories</option>' +
    cats.map(c => `<option value="${esc(c)}">${esc(c.replace(/_/g, ' '))}</option>`).join('');
  renderExceptions();
}

/* ---------- review queue ---------- */
async function loadReview() {
  const d = await api('/api/matches?status=proposed');
  $('#review-list').innerHTML = d.items.length ? d.items.map(m => {
    const gated = m.gate === 'materiality';
    // Only the [0,1]-bounded features get a bar. date_delta_days is in days and
    // cycle_prior_logit is a log-odds -- drawing either on a 0-100% track shows a
    // full bar for the value 2, which reads as "maximum" when it means "two days".
    // A bar whose length does not mean magnitude is worse than no bar.
    const BOUNDED = new Set(['utr_exact', 'utr_similarity', 'utr_in_narration',
      'amount_exact', 'amount_rel_delta', 'within_fee_band', 'date_delta_abs',
      'narration_cosine', 'currency_match', 'amount_uniqueness', 'is_subset',
      'left_ambiguity', 'right_ambiguity', 'subset_size']);
    const feats = Object.entries(m.features || {})
      .filter(([k]) => BOUNDED.has(k))
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 7);
    return `<div class="card">
      <div style="display:flex;justify-content:space-between;gap:16px;align-items:baseline">
        <div>
          <h2>${esc(m.algorithm)} <span class="dim">· tier ${m.tier} · ${esc(m.cardinality)}</span></h2>
          <p class="hint mono" style="margin-bottom:8px">${esc(m.left_ids.join(', '))}
             → ${esc(m.right_ids.join(', '))}</p>
        </div>
        <div style="text-align:right">
          <div class="mono num" style="font-size:19px">${esc(m.exposure.display)}</div>
          <div class="dim mono" style="font-size:11.5px">confidence ${m.confidence.toFixed(4)}</div>
        </div>
      </div>

      ${gated ? `<div class="gate-note"><span class="glyph">■</span><div>
        <b>Materiality gate — held for human approval.</b>
        Exposure ${esc(m.exposure.display)} is above the ₹50,000 line, so this
        cannot auto-confirm even at confidence ${m.confidence.toFixed(4)}.
        The rule sits outside every model: no score, from any tier, overrides it.
        </div></div>` : ''}

      <div style="margin:12px 0">
        <div class="dim" style="font-size:11.5px;margin-bottom:7px">
          ${esc(m.left_display)} vs ${esc(m.right_display)} · residual
          ${esc(m.residual_display)} · ${esc((m.evidence || {}).rule || '')}
        </div>
        ${feats.map(([k, v]) => `<div class="featbar">
          <span class="fname">${esc(k)}</span>
          <span class="track"><span class="fill" style="width:${Math.min(100, Math.abs(v) * 100)}%"></span></span>
          <span class="fval">${v.toFixed(3)}</span></div>`).join('')}
      </div>

      ${(m.runners_up || []).length ? `<div class="dim mono" style="font-size:11.5px">
        also considered: ${m.runners_up.slice(0, 2).map(r =>
          `${esc(r.right_ids.join(','))} @ ${r.score.toFixed(3)}`).join(' · ')}</div>` : ''}

      <div class="actions">
        <button class="act primary" data-match="${esc(m.match_key)}" data-act="confirm">Confirm</button>
        <button class="act" data-match="${esc(m.match_key)}" data-act="reject">Reject</button>
        <button class="act" data-audit="${esc(m.left_ids[0] || '')}">Audit trail</button>
      </div>
    </div>`;
  }).join('') : '<div class="empty">nothing awaiting approval</div>';
}

/* ---------- audit ---------- */
const EVENT_CLASS = {
  MatchConfirmed: 'confirm', ExceptionRaised: 'exception',
  JournalEntryPosted: 'journal',
};

async function loadAudit(txnId) {
  const out = $('#audit-out');
  if (!txnId) { out.innerHTML = ''; return; }
  $('#audit-q').value = txnId;
  out.innerHTML = '<div class="empty">replaying…</div>';
  let d;
  try { d = await api('/api/audit/' + encodeURIComponent(txnId)); }
  catch (e) { out.innerHTML = `<div class="err">${esc(e.detail || 'not found')}</div>`; return; }

  $('#audit-hint').textContent =
    `${d.events} events replayed from the append-only log for ${txnId}`;

  out.innerHTML = `<div class="card"><div class="trail">` + d.trail.map(ev => {
    const p = ev.payload;
    let body = '';
    if (ev.event_type === 'TransactionIngested') {
      body = `<div class="kv"><b>${esc(p.source)}</b> ${(p.amount_paise / 100).toLocaleString('en-IN',
        { minimumFractionDigits: 2 })} on ${esc(p.value_date)}<br>
        reference candidates: ${esc((p.ref_candidates || []).join(', ') || 'none')}<br>
        narration: ${esc((p.narration || '').slice(0, 74))}<br>
        <span class="tk">from ${esc((p.provenance || {}).source_file || '?')}
        row ${esc((p.provenance || {}).row ?? '?')}</span></div>`;
    } else if (ev.event_type === 'MatchCandidateProposed') {
      const f = Object.entries(p.features || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
      body = `<div class="kv"><b>${esc(p.algorithm)}</b> (tier ${p.tier}) scored
        <b>${p.confidence.toFixed(4)}</b><br>
        ${esc((p.evidence || {}).rule || '')}<br>
        residual ${(p.residual_paise / 100).toFixed(2)}<br>
        <span class="tk">top features: ${f.map(([k, v]) => `${k}=${v.toFixed(3)}`).join(' · ')}</span>
        ${(p.runners_up || []).length ? `<br><span class="tk">rejected alternatives:
          ${p.runners_up.slice(0, 2).map(r => `${esc(r.right_ids.join(','))} @ ${r.score.toFixed(3)}`).join(' · ')}</span>` : ''}
        </div>`;
    } else if (ev.event_type === 'MatchConfirmed') {
      body = `<div class="kv">confirmed by <b>${esc(p.confirmed_by)}</b>
        via gate <b>${esc(p.gate)}</b><br>${esc(p.rationale || '')}</div>`;
    } else if (ev.event_type === 'JournalEntryPosted') {
      body = `<div class="kv">${(p.legs || []).map(l =>
        `${l.debit_paise ? 'Dr' : 'Cr'} ${esc(l.account)} —
         <b>${((l.debit_paise || l.credit_paise) / 100).toLocaleString('en-IN',
           { minimumFractionDigits: 2 })}</b>`).join('<br>')}</div>`;
    } else if (ev.event_type === 'ExceptionRaised') {
      body = `<div class="kv"><b>${esc(p.category)}</b><br>${esc(p.suggested_action)}</div>`;
    } else {
      body = `<div class="kv">${esc(JSON.stringify(p).slice(0, 220))}</div>`;
    }
    return `<div class="tevent ${EVENT_CLASS[ev.event_type] || ''}">
      <div class="thead">${esc(ev.event_type)}
        <span class="tseq">seq ${ev.seq} · ${esc(ev.occurred_at)} · hash ${esc(ev.hash.slice(0, 10))}…</span>
      </div>
      <div class="tbody">${body}</div></div>`;
  }).join('') + `</div></div>`;
}

/* ---------- events ---------- */
document.addEventListener('click', async (e) => {
  const tab = e.target.closest('nav.tabs button');
  if (tab) {
    $$('nav.tabs button').forEach(b => b.setAttribute('aria-selected', String(b === tab)));
    $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab.dataset.panel));
    if (tab.dataset.panel === 'exceptions' && !EXC.length) loadExceptions();
    if (tab.dataset.panel === 'review') loadReview();
    return;
  }

  const head = e.target.closest('.exc-head');
  if (head) { head.parentElement.classList.toggle('open'); return; }

  const auditBtn = e.target.closest('[data-audit]');
  if (auditBtn) {
    $$('nav.tabs button').forEach(b =>
      b.setAttribute('aria-selected', String(b.dataset.panel === 'audit')));
    $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-audit'));
    loadAudit(auditBtn.dataset.audit);
    return;
  }

  const res = e.target.closest('[data-res]');
  if (res) {
    const id = res.closest('.exc').dataset.id;
    res.disabled = true;
    await fetch(`/api/exceptions/${encodeURIComponent(id)}/resolve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resolution: res.dataset.res, resolved_by: 'workbench' }),
    });
    await loadExceptions();
    loadDash();
    return;
  }

  const mact = e.target.closest('[data-match]');
  if (mact) {
    mact.disabled = true;
    await fetch(`/api/matches/${encodeURIComponent(mact.dataset.match)}/action`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: mact.dataset.act, actor: 'workbench' }),
    });
    await loadReview();
    loadDash();
  }
});

$('#audit-go').addEventListener('click', () => loadAudit($('#audit-q').value.trim()));
$('#audit-q').addEventListener('keydown', e => {
  if (e.key === 'Enter') loadAudit($('#audit-q').value.trim());
});
['#f-cat', '#f-sort', '#f-q'].forEach(s =>
  $(s).addEventListener('input', renderExceptions));

addEventListener('resize', () => { if (METRICS) loadDash(); });
loadDash().catch(e => {
  $('#tiles').innerHTML = `<div class="err">could not load: ${esc(e.detail || e.message)}</div>`;
});
