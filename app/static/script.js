// ── Metrics ────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const d = await res.json();
        animateValue("activePrompts", d.active_prompts);
        document.getElementById("latestScore").innerText = d.latest_score;
        document.getElementById("modelVersion").innerText = d.model_version;
    } catch {
        ["activePrompts","latestScore","modelVersion"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = "Err";
        });
    }
}

function animateValue(id, end, dur = 900) {
    const el = document.getElementById(id);
    if (!el) return;
    let t0 = null;
    const step = ts => {
        if (!t0) t0 = ts;
        const p = Math.min((ts - t0) / dur, 1);
        el.textContent = Math.floor(p * end);
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

// ── Textarea auto-resize ───────────────────────
function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

// ── Model change ───────────────────────────────
function onModelChange(select) {
    select.style.color = select.value ? '#ccc' : '#555';
}

// ── Sample dialogue ────────────────────────────
function fillSample() {
    const ta = document.getElementById("dialogueInput");
    if (!ta) return;
    ta.value = "Alice: Hey Bob, did you finish the machine learning assignment?\nBob: Not yet. I was working on the transformer model. It took longer than expected.\nAlice: Same here. I struggled with self-attention and multi-head attention.\nBob: Once you get queries, keys, and values it becomes easier.\nAlice: True. I also watched some tutorials online.\nBob: Great. Let's complete it tonight and submit tomorrow.";
    autoResize(ta);
}

function clearAll() {
    const ta = document.getElementById("dialogueInput");
    if (ta) { ta.value = ''; ta.style.height = 'auto'; }
    const o = document.getElementById("summaryOutput");
    if (o) o.innerHTML = '<span class="placeholder-text">Run the model to see results...</span>';
    const tag = document.getElementById("outputModelTag");
    if (tag) tag.textContent = '';
}

// ── Summarize ──────────────────────────────────
async function testSummarize() {
    const dialogue = (document.getElementById("dialogueInput") || {}).value || '';
    const outputEl  = document.getElementById("summaryOutput");
    const tagEl     = document.getElementById("outputModelTag");
    const btn       = document.getElementById("summarizeBtn");
    const icon      = document.getElementById("btnIcon");
    const spinner   = document.getElementById("btnSpinner");
    const select    = document.getElementById("modelSelect");
    const model     = select ? select.value : "custom";

    if (!dialogue.trim()) {
        if (outputEl) outputEl.innerHTML = '<span style="color:#ef4444"><i class="fa-solid fa-triangle-exclamation"></i> Please enter a dialogue first.</span>';
        if (document.getElementById("dialogueInput")) document.getElementById("dialogueInput").focus();
        return;
    }
    if (!model) {
        if (outputEl) outputEl.innerHTML = '<span style="color:#ef4444"><i class="fa-solid fa-triangle-exclamation"></i> Please select a model first.</span>';
        if (select) select.focus();
        return;
    }

    btn.disabled = true;
    if (icon) icon.style.display = 'none';
    if (spinner) spinner.style.display = 'block';
    if (outputEl) outputEl.innerHTML = '<span class="placeholder-text"><i class="fa-solid fa-circle-notch fa-spin"></i> Generating summary…</span>';
    if (tagEl) tagEl.textContent = '';

    try {
        const res = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dialogue, num_beams: 8, model_choice: model })
        });
        const data = await res.json();
        if (res.ok) {
            if (outputEl) outputEl.innerHTML = data.summary;
            if (tagEl) tagEl.textContent = data.model_version;
        } else {
            if (outputEl) outputEl.innerHTML = `<span style="color:#ef4444"><i class="fa-solid fa-circle-xmark"></i> ${data.detail}</span>`;
        }
    } catch {
        if (outputEl) outputEl.innerHTML = '<span style="color:#ef4444"><i class="fa-solid fa-plug-circle-xmark"></i> Failed to connect.</span>';
    } finally {
        btn.disabled = false;
        if (icon) icon.style.display = 'block';
        if (spinner) spinner.style.display = 'none';
    }
}

// ── Init ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    const ta = document.getElementById("dialogueInput");
    if (ta) {
        ta.addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); testSummarize(); }
        });
    }
});
