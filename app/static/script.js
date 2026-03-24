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
let inferenceStats = { totalRequests: 0, totalLatency: 0, lastLatency: 0, modelLoaded: false };

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

    const startTime = performance.now();

    try {
        const res = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dialogue, num_beams: 8, model_choice: model })
        });
        const data = await res.json();
        const endTime = performance.now();
        const latency = ((endTime - startTime) / 1000).toFixed(2);

        // Update inference stats
        inferenceStats.totalRequests++;
        inferenceStats.lastLatency = latency;
        inferenceStats.totalLatency += parseFloat(latency);
        inferenceStats.modelLoaded = true;
        updateInferencePanel();

        if (res.ok) {
            if (outputEl) outputEl.innerHTML = data.summary;
            if (tagEl) tagEl.textContent = data.model_version;
        } else {
            if (outputEl) outputEl.innerHTML = `<span style="color:#ef4444"><i class="fa-solid fa-circle-xmark"></i> ${data.detail}</span>`;
        }
    } catch {
        if (outputEl) outputEl.innerHTML = '<span style="color:#ef4444"><i class="fa-solid fa-plug-circle-xmark"></i> Failed to connect.</span>';
        const el = document.getElementById("statHealth");
        if (el) { el.textContent = "● Offline"; el.style.color = "#f87171"; }
    } finally {
        btn.disabled = false;
        if (icon) icon.style.display = 'block';
        if (spinner) spinner.style.display = 'none';
    }
}

// ── Inference Panel ────────────────────────────
function updateInferencePanel() {
    const s = inferenceStats;
    const el = id => document.getElementById(id);
    if (el("statLatency")) el("statLatency").textContent = s.lastLatency + "s";
    if (el("statRequests")) el("statRequests").textContent = s.totalRequests;
    if (el("statAvgLatency")) el("statAvgLatency").textContent = (s.totalLatency / s.totalRequests).toFixed(2) + "s";
    if (el("statModelLoaded")) {
        el("statModelLoaded").textContent = s.modelLoaded ? "Yes" : "No";
        el("statModelLoaded").style.color = s.modelLoaded ? "#34d399" : "#aaa";
    }
}

function toggleInferencePanel() {
    const panel = document.getElementById("inferencePanel");
    const btn = document.getElementById("inferenceToggle");
    const other = document.getElementById("privatePanel");
    const otherBtn = document.getElementById("privateToggle");
    if (other) { other.classList.remove("open"); otherBtn.classList.remove("active"); }
    panel.classList.toggle("open");
    btn.classList.toggle("active");
}

// ── Private Mode ───────────────────────────────
let isPrivate = false;

function togglePrivateMode() {
    const panel = document.getElementById("privatePanel");
    const btn = document.getElementById("privateToggle");
    const other = document.getElementById("inferencePanel");
    const otherBtn = document.getElementById("inferenceToggle");
    if (other) { other.classList.remove("open"); otherBtn.classList.remove("active"); }
    panel.classList.toggle("open");
    btn.classList.toggle("active");
}

function switchPrivacy() {
    isPrivate = !isPrivate;
    const modeEl = document.getElementById("privacyMode");
    const btnEl = document.getElementById("privacyBtn");
    const topBtn = document.getElementById("privateToggle");
    const docsLink = document.getElementById("apiDocsLink");

    if (isPrivate) {
        modeEl.textContent = "Private";
        modeEl.style.color = "#f59e0b";
        btnEl.innerHTML = '<i class="fa-solid fa-lock"></i> Switch to Public';
        topBtn.innerHTML = '<i class="fa-solid fa-lock"></i> Private';
        if (docsLink) { docsLink.textContent = "Disabled"; docsLink.removeAttribute("href"); docsLink.style.color = "#666"; }
        showToast('<i class="fa-solid fa-lock" style="color:#f59e0b"></i> API set to Private — docs disabled');
    } else {
        modeEl.textContent = "Public";
        modeEl.style.color = "#34d399";
        btnEl.innerHTML = '<i class="fa-solid fa-lock-open"></i> Switch to Private';
        topBtn.innerHTML = '<i class="fa-solid fa-lock-open"></i> Public';
        if (docsLink) { docsLink.textContent = "/docs"; docsLink.href = "/docs"; docsLink.style.color = "#4a7cf7"; }
        showToast('<i class="fa-solid fa-lock-open" style="color:#34d399"></i> API set to Public — docs enabled');
    }
}

// ── Toast ──────────────────────────────────────
function showToast(html) {
    let toast = document.getElementById("appToast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "appToast";
        toast.className = "toast";
        document.body.appendChild(toast);
    }
    toast.innerHTML = html;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2500);
}

// ── Click outside to close dropdowns ───────────
document.addEventListener("click", e => {
    document.querySelectorAll(".topbar-dropdown-wrap").forEach(wrap => {
        if (!wrap.contains(e.target)) {
            wrap.querySelector(".topbar-dropdown")?.classList.remove("open");
            wrap.querySelector(".topbar-btn")?.classList.remove("active");
        }
    });
});

// ── Health check ───────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const el = document.getElementById("statHealth");
        if (res.ok && el) { el.textContent = "● Healthy"; el.style.color = "#34d399"; }
    } catch {
        const el = document.getElementById("statHealth");
        if (el) { el.textContent = "● Offline"; el.style.color = "#f87171"; }
    }
}

// ── Init ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    checkHealth();
    const ta = document.getElementById("dialogueInput");
    if (ta) {
        ta.addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); testSummarize(); }
        });
    }
});

