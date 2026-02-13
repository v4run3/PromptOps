async function loadMetrics() {
    const response = await fetch('/api/metrics');
    const data = await response.json();

    document.getElementById("activePrompts").innerText = data.active_prompts;
    document.getElementById("latestScore").innerText = data.latest_score;
    document.getElementById("modelVersion").innerText = data.model_version;
}

loadMetrics();
