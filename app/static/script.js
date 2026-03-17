async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();

        // Add a small animation effect when loading
        animateValue("activePrompts", data.active_prompts);
        
        // Format score as percentage or 2 decimals
        document.getElementById("latestScore").innerText = data.latest_score;
        document.getElementById("modelVersion").innerText = data.model_version;
    } catch (error) {
        console.error("Failed to load metrics", error);
        document.getElementById("activePrompts").innerText = "Err";
        document.getElementById("latestScore").innerText = "Err";
        document.getElementById("modelVersion").innerText = "Offline";
    }
}

// Simple number counting animation
function animateValue(id, end, duration = 1000) {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * end);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

async function testSummarize() {
    const dialogue = document.getElementById("dialogueInput").value;
    const outputElem = document.getElementById("summaryOutput");
    const btn = document.getElementById("summarizeBtn");
    const btnIcon = document.getElementById("btnIcon");
    const spinner = document.getElementById("btnSpinner");
    const btnText = btn.querySelector("span");
    
    if (!dialogue.trim()) {
        outputElem.innerHTML = '<span style="color: var(--error);"><i class="fa-solid fa-triangle-exclamation"></i> Please enter a dialogue first.</span>';
        document.getElementById("dialogueInput").focus();
        return;
    }

    // Set Loading State
    btn.disabled = true;
    btnText.innerText = "Processing...";
    btnIcon.style.display = "none";
    spinner.style.display = "block";
    
    outputElem.innerHTML = '<span class="placeholder-text"><i class="fa-solid fa-circle-notch fa-spin"></i> Generating summary from model...</span>';

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dialogue: dialogue,
                num_beams: 8
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Success state
            outputElem.innerHTML = data.summary;
            outputElem.style.color = "var(--success)";
            setTimeout(() => { outputElem.style.color = "var(--text-main)"; }, 500);
        } else {
            outputElem.innerHTML = `<span style="color: var(--error);"><i class="fa-solid fa-circle-xmark"></i> Model Error: ${data.detail}</span>`;
        }
    } catch (error) {
        outputElem.innerHTML = '<span style="color: var(--error);"><i class="fa-solid fa-plug-circle-xmark"></i> Failed to connect to the PromptOps API.</span>';
    } finally {
        // Reset Button State
        btn.disabled = false;
        btnText.innerText = "Generate Summary";
        spinner.style.display = "none";
        btnIcon.style.display = "block";
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
});
