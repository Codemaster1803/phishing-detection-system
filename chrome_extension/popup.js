document.getElementById("scan").addEventListener("click", async () => {

    const resultElement = document.getElementById("result");
    const confidenceElement = document.getElementById("confidence");
    const explanationList = document.getElementById("explanation");

    const loading = document.getElementById("loading");
    const loadingText = document.getElementById("loadingText");
    const scanBtn = document.getElementById("scan");
    const resultCard = document.getElementById("resultCard");

    // Badge helper function (clean + reusable)
    function updateBadge(type, text) {
        const badge = document.getElementById("statusBadge");
        badge.className = "badge " + type;
        badge.innerText = text;
    }

    // =========================
    // SHOW RESULT SECTION ONLY AFTER CLICK
    // =========================
    resultCard.classList.remove("hidden");

    // =========================
    // START LOADING STATE
    // =========================
    loading.classList.remove("hidden");
    loadingText.innerText = "Reading page...";
    scanBtn.disabled = true;

    resultElement.innerText = "Reading page...";
    confidenceElement.innerText = "";
    explanationList.innerHTML = "";

    updateBadge("neutral", "SCANNING...");

    // Reset bars
    document.getElementById("url_bar").style.width = "0%";
    document.getElementById("nlp_bar").style.width = "0%";
    document.getElementById("domain_bar").style.width = "0%";

    try {

        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });



        loadingText.innerText = "Extracting page data...";

        chrome.tabs.sendMessage(tab.id, { action: "getPageData" }, async (response) => {

            if (!response) {
                resultElement.innerText = "❌ Could not read page content";
                updateBadge("danger", "ERROR");

                loading.classList.add("hidden");
                scanBtn.disabled = false;
                return;
            }

            resultElement.innerText = "Analyzing with AI...";
            loadingText.innerText = "Running AI agents...";

            try {

                const apiResponse = await fetch("http://127.0.0.1:8000/analyze", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        url: response.url,
                        text: response.text,
                        form_destinations: response.form_destinations
                    })
                });

                const data = await apiResponse.json();

                // =========================
                // RESULT + BADGE
                // =========================
                resultElement.innerText = data.final_label;
                confidenceElement.innerText = "Confidence: " + data.confidence;

                if (data.final_label === "Phishing") {
                    updateBadge("danger", "🚨 PHISHING");
                } else if (data.final_label === "Suspicious") {
                    updateBadge("warn", "⚠️ SUSPICIOUS");
                } else {
                    updateBadge("safe", "🟢 SAFE");
                }

                // =========================
                // AGENT SCORES (BARS)
                // =========================
                document.getElementById("url_bar").style.width =
                    (data.agents.url_agent.score * 100) + "%";

                document.getElementById("nlp_bar").style.width =
                    (data.agents.nlp_agent.score * 100) + "%";

                document.getElementById("domain_bar").style.width =
                    (data.agents.domain_agent.score * 100) + "%";

                // =========================
                // EXPLANATION (LIST)
                // =========================
                explanationList.innerHTML = "";

                if (data.explanation) {
                    data.explanation.split(".").forEach(item => {
                        if (item.trim()) {
                            const li = document.createElement("li");
                            li.innerText = item.trim();
                            explanationList.appendChild(li);
                        }
                    });
                }

                // =========================
                // END LOADING
                // =========================
                loading.classList.add("hidden");
                scanBtn.disabled = false;

            } catch (apiError) {
                resultElement.innerText = "❌ Backend error";
                updateBadge("danger", "ERROR");

                loading.classList.add("hidden");
                scanBtn.disabled = false;
            }

        });

    } catch (error) {
        resultElement.innerText = "❌ Error connecting to backend";
        updateBadge("danger", "ERROR");

        loading.classList.add("hidden");
        scanBtn.disabled = false;
    }

});