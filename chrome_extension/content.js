chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getPageData") {
        let riskContext = new Set();
        let formActions = [];

        // 1. Target Forms & grab their action URLs
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            if (form.innerText) riskContext.add(form.innerText.trim());
            if (form.action) formActions.push(form.action);
        });

        // 2. Target Context around Password Fields
        const passwordFields = document.querySelectorAll('input[type="password"]');
        passwordFields.forEach(field => {
            if (field.parentElement && field.parentElement.innerText) {
                riskContext.add(field.parentElement.innerText.trim());
            }
        });

        // 3. Target Buttons
        const buttons = document.querySelectorAll('button, a.button, input[type="submit"]');
        buttons.forEach(btn => {
            if (btn.innerText) riskContext.add(btn.innerText.trim());
        });

        // 4. Headings and paragraphs (core phishing signal text)
        // Phishing pages rely heavily on urgency text in h1/h2/p/label
        const textNodes = document.querySelectorAll('h1, h2, h3, p, span, label');
        textNodes.forEach(node => {
            const t = node.innerText?.trim();
            if (t && t.length > 20 && t.length < 500) {
                riskContext.add(t);
            }
        });

        // Clean up the text: remove massive blank spaces and limit to ~800 words
        // (DistilBERT handles 512 tokens — 800 words gives enough signal without overflow)
        let rawText = Array.from(riskContext).join(' | ');
        let cleanedText = rawText.replace(/\s+/g, ' ').trim();
        let finalText = cleanedText.split(' ').slice(0, 800).join(' ');

        // Fallback: If the page has no forms/buttons/headings, grab top body text
        if (!finalText) {
            finalText = document.body.innerText.replace(/\s+/g, ' ').trim().split(' ').slice(0, 800).join(' ');
        }

        // Send clean, targeted data back to popup
        sendResponse({
            url: window.location.href,
            text: finalText,
            form_destinations: formActions
        });
    }
});