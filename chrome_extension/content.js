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

        // Clean up the text: remove massive blank spaces and limit to ~500 words 
        // (to keep DistilBERT happy and avoid token limits)
        let rawText = Array.from(riskContext).join(' | ');
        let cleanedText = rawText.replace(/\s+/g, ' ').trim();
        let finalText = cleanedText.split(' ').slice(0, 500).join(' ');

        // Fallback: If the page is weird and has no forms/buttons, just grab the top text
        if (!finalText) {
            finalText = document.body.innerText.replace(/\s+/g, ' ').trim().split(' ').slice(0, 500).join(' ');
        }

        // Send this clean, targeted data back to the popup
        sendResponse({
            url: window.location.href,
            text: finalText,
            form_destinations: formActions
        });
    }
});