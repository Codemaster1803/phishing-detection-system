chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

if(request.action === "getPageData"){

sendResponse({
url: window.location.href,
text: document.body.innerText.substring(0,3000)
});

}

});