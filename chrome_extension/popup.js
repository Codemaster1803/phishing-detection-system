document.getElementById("scan").addEventListener("click", async () => {

document.getElementById("result").innerText = "Reading page...";
document.getElementById("confidence").innerText = "";
document.getElementById("explanation").innerText = "";

const [tab] = await chrome.tabs.query({active:true,currentWindow:true});

try{

await chrome.scripting.executeScript({
target: {tabId: tab.id},
files: ["content.js"]
});

chrome.tabs.sendMessage(tab.id,{action:"getPageData"}, async (response)=>{

if(!response){
document.getElementById("result").innerText="Could not read page content";
return;
}

document.getElementById("result").innerText = "Analyzing with AI agents...";

const apiResponse = await fetch("http://127.0.0.1:8000/analyze",{

method:"POST",
headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
url:response.url,
text:response.text
})

});

const data = await apiResponse.json();

const resultElement = document.getElementById("result");

resultElement.innerText =
data.final_label + " (" + data.final_probability.toFixed(2) + ")";

if(data.final_label === "Phishing"){
resultElement.style.color = "#ff4d4d";
}else{
resultElement.style.color = "#22c55e";
}

document.getElementById("confidence").innerText =
"Confidence: " + data.confidence;

document.getElementById("url_score").innerText =
data.agents.url_agent.score.toFixed(2);

document.getElementById("nlp_score").innerText =
data.agents.nlp_agent.score.toFixed(2);

document.getElementById("domain_score").innerText =
data.agents.domain_agent.score.toFixed(2);

document.getElementById("explanation").innerText =
data.explanation;

});

}catch(error){

document.getElementById("result").innerText =
"Error connecting to backend";

}

});