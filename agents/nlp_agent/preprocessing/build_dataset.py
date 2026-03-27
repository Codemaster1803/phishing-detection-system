import os  # <-- Added to handle folder creation and paths
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import urllib3

# Disable SSL warnings (phishing sites often have broken SSL certificates)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_dom_context(url):
    """Mimics your Chrome Extension's content.js extraction logic"""
    if not url.startswith('http'):
        url = 'https://' + url

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'lxml')
        risk_context = set()

        for form in soup.find_all('form'):
            text = form.get_text(separator=' ', strip=True)
            if text: risk_context.add(text)

        for pwd in soup.find_all('input', type='password'):
            parent = pwd.find_parent()
            if parent:
                text = parent.get_text(separator=' ', strip=True)
                if text: risk_context.add(text)

        for btn in soup.find_all(['button', 'input'], type=['submit', 'button']):
            text = btn.get_text(separator=' ', strip=True)
            if not text and btn.has_attr('value'):
                text = btn['value']
            if text: risk_context.add(text)

        raw_text = ' | '.join(risk_context)
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
        final_text = ' '.join(cleaned_text.split()[:500]) 
        
        return final_text if final_text else None

    except Exception:
        return None

def main():
    print("Loading URL lists...")
    
    with open('phishing.txt', 'r', encoding='utf-8') as f:
        phishing_urls = [line.strip() for line in f if line.strip()]
    
    tranco_df = pd.read_csv('tranco.csv', names=['rank', 'domain'])
    safe_urls = tranco_df['domain'].tolist()

    dataset = []
    target_samples = 300 

    print(f"\n--- Scraping {target_samples} Phishing Sites ---")
    phishing_count = 0
    for url in phishing_urls:
        text = extract_dom_context(url)
        if text:
            dataset.append({"text": text, "label": "phishing"})
            phishing_count += 1
            print(f"[{phishing_count}/{target_samples}] Scraped Phishing: {url}")
        
        if phishing_count >= target_samples:
            break

    print(f"\n--- Scraping {target_samples} Safe Sites ---")
    safe_count = 0
    for url in safe_urls:
        text = extract_dom_context(url)
        if text:
            dataset.append({"text": text, "label": "legitimate"})
            safe_count += 1
            print(f"[{safe_count}/{target_samples}] Scraped Safe: {url}")
        
        if safe_count >= target_samples:
            break

    df = pd.DataFrame(dataset)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # --- NEW FOLDER LOGIC ---
    folder_name = "dataset"
    
    # Check if 'dataset' folder exists, if not, create it automatically
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"\n📁 Created new folder: '{folder_name}'")
        
    # Create the full path: dataset/nlp_finetune_dataset.csv
    file_path = os.path.join(folder_name, "nlp_finetune_dataset.csv")
    
    # Save the CSV to that specific path
    df.to_csv(file_path, index=False)
    # ------------------------
    
    print("\n✅ SUCCESS!")
    print(f"Dataset saved at: '{file_path}' with {len(df)} total samples.")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()