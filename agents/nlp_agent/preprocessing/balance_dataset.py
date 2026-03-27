import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_dom_context(url):
    """The same exact extraction logic to ensure perfectly matching data."""
    if not isinstance(url, str) or not url.startswith('http'):
        url = 'https://' + str(url)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
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
    dataset_path = "dataset/nlp_finetune_dataset.csv"
    
    # 1. Load your existing dataset
    print("Loading existing dataset...")
    df = pd.read_csv(dataset_path)
    
    current_phish = len(df[df['label'] == 'phishing'])
    current_safe = len(df[df['label'] == 'legitimate'])
    
    print(f"📊 Current Balance: {current_phish} Phishing | {current_safe} Safe")
    
    needed_phish = current_safe - current_phish
    
    if needed_phish <= 0:
        print("Dataset is already balanced or has more phishing than safe!")
        return

    print(f"🎯 Goal: Scrape {needed_phish} more phishing sites to balance the data.")

    # 2. Load your NEW PhishTank CSV
    new_csv_filename = 'phishtank.csv' # <-- Make sure this matches your file name!
    
    print(f"Loading new URLs from {new_csv_filename}...")
    new_phish_df = pd.read_csv(new_csv_filename)
    
    # Extract just the 'url' column and convert it to a list
    new_urls = new_phish_df['url'].dropna().tolist()

    print(f"Loaded {len(new_urls)} new URLs to try...")

    # 3. Scrape until we hit the exact number needed
    new_data = []
    added_count = 0
    
    for url in new_urls:
        text = extract_dom_context(url)
        if text:
            new_data.append({"text": text, "label": "phishing"})
            added_count += 1
            print(f"[{added_count}/{needed_phish}] Scraped New Phishing: {url}")
            
        if added_count >= needed_phish:
            break

    # 4. Merge, Shuffle, and Save
    if new_data:
        new_df = pd.DataFrame(new_data)
        final_df = pd.concat([df, new_df], ignore_index=True)
        
        final_df = final_df.sample(frac=1).reset_index(drop=True)
        final_df.to_csv(dataset_path, index=False)
        
        print("\n✅ SUCCESS! Dataset is now perfectly balanced.")
        print(final_df['label'].value_counts())
    else:
        print("\n❌ Could not scrape any new sites. All new URLs might be dead.")

if __name__ == "__main__":
    main()