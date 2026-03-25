"""
WHOIS Dataset Enrichment Script
================================
Run this ONCE to enrich your url_data_clean.csv with WHOIS features.
Results are saved to url_data_whois.csv which your training script uses.

WHY SEPARATE SCRIPT:
- 188k URLs × 1 second/query = 52 hours if done naively
- But there are only ~50k-80k UNIQUE domains in 188k URLs
- With caching, each domain is only queried ONCE
- Estimated time: 14-22 hours (run overnight)

HOW TO RUN:
    python whois_enrichment.py

It saves progress every 1000 URLs — safe to interrupt and resume.
"""

import pandas as pd
import os
import time
from whois_features import extract_whois_features, extract_root_domain, get_cached

# ── Config ─────────────────────────────────
INPUT_CSV  = "../datasets/url_data_clean.csv"
OUTPUT_CSV = "../datasets/url_data_whois.csv"
DELAY_SECS = 1.0     # seconds between NEW domain queries
SAVE_EVERY = 1000    # save progress every N rows
# ───────────────────────────────────────────


def enrich_dataset():
    print("=" * 60)
    print("WHOIS DATASET ENRICHMENT")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} URLs")

    # Resume from checkpoint if exists
    start_idx = 0
    if os.path.exists(OUTPUT_CSV):
        done = pd.read_csv(OUTPUT_CSV)
        start_idx = len(done)
        print(f"Resuming from row {start_idx} (already processed {start_idx} rows)")
        df = df.iloc[start_idx:].reset_index(drop=True)

    # Count unique domains not yet cached
    unique_domains = df["url"].apply(extract_root_domain).unique()
    uncached = [d for d in unique_domains if get_cached(d) is None]
    print(f"Unique domains: {len(unique_domains)} | Uncached (need querying): {len(uncached)}")
    print(f"Estimated time for uncached: ~{len(uncached) * DELAY_SECS / 3600:.1f} hours")
    print(f"Saving progress every {SAVE_EVERY} rows to {OUTPUT_CSV}")
    print("\nStarting enrichment... (safe to Ctrl+C and resume)\n")

    results = []
    queried = set()

    for i, row in df.iterrows():
        url = str(row["url"])
        domain = extract_root_domain(url)

        # Delay only for new uncached domains
        if domain and domain not in queried and get_cached(domain) is None:
            queried.add(domain)
            feats = extract_whois_features(url, live_lookup=True)
            time.sleep(DELAY_SECS)
        else:
            feats = extract_whois_features(url, live_lookup=True)

        result_row = {
            "url":   url,
            "label": row["label"],
            **feats
        }
        results.append(result_row)

        # Print progress
        if (i + 1) % 100 == 0:
            cached_hits = len(queried)
            print(f"  [{i+1}/{len(df)}] Domains queried: {cached_hits} | "
                  f"Last: {domain[:40]}")

        # Save checkpoint
        if (i + 1) % SAVE_EVERY == 0:
            chunk = pd.DataFrame(results)
            if os.path.exists(OUTPUT_CSV) and start_idx > 0:
                chunk.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            else:
                chunk.to_csv(OUTPUT_CSV, index=False)
            results = []
            print(f"  ✅ Checkpoint saved at row {start_idx + i + 1}")

    # Save remaining
    if results:
        chunk = pd.DataFrame(results)
        if os.path.exists(OUTPUT_CSV) and start_idx > 0:
            chunk.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        else:
            chunk.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Done! Enriched dataset saved to: {OUTPUT_CSV}")

    # Show stats
    final = pd.read_csv(OUTPUT_CSV)
    print(f"\nDataset shape: {final.shape}")
    print(f"Failed WHOIS lookups: {final['whois_lookup_failed'].sum()} "
          f"({final['whois_lookup_failed'].mean()*100:.1f}%)")
    print(f"\nDomain age stats (days):")
    age_known = final[final["domain_age_days"] > 0]["domain_age_days"]
    print(age_known.describe())


if __name__ == "__main__":
    enrich_dataset()
