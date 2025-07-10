import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import csv

def google_search(query, max_results=20, pause=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    urls = []
    start = 0

    while len(urls) < max_results:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&start={start}"
        response = requests.get(search_url, headers=headers)

        if response.status_code != 200:
            print("Failed to fetch search results")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select('a[href^="/url?q="]')

        for link in results:
            try:
                raw_url = link["href"].split("/url?q=")[1].split("&")[0]
                decoded_url = urllib.parse.unquote(raw_url)
                if any(site in decoded_url for site in ["youtube.com", "facebook.com", "instagram.com"]):
                    if decoded_url not in urls:
                        urls.append(decoded_url)
                        print(f"[+] Found: {decoded_url}")
                        if len(urls) >= max_results:
                            break
            except IndexError:
                continue

        start += 10
        time.sleep(pause)

    return urls

def save_to_csv(urls, filename="social_media_urls.csv"):
    with open(filename, mode="w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL"])  # Header
        for url in urls:
            writer.writerow([url])
    print(f"\n✅ Saved {len(urls)} URLs to {filename}")

# Example usage
if __name__ == "__main__":
    query = "site:youtube.com tutorial OR site:facebook.com post OR site:instagram.com reel"
    urls = google_search(query, max_results=500)
    save_to_csv(urls)
