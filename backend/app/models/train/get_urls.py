import requests
import pandas as pd
import zipfile
import io
import csv

def get_urlhaus_links():
    url = "https://urlhaus.abuse.ch/downloads/text_online/"
    response = requests.get(url)
    lines = response.text.split('\n')
    urls = [line for line in lines if line.startswith("http")]
    return urls

urlhaus_urls = get_urlhaus_links()
df_phish = pd.DataFrame({'url': urlhaus_urls, 'label': 0})

def download_tranco_list():
    url = "https://tranco-list.eu/top-1m.csv.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("tranco_list")
    df = pd.read_csv("tranco_list/top-1m.csv", header=None, names=["rank", "domain"])
    return df

tranco_df = download_tranco_list()
# Convert top 10,000 domains to URLs
legit_urls = tranco_df["domain"].head(12821).apply(lambda d: f"https://{d}")
df_legit = pd.DataFrame({'url': legit_urls, 'label': 1})

df_phiusiil = pd.read_csv("backend/app/models/train/datasets/PhiUSIIL_Phishing_URL_Dataset.csv")
df_phiusiil = df_phiusiil.rename(columns=lambda x: x.strip().lower())
df_phiusiil = df_phiusiil.head(14358)
df_phiusiil = df_phiusiil[['url', 'label']]
df_phiusiil = df_phiusiil[df_phiusiil['url'].apply(lambda x: isinstance(x, str) and x.startswith("http"))]

df_all = pd.concat([df_phish, df_legit, df_phiusiil], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(df_all.head())

def clean_url(url):
    return url.strip().strip('"').strip("'")

df_all["url"] = df_all["url"].apply(clean_url)

print(df_all['label'].value_counts())

# ✅ Save with all values quoted (helps avoid parsing issues later)
df_all.to_csv("backend/app/models/train/datasets/phishing_legit_dataset.csv", index=False)
print("✅ Saved as phishing_legit_dataset.csv")
