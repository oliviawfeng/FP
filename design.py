#Individualized News Aggregator — CS32 Final Project
import requests
import string

#Sentiment word lists (Ch 10b: index-building pattern)

POSITIVE_WORDS = {
    "good", "great", "win", "success", "growth", "rise", "gain", "profit",
    "improve", "strong", "record", "breakthrough", "recover", "innovation",
    "approve", "deal", "expand", "stable", "surge", "boom"
}

NEGATIVE_WORDS = {
    "bad", "fail", "loss", "crash", "crisis", "risk", "decline", "drop",
    "fall", "plunge", "threat", "danger", "warning", "collapse", "scandal",
    "fraud", "bankrupt", "layoff", "deficit", "recession", "war", "shortage"
}


#String processing helpers (Ch 9-10)

def get_wordlist(text):
    #Clean and split text into a list of lowercase words.
    if not text:
        return []
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower().split()


def score_article(headline, description):
    # Score sentiment: (positive - negative) / total words.
    words = get_wordlist((headline or "") + " " + (description or ""))
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return round((pos - neg) / len(words), 4)


def remove_duplicates(articles):
    #Remove duplicate articles by hashing titles (Ch 10: hashing).
    seen = set()
    unique = []
    for article in articles:
        title = (article.get("title") or "").lower().strip()
        if title and title not in seen and title != "[removed]":
            seen.add(title)
            unique.append(article)
    return unique


# Fetch from NewsAPI (Ch 4: requests + JSON parsing)

def fetch_headlines(topic, api_key):
    #Fetch articles from NewsAPI for a given topic.
    params = {
        "q": topic,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 10,
        "apiKey": api_key,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    data = response.json()
    if data.get("status") != "ok":
        print(f"API Error: {data.get('message')}")
        return []
    return data.get("articles", [])


#Display results

def print_report(topic, articles):
    #Print a clean summary report to the terminal.
    print(f"\n{'='*60}")
    print(f"  News results for: {topic.upper()}  ({len(articles)} articles)")
    print(f"{'='*60}")
    for i, a in enumerate(articles, 1):
        score = score_article(a.get("title"), a.get("description"))
        label = "POSITIVE" if score > 0.02 else ("NEGATIVE" if score < -0.02 else "NEUTRAL")
        print(f"\n[{i}] {a.get('title')}")
        print(f"    Source    : {a.get('source', {}).get('name')}")
        print(f"    Sentiment : {label} ({score})")
        print(f"    Summary   : {(a.get('description') or 'N/A')[:120]}")
    print(f"\n{'='*60}\n")


#Main pipeline

def main():
    api_key = input("Enter your NewsAPI key: ").strip()
    topic   = input("Enter a topic to search: ").strip()

    raw_articles = fetch_headlines(topic, api_key)
    articles     = remove_duplicates(raw_articles)

    print_report(topic, articles)

if __name__ == "__main__":
    main()
