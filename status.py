import requests
import string


#  Sentiment word lists (Ch 10b: index-building pattern)

POSITIVE_WORDS = {
    "good", "great", "win", "success", "growth", "rise", "gain", "profit",
    "improve", "strong", "record", "breakthrough", "recover", "innovation",
    "approve", "deal", "expand", "stable", "surge", "boom", "optimistic",
    "progress", "positive", "benefit", "efficient", "support"
}

NEGATIVE_WORDS = {
    "bad", "fail", "loss", "crash", "crisis", "risk", "decline", "drop",
    "fall", "plunge", "threat", "danger", "warning", "collapse", "scandal",
    "fraud", "bankrupt", "layoff", "deficit", "recession", "war", "shortage",
    "negative", "conflict", "fear", "concern", "weak", "slowdown"
}

# Small stopword set for title similarity/keyword matching
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "into", "over", "after",
    "amid", "about"
}


#  String processing helpers (Ch 9-10)

def get_wordlist(text):
    #Clean and split text into a list of lowercase words.#
    if not text:
        return []
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower().split()


def meaningful_words(text):
    #Return a set of non-trivial words for similarity checks.#
    return {
        w for w in get_wordlist(text)
        if len(w) > 2 and w not in STOPWORDS
    }


def sentiment_details(headline, description):

    #Return sentiment details using only matched sentiment words.
    #Score is normalized by matched sentiment words, not total words.

    words = get_wordlist((headline or "") + " " + (description or ""))

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    matched = pos + neg

    if matched == 0:
        score = 0.0
    else:
        score = round((pos - neg) / matched, 4)

    return {
        "score": score,
        "positive_matches": pos,
        "negative_matches": neg,
        "matched_words": matched
    }


def score_article(headline, description):
    #Compatibility wrapper returning only the sentiment score
    return sentiment_details(headline, description)["score"]


#  Relevance scoring (custom article ranking)

def relevance_score(topic, headline, description):
    # Score how relevant an article is to the user's topic.
  #Higher is better.

    topic = (topic or "").strip().lower()
    title_text = headline or ""
    desc_text = description or ""
    full_text = (title_text + " " + desc_text).lower()

    topic_words = [
        w for w in get_wordlist(topic)
        if w not in STOPWORDS
    ]

    title_words = get_wordlist(title_text)
    desc_words = get_wordlist(desc_text)

    score = 0

    # Exact phrase bonuses
    if topic and topic in title_text.lower():
        score += 8
    if topic and topic in full_text:
        score += 4

    # Keyword frequency bonuses
    distinct_matches = 0
    for word in topic_words:
        title_count = title_words.count(word)
        desc_count = desc_words.count(word)

        if title_count > 0 or desc_count > 0:
            distinct_matches += 1

        # Weight title matches more heavily
        score += title_count * 3
        score += desc_count * 1

    # Reward coverage of multiple topic terms
    score += distinct_matches * 2

    return score


def enrich_article(article, topic):
    #Attach computed fields used for ranking and display
    title = article.get("title") or ""
    description = article.get("description") or ""

    sentiment = sentiment_details(title, description)
    rel = relevance_score(topic, title, description)

    enriched = dict(article)
    enriched["custom_relevance"] = rel
    enriched["sentiment_score"] = sentiment["score"]
    enriched["positive_matches"] = sentiment["positive_matches"]
    enriched["negative_matches"] = sentiment["negative_matches"]
    enriched["matched_sentiment_words"] = sentiment["matched_words"]

    return enriched


def sort_articles(articles, topic):
    #Rank articles using custom relevance, then sentiment strength
    enriched = [enrich_article(article, topic) for article in articles]

    enriched.sort(
        key=lambda a: (
            a["custom_relevance"],
            abs(a["sentiment_score"]),
            a.get("publishedAt", "")
        ),
        reverse=True
    )
    return enriched


#  Duplicate removal

def jaccard_similarity(set1, set2):
    #Return Jaccard similarity between two sets.#
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def remove_duplicates(articles, similarity_threshold=0.75):
    #Remove duplicate and near-duplicate articles.
    #Exact duplicates are checked by normalized title string.
    #Near-duplicates are checked by title word overlap.

    seen_titles = set()
    kept = []
    kept_title_wordsets = []

    for article in articles:
        title = (article.get("title") or "").strip()
        normalized_title = title.lower()

        if not title or normalized_title == "[removed]":
            continue

        # Exact duplicate title
        if normalized_title in seen_titles:
            continue

        # Near-duplicate title
        current_words = meaningful_words(title)
        is_near_duplicate = False

        for prior_words in kept_title_wordsets:
            if jaccard_similarity(current_words, prior_words) >= similarity_threshold:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        seen_titles.add(normalized_title)
        kept.append(article)
        kept_title_wordsets.append(current_words)

    return kept


#  Fetch from NewsAPI (Ch 4: requests + JSON parsing)

def fetch_headlines(topic, api_key):
    #Fetch articles from NewsAPI for a given topic.#
    params = {
        "q": topic,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 15,
        "apiKey": api_key,
    }

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10
        )
        data = response.json()
    except requests.RequestException as exc:
        print(f"Network Error: {exc}")
        return []
    except ValueError:
        print("Error: could not decode JSON response.")
        return []

    if data.get("status") != "ok":
        print(f"API Error: {data.get('message')}")
        return []

    return data.get("articles", [])


#  Display results

def sentiment_label(score):
    #Map numeric sentiment score to a label.#
    if score > 0.2:
        return "POSITIVE"
    if score < -0.2:
        return "NEGATIVE"
    return "NEUTRAL"


def print_report(topic, articles):
    #Print a clean summary report to the terminal.#
    print(f"\n{'=' * 72}")
    print(f"  News results for: {topic.upper()}  ({len(articles)} articles)")
    print(f"{'=' * 72}")

    if not articles:
        print("\nNo articles found.\n")
        print(f"{'=' * 72}\n")
        return

    for i, article in enumerate(articles, 1):
        title = article.get("title") or "N/A"
        source = article.get("source", {}).get("name", "Unknown")
        summary = (article.get("description") or "N/A")[:140]

        rel = article.get("custom_relevance", 0)
        score = article.get("sentiment_score", 0.0)
        pos = article.get("positive_matches", 0)
        neg = article.get("negative_matches", 0)
        label = sentiment_label(score)

        print(f"\n[{i}] {title}")
        print(f"    Source       : {source}")
        print(f"    Relevance    : {rel}")
        print(f"    Sentiment    : {label} ({score})")
        print(f"    Pos/Neg Hits : {pos}/{neg}")
        print(f"    Summary      : {summary}")

    print(f"\n{'=' * 72}\n")


#  Main pipeline

def main():
    api_key = input("Enter your NewsAPI key: ").strip()
    topic = input("Enter a topic to search: ").strip()

    raw_articles = fetch_headlines(topic, api_key)
    unique_articles = remove_duplicates(raw_articles)
    ranked_articles = sort_articles(unique_articles, topic)

    print_report(topic, ranked_articles)


if __name__ == "__main__":
    main()

