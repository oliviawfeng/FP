"""

Individualized News Aggregator CS32 Final Project

This program lets a user enter a news topic and returns a cleaned,
ranked, and organized terminal report of relevant articles.

Main steps:
1. Fetch articles from NewsAPI.
2. Remove duplicate and near-duplicate articles.
3. Score articles by custom relevance.
4. Score article sentiment.
5. Extract simple subtopics.
6. Print a grouped report.

Setup:
    pip install requests

Usage:
    python news_aggregator.py

External service:
    NewsAPI — https://newsapi.org/
"""

import requests
import string
from collections import Counter


# Sentiment word lists

POSITIVE_WORDS = {
    "good", "great", "win", "success", "growth", "rise", "gain", "profit",
    "improve", "strong", "record", "breakthrough", "recover", "innovation",
    "approve", "deal", "expand", "stable", "surge", "boom", "optimistic",
    "progress", "positive", "benefit", "efficient", "support", "advance",
    "boost", "leader", "strength", "opportunity"
}

NEGATIVE_WORDS = {
    "bad", "fail", "loss", "crash", "crisis", "risk", "decline", "drop",
    "fall", "plunge", "threat", "danger", "warning", "collapse", "scandal",
    "fraud", "bankrupt", "layoff", "deficit", "recession", "war", "shortage",
    "negative", "conflict", "fear", "concern", "weak", "slowdown", "lawsuit",
    "probe", "attack", "uncertainty"
}


# Common words we do not want to use for keyword relevance or subtopics.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "into",
    "over", "after", "before", "amid", "about", "new", "more", "most",
    "will", "would", "could", "should", "can", "may", "than", "then",
    "their", "his", "her", "our", "your", "they", "them", "you", "we",
    "he", "she", "i", "not", "no", "yes", "said", "says", "say", "who",
    "what", "when", "where", "why", "how"
}


# Text processing helpers

def get_wordlist(text):
    """
    Clean text and split it into lowercase words.

    This function gives the rest of the program a consistent text format.
    For example, "Growth!" becomes "growth", so it can match our word sets.
    """
    if not text:
        return []

    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.translate(translator).lower()
    return cleaned.split()


def meaningful_words(text):
    """
    Return non-stopword words that are useful for matching.

    This removes very common words like "the" and "and" so they do not
    affect relevance scoring, duplicate detection, or subtopic extraction.
    """
    return [
        word for word in get_wordlist(text)
        if len(word) > 2 and word not in STOPWORDS
    ]


def get_article_text(article):
    """Combine the most useful text fields from an article."""
    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""

    return f"{title} {description} {content}"


def normalize_topic(topic):
    """
    Turn a user topic into a clean phrase and useful keyword list.

    For multi-word topics, the exact phrase matters most. Individual words
    are only used if they are meaningful, non-stopword terms.
    """
    phrase = " ".join(get_wordlist(topic))
    keywords = meaningful_words(topic)

    return phrase, keywords


# Sentiment scoring

def sentiment_details(headline, description):
    """
    Score sentiment using only matched sentiment words.

    Score range:
        1.0   = all matched sentiment words are positive
        -1.0  = all matched sentiment words are negative
        0.0   = no sentiment words or equal positive/negative balance

    We divide by matched sentiment words instead of total words so the
    score is not diluted by neutral words.
    """
    words = get_wordlist((headline or "") + " " + (description or ""))

    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    matched = positive_count + negative_count

    if matched == 0:
        score = 0.0
    else:
        score = round((positive_count - negative_count) / matched, 4)

    return {
        "score": score,
        "positive_matches": positive_count,
        "negative_matches": negative_count,
        "matched_words": matched
    }


def sentiment_label(score):
    """Convert a numeric sentiment score into a label."""
    if score > 0.2:
        return "POSITIVE"

    if score < -0.2:
        return "NEGATIVE"

    return "NEUTRAL"


# Custom relevance scoring

def relevance_score(topic, article):
    """
    Compute a custom relevance score for an article.

    Exact multi-word phrase matches are weighted much more heavily than
    individual keyword matches. This helps prevent a search like
    "stable coins" from matching unrelated articles that only contain
    the generic word "stable."
    """
    phrase, keywords = normalize_topic(topic)

    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""

    title_clean = " ".join(get_wordlist(title))
    description_clean = " ".join(get_wordlist(description))
    content_clean = " ".join(get_wordlist(content))
    full_clean = f"{title_clean} {description_clean} {content_clean}"

    title_words = get_wordlist(title)
    description_words = get_wordlist(description)
    content_words = get_wordlist(content)

    score = 0

    # Exact phrase matching is the strongest signal.
    if phrase and phrase in title_clean:
        score += 30

    if phrase and phrase in description_clean:
        score += 18

    if phrase and phrase in content_clean:
        score += 8

    full_words = full_clean.split()

    # If the topic has multiple useful words, reward articles containing all of them.
    if len(keywords) >= 2:
        matched_keywords = [word for word in keywords if word in full_words]

        if len(matched_keywords) == len(keywords):
            score += 12
    else:
        matched_keywords = [word for word in keywords if word in full_words]

    # Individual keyword matching still matters, but less than phrase matching.
    distinct_matches = 0

    for word in keywords:
        title_count = title_words.count(word)
        description_count = description_words.count(word)
        content_count = content_words.count(word)

        if title_count + description_count + content_count > 0:
            distinct_matches += 1

        # Title matches receive the highest weight because titles usually
        # capture the main topic of the article.
        score += title_count * 5
        score += description_count * 2
        score += content_count * 1

    # Reward broader coverage of topic words.
    score += distinct_matches * 3

    return score


def enrich_article(article, topic):
    """Add computed relevance and sentiment fields to an article."""
    title = article.get("title") or ""
    description = article.get("description") or ""

    sentiment = sentiment_details(title, description)
    enriched = dict(article)

    enriched["custom_relevance"] = relevance_score(topic, article)
    enriched["sentiment_score"] = sentiment["score"]
    enriched["sentiment_label"] = sentiment_label(sentiment["score"])
    enriched["positive_matches"] = sentiment["positive_matches"]
    enriched["negative_matches"] = sentiment["negative_matches"]
    enriched["matched_sentiment_words"] = sentiment["matched_words"]

    return enriched


def sort_articles(articles, topic):
    """Sort articles by custom relevance, then sentiment strength, then date."""
    enriched_articles = [enrich_article(article, topic) for article in articles]

    enriched_articles.sort(
        key=lambda article: (
            article["custom_relevance"],
            abs(article["sentiment_score"]),
            article.get("publishedAt", "")
        ),
        reverse=True
    )

    return enriched_articles


# Duplicate and near-duplicate detection

def jaccard_similarity(words1, words2):
    """
    Return Jaccard similarity between two word collections.

    Jaccard similarity = size of intersection / size of union.
    A higher score means the two sets share more words.
    """
    set1 = set(words1)
    set2 = set(words2)

    if not set1 and not set2:
        return 1.0

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / len(set1 | set2)


def is_near_duplicate(current_article, kept_article, threshold=0.82):
    """
    Decide whether two articles are near-duplicates.

    The threshold is intentionally conservative. We do not want to remove
    genuinely different articles just because they share topic vocabulary.
    """
    current_title = current_article.get("title") or ""
    kept_title = kept_article.get("title") or ""

    current_words = meaningful_words(current_title)
    kept_words = meaningful_words(kept_title)

    title_similarity = jaccard_similarity(current_words, kept_words)

    # If title overlap is extremely high, it is probably a duplicate.
    if title_similarity >= threshold:
        return True

    # If titles are somewhat similar, also require description similarity.
    # This prevents over-filtering distinct articles with similar topic words.
    if title_similarity >= 0.65:
        current_desc = meaningful_words(current_article.get("description") or "")
        kept_desc = meaningful_words(kept_article.get("description") or "")
        desc_similarity = jaccard_similarity(current_desc, kept_desc)

        if desc_similarity >= 0.50:
            return True

    return False


def remove_duplicates(articles):
    """
    Remove exact duplicates and conservative near-duplicates.

    Exact duplicates use normalized title strings.
    Near-duplicates use word overlap with a conservative threshold.
    """
    seen_titles = set()
    kept_articles = []

    for article in articles:
        title = (article.get("title") or "").strip()
        normalized_title = title.lower()

        if not title or normalized_title == "[removed]":
            continue

        if normalized_title in seen_titles:
            continue

        duplicate_found = False

        for kept_article in kept_articles:
            if is_near_duplicate(article, kept_article):
                duplicate_found = True
                break

        if not duplicate_found:
            seen_titles.add(normalized_title)
            kept_articles.append(article)

    return kept_articles


# Subtopic grouping

def top_terms_for_article(article, topic, max_terms=3):
    """
    Extract a few meaningful terms from an article.

    This is a simple subtopic feature based on high-frequency non-stopwords.
    """
    _, topic_keywords = normalize_topic(topic)

    words = meaningful_words(get_article_text(article))

    # Remove original search terms so subtopics add new information.
    filtered_words = [
        word for word in words
        if word not in topic_keywords
    ]

    counts = Counter(filtered_words)
    return [word for word, count in counts.most_common(max_terms)]


def assign_subtopic(article, topic):
    """
    Assign a simple subtopic label.

    Because we do not have labeled categories, we use the most frequent
    meaningful non-topic word as a lightweight cluster label.
    """
    terms = top_terms_for_article(article, topic, max_terms=1)

    if not terms:
        return "general"

    return terms[0]


def add_subtopics(articles, topic):
    """Attach a subtopic label and top terms to each article."""
    updated_articles = []

    for article in articles:
        article_copy = dict(article)
        article_copy["subtopic"] = assign_subtopic(article_copy, topic)
        article_copy["top_terms"] = top_terms_for_article(article_copy, topic)
        updated_articles.append(article_copy)

    return updated_articles


def group_by_sentiment(articles):
    """Group articles by sentiment label."""
    groups = {
        "POSITIVE": [],
        "NEUTRAL": [],
        "NEGATIVE": []
    }

    for article in articles:
        label = article.get("sentiment_label", "NEUTRAL")
        groups[label].append(article)

    return groups


# Fetch from NewsAPI

def fetch_headlines(topic, api_key):
    """Fetch articles from NewsAPI for a given topic."""
    params = {
        "q": topic,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 20,
        "apiKey": api_key,
    }

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10
        )

        data = response.json()

    except requests.RequestException as error:
        print(f"Network Error: {error}")
        return []

    except ValueError:
        print("Error: could not decode JSON response.")
        return []

    if data.get("status") != "ok":
        print(f"API Error: {data.get('message')}")
        return []

    return data.get("articles", [])


# Display results

def print_article(article, index):
    """Print one article in a readable format."""
    title = article.get("title") or "N/A"
    source = article.get("source", {}).get("name", "Unknown")
    summary = (article.get("description") or "N/A")[:160]

    relevance = article.get("custom_relevance", 0)
    sentiment = article.get("sentiment_label", "NEUTRAL")
    score = article.get("sentiment_score", 0.0)
    pos = article.get("positive_matches", 0)
    neg = article.get("negative_matches", 0)
    subtopic = article.get("subtopic", "general")
    top_terms = article.get("top_terms", [])

    print(f"\n[{index}] {title}")
    print(f"    Source       : {source}")
    print(f"    Relevance    : {relevance}")
    print(f"    Sentiment    : {sentiment} ({score})")
    print(f"    Pos/Neg Hits : {pos}/{neg}")
    print(f"    Subtopic     : {subtopic}")
    print(f"    Top Terms    : {', '.join(top_terms) if top_terms else 'N/A'}")
    print(f"    Summary      : {summary}")


def print_report(topic, articles):
    """Print a clean summary report grouped by sentiment."""
    print(f"\n{'=' * 78}")
    print(f"  News results for: {topic.upper()}  ({len(articles)} articles)")
    print(f"{'=' * 78}")

    if not articles:
        print("\nNo articles found.")
        print(f"{'=' * 78}\n")
        return

    grouped = group_by_sentiment(articles)
    article_number = 1

    for label in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        group = grouped[label]

        if not group:
            continue

        print(f"\n--- {label} ARTICLES ({len(group)}) ---")

        for article in group:
            print_article(article, article_number)
            article_number += 1

    print(f"\n{'=' * 78}\n")


def print_debug_summary(raw_count, unique_count, final_count):
    """Print quick processing statistics."""
    print("\nProcessing summary:")
    print(f"    Raw articles fetched        : {raw_count}")
    print(f"    After duplicate filtering   : {unique_count}")
    print(f"    Final articles displayed    : {final_count}")


# Main
def main():
    """Run the full news aggregation pipeline."""
    api_key = input("Enter your NewsAPI key: ").strip()
    topic = input("Enter a topic to search: ").strip()

    if not api_key:
        print("Error: API key cannot be empty.")
        return

    if not topic:
        print("Error: topic cannot be empty.")
        return

    raw_articles = fetch_headlines(topic, api_key)
    unique_articles = remove_duplicates(raw_articles)
    ranked_articles = sort_articles(unique_articles, topic)
    final_articles = add_subtopics(ranked_articles, topic)

    print_debug_summary(
        raw_count=len(raw_articles),
        unique_count=len(unique_articles),
        final_count=len(final_articles)
    )

    print_report(topic, final_articles)


if __name__ == "__main__":
    main()

