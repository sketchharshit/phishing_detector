"""
utils.py

Text cleaning and feature extraction helpers for the phishing detector.
Kept in one place so train_model.py and app.py always use the exact same
preprocessing - if they drift, the saved model stops matching what the app
feeds it at prediction time.
"""

import re

# Standard English stopwords, kept as a plain list instead of an NLTK
# download so the app doesn't need internet access at runtime.
STOPWORDS = set('''
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
'''.split())

SUSPICIOUS_TLDS = (".tk", ".xyz", ".top", ".club", ".online", ".support",
                   ".site", ".ru", ".info", ".co")

URGENCY_TERMS = ["urgent", "immediately", "suspend", "verify", "action required",
                 "final notice", "limited", "click here", "confirm", "expire"]

# Real emails in this dataset sometimes write links as "http : / / www . x . com"
# (spaces around every slash/dot) - whether that's a deliberate filter-evasion
# trick or just an artifact of how the corpus was assembled, a plain
# "http\S+" pattern misses it entirely. Allowing optional whitespace around
# the protocol punctuation catches both the normal and the spaced-out form.
# Combined into one pattern (rather than two separate ones summed together)
# so a "http://www...." URL - which matches both alternatives - isn't counted twice.
URL_RE = re.compile(r"https?\s*:\s*/\s*/\s*\S+|\bwww\s*\.\s*\S+", re.IGNORECASE)

# A handful of rows have mangled non-ASCII text (accented characters from
# non-English spam, double-mis-encoded into repeating "ï¿½" sequences). Left
# in, these show up as meaningless high-frequency "words" after cleaning.
MOJIBAKE_RE = re.compile(r"(?:\xef\xbf\xbd|ï¿½)+")


def clean_text(text):
    """Lowercase, strip HTML/URLs/punctuation/digits, drop stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = MOJIBAKE_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def has_url(text):
    return int(bool(URL_RE.search(str(text))))


def count_urls(text):
    return len(URL_RE.findall(str(text)))


def suspicious_domain(sender):
    sender = str(sender).lower()
    domain = sender.split("@")[-1] if "@" in sender else sender
    bad_tld = any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
    hyphen_brand = bool(re.search(r"-(security|verify|support|alert|billing|login|update)", domain))
    digit_swap = bool(re.search(r"[a-z]*0[a-z]*", domain))  # e.g. amaz0n
    return int(bad_tld or hyphen_brand or digit_swap)


def exclamation_count(text):
    return str(text).count("!")


def uppercase_word_ratio(text):
    words = str(text).split()
    if not words:
        return 0.0
    upper = [w for w in words if w.isupper() and len(w) > 1]
    return len(upper) / len(words)


def urgency_word_count(text):
    text_lower = str(text).lower()
    return sum(text_lower.count(term) for term in URGENCY_TERMS)


METADATA_COLS = ["has_url", "url_count", "exclamation_count",
                  "uppercase_ratio", "urgency_score", "email_length"]


def extract_metadata(raw_text):
    """Build the metadata feature vector for one email's text.

    Note: this dataset (Kaggle "Phishing Email Detection") is just raw
    email text with no separate sender/from field, so there's no reliable
    way to compute a sender-domain feature for training. `suspicious_domain()`
    above is still available standalone - the app uses it as a separate,
    informational check when a user provides a sender address, but it is
    not one of the features the trained models see.
    """
    return [
        has_url(raw_text),
        count_urls(raw_text),
        exclamation_count(raw_text),
        uppercase_word_ratio(raw_text),
        urgency_word_count(raw_text),
        len(str(raw_text)),
    ]
