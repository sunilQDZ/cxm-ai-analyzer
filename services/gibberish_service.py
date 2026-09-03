import re
from typing import Dict, List, Optional


def is_gibrish_comment(text, category_mapping: Optional[Dict[str, List[str]]] = None) -> int:
    """
    Universal Multi-Industry VOC & Gibberish Detector.
    Supports ALL domains: Logistics, Healthcare/Medical, Education, E-Commerce, Retail, Telecom, Finance, Real Estate, etc.
    Returns: 1 if comment is gibberish/unreadable/invalid, 0 if valid readable VOC.
    """
    if text is None:
        return 1
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip()
    if not cleaned:
        return 1

    # 1. Check for Non-ASCII / Indic Scripts (Devanagari, Tamil, Telugu, Bengali, Gujarati, etc.)
    # Indic unicode ranges: \u0900-\u0DFF
    if re.search(r'[\u0900-\u0DFF]', cleaned):
        non_ascii_clean = re.sub(r'[^\w\s]', '', cleaned).strip()
        if len(non_ascii_clean) >= 2:
            return 0

    lower = cleaned.lower()

    # 2. Universal Cross-Industry & Multi-Domain Vocabulary
    universal_words = {
        # General Service & Support
        "service", "support", "help", "staff", "team", "agent", "executive", "manager",
        "customer", "care", "call", "response", "reply", "email", "portal", "website",
        "app", "login", "password", "otp", "account", "issue", "problem", "delay",
        "slow", "bad", "poor", "good", "great", "excellent", "nice", "best",
        "worst", "rude", "complaint", "feedback", "query", "request", "status",
        "update", "pending", "resolved", "cancel", "process", "time", "thank", "thanks",

        # Logistics & Supply Chain
        "shipment", "package", "courier", "parcel", "delivery", "dispatch", "order",
        "tracking", "awb", "address", "consignment", "transit", "damaged", "received",

        # Medical & Healthcare
        "doctor", "hospital", "clinic", "appointment", "medicine", "prescription",
        "patient", "consultation", "test", "report", "health", "insurance", "claim",

        # Education & EdTech
        "student", "teacher", "class", "exam", "course", "fee", "fees", "admission",
        "school", "college", "lecture", "certificate", "marks", "result", "batch",

        # E-Commerce, Retail & Telecom
        "product", "item", "refund", "return", "replacement", "cart", "payment",
        "invoice", "bill", "broadband", "network", "speed", "sim", "plan", "recharge",

        # Finance, Banking & Real Estate
        "bank", "loan", "emi", "kyc", "foreclosure", "noc", "interest", "amount",
        "money", "branch", "document", "verification", "sanction", "approval",

        # Hinglish & Regional Terms
        "karo", "nahi", "aaya", "hua", "mera", "meri", "mere", "kab", "tak",
        "batao", "kripya", "paisa", "paise", "kaise", "hoga", "chahiye",
        "gaya", "gayi", "kuch", "bohot", "bahut", "theek", "bekar", "accha"
    }

    # Dynamically include DB category tokens if category_mapping is provided
    if category_mapping:
        for cat, subs in category_mapping.items():
            for t in re.findall(r'[a-zA-Z]{3,}', cat.lower()):
                universal_words.add(t)
            for sub in subs:
                for t in re.findall(r'[a-zA-Z]{3,}', sub.lower()):
                    universal_words.add(t)

    tokens = re.findall(r'[a-zA-Z]{2,}', lower)

    # 3. Known dummy / keyboard mashing patterns
    dummy_patterns = {
        "asdf", "asdfg", "asdfgh", "asdfghj", "asdfghjkl",
        "qwerty", "qwertyu", "qwertyuiop", "zxcv", "zxcvb", "zxcvbnm",
        "qazwsx", "123456", "abcdef", "test", "testing", "dummy", "demo",
        "null", "none", "na", "n/a", "temp", "check", "hello", "hi", "ok", "okay"
    }

    meaningful_tokens = [w for w in tokens if w not in dummy_patterns and (len(w) >= 3 or w in {"ok", "no", "hi"})]

    if any(w in universal_words for w in tokens):
        if not (len(tokens) == 1 and tokens[0] in dummy_patterns):
            return 0

    # 4. Check for repetitive single character (e.g. "aaaaaa", "11111111", "000000")
    if len(cleaned) >= 4:
        max_char_rep = max(cleaned.count(c) for c in set(cleaned))
        if max_char_rep / len(cleaned) > 0.60 and len(set(cleaned)) <= 2:
            return 1
        if re.search(r'(.)\1{3,}', lower):
            if not any(w in universal_words for w in tokens):
                return 1

    # 5. Check alphanumeric character presence
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    if alnum_count < 2:
        return 1

    # 6. Check for purely numeric dummy sequences
    digit_count = sum(1 for c in cleaned if c.isdigit())
    if digit_count >= 5 and alnum_count == digit_count:
        dummy_num_patterns = ["12345", "67890", "01234", "98765", "54321"]
        if any(p in cleaned for p in dummy_num_patterns) or len(set(cleaned)) <= 2:
            return 1
        return 0

    alpha_chars = [c.lower() for c in cleaned if c.isalpha()]

    # 7. Keyboard sequence patterns
    kb_sequences = ["asdf", "qwer", "zxcv", "hjkl", "yuio", "12345", "67890", "abcd"]
    if any(seq in lower for seq in kb_sequences):
        if not any(w in universal_words for w in meaningful_tokens):
            return 1

    # 8. Short word whitelist
    if len(lower) <= 4:
        valid_shorts = {
            "ok", "yes", "no", "good", "bad", "help", "poor", "nice", "fine",
            "fast", "slow", "late", "emi", "kyc", "noc", "loan", "bill", "card",
            "done", "wait", "call", "rate", "cost", "safe", "true", "scam"
        }
        if lower in valid_shorts or any(w in valid_shorts for w in tokens):
            return 0
        return 1

    # 9. Long continuous word without spaces
    words = cleaned.split()
    if len(words) == 1 and len(cleaned) > 16:
        if not any(w in universal_words for w in tokens):
            return 1

    # 10. Vowel ratio & consonant sequence check for English/Latin alphabetic text >= 5 letters
    if len(alpha_chars) >= 5:
        vowels = set("aeiou")
        vowel_count = sum(1 for c in alpha_chars if c in vowels)
        consonant_count = len(alpha_chars) - vowel_count

        if vowel_count == 0 and len(alpha_chars) >= 4:
            valid_no_vowel = {"why", "gym", "sky", "try", "fly", "dry", "shh", "sms", "kyc", "noc", "otp", "rbi", "fir", "awb"}
            if not any(t in valid_no_vowel for t in tokens):
                return 1

        consonant_ratio = consonant_count / len(alpha_chars)
        if consonant_ratio > 0.88 and len(alpha_chars) >= 7:
            if not any(w in universal_words for w in tokens):
                return 1

        if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', lower):
            if not any(w in universal_words for w in tokens):
                return 1

    # 11. High non-alphanumeric symbol ratio
    symbol_count = sum(1 for c in cleaned if not c.isalnum() and not c.isspace())
    if symbol_count / len(cleaned) > 0.40:
        if not any(w in universal_words for w in tokens):
            return 1

    if meaningful_tokens:
        return 0

    return 1
