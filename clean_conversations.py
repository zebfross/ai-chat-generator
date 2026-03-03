"""Filter tawk_conversations.json to remove non-answers and junk."""
import json
import re


def is_non_answer(response):
    """Return True if the agent response doesn't actually answer anything."""
    r = response.lower().strip()

    # Too short to be useful
    if len(r) < 15:
        return True

    # Strip leading greeting to see if there's substance after it
    # e.g. "Hi Bryan!  Our tradelines cost..." -> "Our tradelines cost..."
    substance = re.sub(
        r"^(hi|hello|hey|howdy|greetings)\s+\w+[,!.\s]*", "", r
    ).strip()
    # Also strip trailing pleasantries
    substance = re.sub(
        r"(how can i help|how may i help|what can i help|how can i assist|what can i do for you)[?\s!]*$",
        "", substance
    ).strip()

    # If nothing left after stripping greeting + "how can I help", it's a non-answer
    if len(substance) < 10:
        return True

    # Pure deflection phrases with no real info
    deflection_only = [
        r"^i('d| would) (love|be happy|be glad) to help[!.\s]*$",
        r"^i can (definitely |certainly )?help[!.\s]*$",
        r"^(nice|great|awesome|wonderful) to (meet|hear from) you[!.\s]*$",
        r"^thanks for the info[!.\s]*$",
        r"^thank you for (reaching out|contacting|your message)[!.\s]*$",
        r"^(great|ok|ok\.|sure|absolutely|of course|perfect|that's perfect)[!.\s]*$",
        r"^that's (a great|a good) question[!.\s]*$",
        r"^(ok|thanks)\.? (thanks )?for the (info|additional details|information)[!.\s]*$",
        r"^we can do that[!.\s]*$",
        r"^you too\.? thanks[!.\s]*$",
        r"^it's my pleasure[!😇.\s]*$",
        r"^tradeline works[!.\s]*$",
        r"^that's what we specialize in \w+[!.\s]*$",
        r"^tommy, i would love to help[!.\s]*$",
    ]
    for pattern in deflection_only:
        if re.match(pattern, substance):
            return True

    # Scheduling / internal ops / account-specific (too context-dependent)
    noisy_patterns = [
        r"scheduled a call", r"i'll text you", r"i'll call you",
        r"what('s| is) your time zone", r"what time zone",
        r"what's the best number", r"save our (phone )?number",
        r"we'll (reach out|call)", r"i'll schedule",
        r"is now a good time", r"let me check",
        r"is this the best phone", r"can you please (save|do me)",
        r"your username is", r"your email is verified",
        r"it looks like you're signed in",
        r"please give me a moment", r"perfect! i'll call",
        r"we have \d+.*(am|pm|est|cst|pst)", r"available (in|for|at) about",
    ]
    for pattern in noisy_patterns:
        if re.search(pattern, r):
            return True

    return False


def is_junk_question(question):
    """Return True if the customer message isn't a real question."""
    q = question.lower().strip()

    # Too short to be useful
    if len(q) < 8:
        return True

    # Just a greeting
    if q in ("hi", "hello", "hey", "help", "hi there", "hi there,",
             "hello!", "hey!", "hello there", "hi!", "good morning",
             "good afternoon", "good evening"):
        return True

    # Form fill (Name : ...\nEmail : ...)
    if q.startswith("name :") or ("email :" in q and "phone :" in q):
        return True

    # Just a phone number or email
    if re.match(r"^[\d\s\-\(\)\+]+$", q):
        return True
    if re.match(r"^[\w\.\-]+@[\w\.\-]+\.\w+$", q):
        return True

    # Just "ok", "thanks", "yes", "no", etc.
    if q in ("ok", "ok.", "okay", "thanks", "thanks!", "thank you",
             "yes", "no", "sure", "alright", "got it", "cool",
             "# please", "car", "yes no problem!", "ok thank u"):
        return True

    # Context-dependent fragments that make no sense standalone
    context_patterns = [
        r"^(ok )?(thanks?|thank you)( (so much|very much))?[!.\s]*$",
        r"^(yes|yeah|yep|yea|no|nope)[,!.\s]*$",
        r"^(ok|okay|sure|alright|great|perfect)[!.\s]*$",
        r"^or tomorrow",
        r"^(typo|stnadard|standard)\??$",
        r"^(9am|10am|2pm|3pm)\s",
        r"^a what time$",
        r"^yes (do|to|now|but)",
    ]
    for pattern in context_patterns:
        if re.match(pattern, q):
            return True

    return False


if __name__ == "__main__":
    with open("tawk_conversations.json") as f:
        convos = json.load(f)

    print(f"Starting conversations: {len(convos)}")

    cleaned = []
    reasons = {"non_answer": 0, "junk_question": 0, "duplicate": 0}
    seen = set()

    for c in convos:
        q = c["chat_message"].strip()
        a = c["response"].strip()

        if is_junk_question(q):
            reasons["junk_question"] += 1
            continue

        if is_non_answer(a):
            reasons["non_answer"] += 1
            continue

        # Deduplicate by (question, response) pair
        key = (q.lower(), a.lower())
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)

        cleaned.append(c)

    print(f"\nRemoved:")
    for reason, count in reasons.items():
        print(f"  {count:4d}  {reason}")
    print(f"\nKept: {len(cleaned)} conversations")

    with open("tawk_conversations_clean.json", "w") as f:
        json.dump(cleaned, f, indent=2)
    print("Saved to tawk_conversations_clean.json")

    # Show samples
    print("\nSamples:")
    for c in cleaned[:10]:
        print(f"\n  Q: {c['chat_message'][:80]}")
        print(f"  A: {c['response'][:100]}")
