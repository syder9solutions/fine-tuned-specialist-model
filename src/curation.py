"""Intent-labelled corpus curation.

Curation is 80% of a fine-tune's outcome, so this module does that work
mechanically and reproducibly:

  1. hand-written seed templates per intent (terse, real gateway register)
  2. deterministic paraphrase augmentation with slot substitution
  3. confusers: requests that carry intent vocabulary but must NOT be
     auto-routed — ambiguous, multi-intent, and near-miss. They all map to the
     `escalate` label; their subtype rides along as `kind` for error analysis.
  4. normalisation + dedupe across the whole pool
  5. leakage-safe stratified split with a hard disjointness assertion

Everything is seeded, so the corpus is byte-identical across runs (digests are
recorded in curation_report.json).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.config import (
    CONFUSER_KIND_SET,
    DATA_CONFIG,
    ESCALATE_LABEL,
    INTENT_LABELS,
)

# ---------------------------------------------------------------------------
# Seed templates per intent.
# ---------------------------------------------------------------------------
SEED_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "summarize": (
        "summarize this article for me",
        "give me a tl;dr of the meeting notes",
        "condense these {n} pages into bullet points",
        "shorten this report keeping the key findings",
        "what are the main takeaways from this doc",
        "boil this email thread down to two sentences",
        "make an executive summary of the quarterly deck",
        "sum up the customer feedback we got this week",
        "abstract this paper in plain english",
        "recap what happened in this support thread",
    ),
    "translate": (
        "translate this paragraph into {lang}",
        "how do you say good morning in {lang}",
        "convert this contract clause to {lang}",
        "give me the {lang} version of this message",
        "what does this {lang} sentence mean in english",
        "localize this landing page copy for {lang}",
        "rewrite this reply in polite {lang}",
        "put this status update into {lang}",
        "interpret this customer email from {lang}",
        "render this notice in {lang} please",
    ),
    "qa": (
        "what is the capital of {country}",
        "why does http 502 happen behind a proxy",
        "how many employees does {company} have",
        "explain the difference between tcp and udp",
        "who wrote the paper on attention is all you need",
        "what year did {company} go public",
        "is it safe to mix bleach and ammonia",
        "how tall is mount {mountain}",
        "what causes condensation on cold pipes",
        "define idempotency in one sentence",
    ),
    "retrieval": (
        "find the doc where we discussed the pricing change",
        "search our wiki for the on-call runbook",
        "which ticket mentioned the {company} outage",
        "pull up the contract from last {month}",
        "look through my notes for the api key rotation steps",
        "where did we save the q3 roadmap deck",
        "show me the email thread about the data migration",
        "locate the schema definition for the orders table",
        "grep the repo for the retry helper",
        "find every mention of the deprecated endpoint",
    ),
    "code": (
        "write a python function that parses {data}",
        "fix this traceback in my django view",
        "refactor this loop to use a dict comprehension",
        "why is my sql join returning duplicate rows",
        "add type hints to this function",
        "convert this bash script to python",
        "implement debounce in vanilla javascript",
        "review this diff for off by one errors",
        "write a regex that matches {data}",
        "make this unit test cover the error path",
    ),
    "math": (
        "what is {pct}% of {num}",
        "solve for x in {num}x + {num2} = {num3}",
        "compound interest on {num}k at {pct}% for five years",
        "integrate x squared from zero to {num}",
        "how many combinations of {num} choose {num2}",
        "if a train leaves at {num} going {pct} kmh how far in two hours",
        "round {num}.{num2} to two decimals",
        "whats the derivative of ln of x",
        "split {num} evenly between {num2} people",
        "convert {num} fahrenheit to celsius",
    ),
    "creative": (
        "write a haiku about {topic}",
        "give me a slogan for a {topic} startup",
        "draft a short story opening set in {place}",
        "name ideas for a {topic} brand",
        "write a toast for a {topic} launch party",
        "compose a two line rhyme about {topic}",
        "pitch me three {topic} video concepts",
        "describe {place} as if it were a character",
        "write a playful riddle about {topic}",
        "invent a cocktail inspired by {place}",
    ),
    "smalltalk": (
        "hey hows it going",
        "thanks that was helpful",
        "good morning are you around",
        "lol fair enough",
        "hope you had a good weekend",
        "you there",
        "no worries talk later",
        "appreciate it buddy",
        "sorry i was away for a bit",
        "have a good one",
    ),
}

SLOTS: Dict[str, Tuple[str, ...]] = {
    "n": ("two", "three", "five", "ten", "twelve"),
    "lang": ("spanish", "french", "german", "japanese", "arabic", "urdu", "portuguese"),
    "country": ("peru", "norway", "kenya", "vietnam", "uruguay", "iceland"),
    "company": ("acme", "globex", "initech", "umbrella", "hooli", "wayne"),
    "mountain": ("kilimanjaro", "denali", "fuji", "elbrus", "etna"),
    "month": ("march", "june", "september", "november"),
    "data": ("csv files", "yaml configs", "iso dates", "jwt tokens", "user agents"),
    "pct": ("12", "7", "18", "23", "35", "4"),
    "num": ("12", "48", "120", "250", "9"),
    "num2": ("3", "5", "7", "11"),
    "num3": ("40", "96", "300"),
    "topic": ("coffee", "the ocean", "cycling", "a startup", "autumn", "robots"),
    "place": ("lisbon", "karachi", "reykjavik", "a small port town", "the desert"),
}

# ---------------------------------------------------------------------------
# Confusers — all labelled `escalate`, distinguished by `kind`.
# ---------------------------------------------------------------------------
CONFUSERS: Dict[str, Tuple[str, ...]] = {
    # no actionable intent at all
    "ambiguous": (
        "can you handle this for me",
        "do the thing we talked about",
        "same as last time please",
        "just make it better",
        "you know what i mean",
        "sort it out",
        "whatever you think is best",
        "the usual",
        "take care of that",
        "fix it up",
        "deal with this",
        "you know the drill",
    ),
    # two or more intents in one request -> must not pick one silently
    "multi_intent": (
        "summarize this article and then translate the summary to spanish",
        "find the pricing doc and also explain what changed",
        "translate this email and draft a reply in french",
        "search the wiki for the runbook then write a bash script for it",
        "explain http 502 and write a patch for the proxy config",
        "give me a haiku and then solve the equation on the slide",
        "whats our revenue and summarize the board deck",
        "find the ticket and update it with a summary",
        "review this diff and explain the math behind the bug",
        "say hi and then pull up last quarters roadmap",
        "count the rows and write a sql query to dedupe them",
        "translate the changelog and summarize it for execs",
    ),
    # intent word used as a noun/entity/quantity, not a request
    "near_miss_lexical": (
        "summarize module is throwing an import error",
        "the translate endpoint returned 500",
        "qa environment is down again",
        "retrieval service latency spiked overnight",
        "code review meeting moved to thursday",
        "math library version mismatch",
        "creative team wants new fonts",
        "smalltalk channel is archived now",
        "summarize.ai raised a series a",
        "translate.com is down",
        "summarize 3 things",
        "translate 2 sentences",
        "qa 5 builds",
        "retrieval of 4 records",
        "code 6 lines",
    ),
    # explicitly negated intent
    "near_miss_negation": (
        "do not summarize this yet",
        "no translation needed here",
        "dont run the qa suite",
        "never search the wiki for that",
        "no code changes please",
        "not a math question",
        "skip the creative writing part",
        "no smalltalk just give me the data",
        "you dont need to summarize anything",
        "without translating it just tell me the gist",
        "do not retrieve the old records",
        "no need to write any code for this",
    ),
    # intent word as a person/tool name
    "near_miss_entity": (
        "summarize is our new intern",
        "translate is the name of the vendor",
        "qa is the person who owns the release",
        "retrieval is on vacation this week",
        "code is a nickname for the team lead",
        "math is the contractor we hired",
        "creative is the agency we use",
        "smalltalk is her podcast",
        "summarize left the company",
        "translate pinged me earlier",
        "qa said they will sign off",
        "creative billed us twice",
    ),
}


@dataclass(frozen=True)
class CurationResult:
    train: List[Dict]
    test: List[Dict]
    stats: Dict[str, object]


def normalize(text: str) -> str:
    """Canonical form used for dedupe and leakage checks."""
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    return text


def fill_slots(template: str, rng: random.Random) -> str:
    """Deterministically fill {slot} placeholders, each occurrence independently."""
    def repl(match: re.Match) -> str:
        name = match.group(1)
        pool = SLOTS.get(name)
        if not pool:
            raise KeyError(f"unknown slot {name!r} in template {template!r}")
        return rng.choice(pool)

    return re.sub(r"\{(\w+)\}", repl, template)


def _paraphrase(text: str, rng: random.Random) -> str:
    """Deterministic register variation — no semantic change."""
    style = rng.randrange(5)
    if style == 0:
        return f"hey can you {text}"
    if style == 1:
        return f"{text} please"
    if style == 2:
        return f"i need you to {text}"
    if style == 3:
        return f"{text} thanks"
    return f"quick one: {text}"


def build_corpus(
    *,
    seed: int = DATA_CONFIG.seed,
    paraphrases_per_seed: int = DATA_CONFIG.paraphrases_per_seed,
    confusers_per_kind: int = DATA_CONFIG.confusers_per_kind,
    test_ratio: float = DATA_CONFIG.test_ratio,
) -> CurationResult:
    """Build the curated corpus and a leakage-safe split."""
    if not 0.0 < test_ratio < 0.9:
        raise ValueError(f"test_ratio must be in (0, 0.9), got {test_ratio}")
    if paraphrases_per_seed < 1:
        raise ValueError("paraphrases_per_seed must be >= 1")
    rng = random.Random(seed)

    # 1. intent rows: each template filled, plus paraphrases. `family` identifies the
#    template so the split can hold out whole families (paraphrases of one
#    template are near-duplicates by construction — row-level splitting leaks).
    intent_rows: List[Dict] = []
    for label in INTENT_LABELS:
        for t_idx, template in enumerate(SEED_TEMPLATES[label]):
            family = f"{label}::t{t_idx}"
            base = fill_slots(template, rng)
            intent_rows.append(
                {"text": base, "label": label, "kind": "intent", "family": family}
            )
            for _ in range(paraphrases_per_seed - 1):
                variant = _paraphrase(fill_slots(template, rng), rng)
                intent_rows.append(
                    {"text": variant, "label": label, "kind": "intent", "family": family}
                )

    # 2. confuser rows: cycle the kind's pool, paraphrase to hit the volume.
    #    Each confuser template is its own family for the same reason.
    confuser_rows: List[Dict] = []
    for kind in sorted(CONFUSER_KIND_SET):
        pool = CONFUSERS[kind]
        produced = 0
        i = 0
        while produced < confusers_per_kind:
            t_idx = i % len(pool)
            template = pool[t_idx]
            family = f"escalate::{kind}::t{t_idx}"
            text = fill_slots(template, rng)
            if i >= len(pool):
                text = _paraphrase(text, rng)
            confuser_rows.append(
                {
                    "text": text,
                    "label": ESCALATE_LABEL,
                    "kind": kind,
                    "family": family,
                }
            )
            produced += 1
            i += 1

    # 3. normalise + dedupe (first occurrence wins; order deterministic)
    seen: set = set()
    deduped: List[Dict] = []
    dropped = 0
    for row in intent_rows + confuser_rows:
        key = normalize(row["text"])
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)

    # 4. stratified FAMILY-LEVEL split: a template family goes wholly to one side.
    #    Splitting by row would put paraphrases of the same template on both
    #    sides, which is leakage the model can exploit. Grouping by
    #    (label, kind) — not label alone — keeps every confuser subtype
    #    represented in the test set, so the escalation metrics have support.
    by_label: Dict[Tuple[str, str], Dict[str, List[Dict]]] = {}
    for row in deduped:
        key = (row["label"], row["kind"])
        by_label.setdefault(key, {}).setdefault(row["family"], []).append(row)

    train: List[Dict] = []
    test: List[Dict] = []
    for key in sorted(by_label):
        families = sorted(by_label[key])
        rng.shuffle(families)
        n_test_fams = max(1, round(test_ratio * len(families)))
        n_test_fams = min(n_test_fams, max(1, len(families) - 1))
        test_fams = set(families[:n_test_fams])
        for family in families:
            target = test if family in test_fams else train
            target.extend(by_label[key][family])

    rng.shuffle(train)
    rng.shuffle(test)

    # 5. hard disjointness assertion — the leakage guarantee
    train_keys = {normalize(r["text"]) for r in train}
    test_keys = {normalize(r["text"]) for r in test}
    if train_keys & test_keys:
        raise AssertionError(
            f"train/test leakage: {len(train_keys & test_keys)} shared rows"
        )
    if not train or not test:
        raise AssertionError("split produced an empty side")

    stats = {
        "seed": seed,
        "paraphrases_per_seed": paraphrases_per_seed,
        "confusers_per_kind": confusers_per_kind,
        "raw_rows": len(intent_rows) + len(confuser_rows),
        "deduped_rows": len(deduped),
        "dropped_duplicates": dropped,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_ratio_actual": round(len(test) / len(deduped), 4),
        "label_counts_train": _counts(train, "label"),
        "label_counts_test": _counts(test, "label"),
        "kind_counts_train": _counts(train, "kind"),
        "kind_counts_test": _counts(test, "kind"),
        "train_sha256": _digest(train),
        "test_sha256": _digest(test),
    }
    return CurationResult(train=train, test=test, stats=stats)


def _counts(rows: Sequence[Dict], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        out[row[field]] = out.get(row[field], 0) + 1
    return dict(sorted(out.items()))


def _digest(rows: Sequence[Dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update((row["label"] + "\t" + normalize(row["text"]) + "\n").encode())
    return h.hexdigest()[:16]


def write_corpus(out_dir: Path, *, seed: int = DATA_CONFIG.seed) -> Dict[str, object]:
    """Curate and write train.jsonl / test.jsonl / curation_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_corpus(seed=seed)
    for name, rows in (("train.jsonl", result.train), ("test.jsonl", result.test)):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(
                    json.dumps(
                        {
                            "text": row["text"],
                            "label": row["label"],
                            "kind": row["kind"],
                            "family": row["family"],
                        }
                    )
                    + "\n"
                )
    (out_dir / "curation_report.json").write_text(
        json.dumps(result.stats, indent=2) + "\n", encoding="utf-8"
    )
    return result.stats
