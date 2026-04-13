from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------
# Abbreviations / nicknames → canonical state name  (always HIGH confidence)
# ----------------------------------------------------------------
_STATE_ALIASES: Dict[str, str] = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "cali": "California", "calif": "California",
    "co": "Colorado", "conn": "Connecticut", "ct": "Connecticut",
    "de": "Delaware", "fl": "Florida", "fla": "Florida",
    "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "ill": "Illinois", "in": "Indiana", "ind": "Indiana",
    "ia": "Iowa", "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana",
    "me": "Maine", "md": "Maryland", "ma": "Massachusetts", "mass": "Massachusetts",
    "mi": "Michigan", "mich": "Michigan", "mn": "Minnesota", "minn": "Minnesota",
    "ms": "Mississippi", "mo": "Missouri", "mt": "Montana",
    "ne": "Nebraska", "neb": "Nebraska", "nv": "Nevada", "nev": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico",
    "ny": "New York", "nc": "North Carolina", "nd": "North Dakota",
    "oh": "Ohio", "ok": "Oklahoma", "okla": "Oklahoma",
    "or": "Oregon", "ore": "Oregon",
    "pa": "Pennsylvania", "penn": "Pennsylvania", "penna": "Pennsylvania",
    "ri": "Rhode Island", "sc": "South Carolina", "sd": "South Dakota",
    "tn": "Tennessee", "tenn": "Tennessee",
    "tx": "Texas", "tex": "Texas",
    "ut": "Utah", "vt": "Vermont", "va": "Virginia",
    "wa": "Washington", "wash": "Washington",
    "wv": "West Virginia", "wi": "Wisconsin", "wis": "Wisconsin",
    "wy": "Wyoming", "dc": "District of Columbia",
    "socal": "California", "norcal": "California",
}

_STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "in", "of", "for", "and", "or",
    "how", "many", "show", "me", "give", "all", "state", "states", "county",
    "counties", "data", "population", "rent", "burden", "rate", "ratio",
    "percent", "percentage", "number", "count", "total", "most", "least",
    "top", "bottom", "highest", "lowest", "more", "less", "than", "which",
    "where", "who", "much", "average", "avg", "sum", "list", "find", "get",
    "compare", "comparison", "between", "about", "with", "has", "have",
    "does", "do", "severe", "burdened", "renters", "renter", "housing",
    "units", "census", "block", "group", "tract", "city", "town", "area",
    "region", "national", "across", "per", "each", "every", "by", "from",
}

# Confidence levels
HIGH = "HIGH"
MEDIUM = "MEDIUM"

# difflib score thresholds
_HIGH_CUTOFF = 0.88
_MEDIUM_CUTOFF = 0.70


class FuzzyLocationResolver:
    """
    Offline fuzzy resolver for US state and county names with confidence scoring.

    Returns HIGH confidence when the match is unambiguous:
      - alias/abbreviation dict
      - exact match (case-insensitive)
      - unique prefix match (token ≥ 4 chars)
      - difflib score ≥ 0.88

    Returns MEDIUM confidence when the match is plausible but uncertain:
      - unique prefix match (token < 4 chars)
      - difflib score 0.70 – 0.87

    The orchestrator should:
      - Silently use HIGH matches (append to question as context)
      - Pause and ask the user to confirm MEDIUM matches before proceeding
    """

    def __init__(self, state_names: List[str], county_names: List[str]) -> None:
        self._states = state_names
        self._state_lower: Dict[str, str] = {s.lower(): s for s in state_names}

        self._counties = county_names
        self._county_lower: Dict[str, str] = {c.lower(): c for c in county_names}

        self._county_short: Dict[str, str] = {}
        for c in county_names:
            m = re.match(r"^(.+?)\s+County\b", c, re.IGNORECASE)
            if m:
                short = m.group(1).strip().lower()
                self._county_short.setdefault(short, c)
            before_comma = c.split(",")[0].strip().lower()
            self._county_short.setdefault(before_comma, c)

        self._state_keys = list(self._state_lower.keys())
        self._county_short_keys = list(self._county_short.keys())

    # ------------------------------------------------------------------

    def resolve_token(self, token: str) -> Optional[Tuple[str, str, str]]:
        """
        Try to resolve a single token to (loc_type, canonical_name, confidence).
        loc_type is 'state' or 'county'.  confidence is HIGH or MEDIUM.
        Returns None if no match meets the minimum threshold.
        """
        t = token.strip().lower()
        if not t or t in _STOP_WORDS or len(t) < 2:
            return None

        # 1. Alias dict → always HIGH
        if t in _STATE_ALIASES:
            return ("state", _STATE_ALIASES[t], HIGH)

        # 2. Exact state match → HIGH
        if t in self._state_lower:
            return ("state", self._state_lower[t], HIGH)

        # 3. State prefix — unique
        prefix_states = [s for s in self._states if s.lower().startswith(t)]
        if len(prefix_states) == 1:
            conf = HIGH if len(t) >= 4 else MEDIUM
            return ("state", prefix_states[0], conf)

        # 4. County short-name exact → HIGH
        if t in self._county_short:
            return ("county", self._county_short[t], HIGH)

        # 5. County short-name prefix — unique
        prefix_counties = [v for k, v in self._county_short.items() if k.startswith(t)]
        if len(prefix_counties) == 1:
            conf = HIGH if len(t) >= 4 else MEDIUM
            return ("county", prefix_counties[0], conf)

        # 6. difflib fuzzy (only tokens ≥ 4 chars to suppress noise)
        if len(t) >= 4:
            close_state = difflib.get_close_matches(
                t, self._state_keys, n=1, cutoff=_MEDIUM_CUTOFF
            )
            if close_state:
                score = difflib.SequenceMatcher(None, t, close_state[0]).ratio()
                conf = HIGH if score >= _HIGH_CUTOFF else MEDIUM
                return ("state", self._state_lower[close_state[0]], conf)

            close_county = difflib.get_close_matches(
                t, self._county_short_keys, n=1, cutoff=_MEDIUM_CUTOFF
            )
            if close_county:
                score = difflib.SequenceMatcher(None, t, close_county[0]).ratio()
                conf = HIGH if score >= _HIGH_CUTOFF else MEDIUM
                return ("county", self._county_short[close_county[0]], conf)

        return None

    # ------------------------------------------------------------------

    def augment_question(
        self, question: str
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """
        Scan the question for location tokens, split matches by confidence.

        Returns:
          augmented_question  — original question with HIGH-confidence resolutions
                                 appended as a [location context: ...] annotation
          clarifications      — list of (original_token, canonical_name, loc_type)
                                 for MEDIUM-confidence matches that need user confirmation
        """
        words = re.findall(r"[a-zA-Z]+", question)
        candidates: List[str] = list(words)
        for i in range(len(words) - 1):
            candidates.append(f"{words[i]} {words[i + 1]}")

        seen: set = set()
        high_matches: List[Tuple[str, str, str]] = []   # (original, canonical, type)
        medium_matches: List[Tuple[str, str, str]] = []

        for token in candidates:
            tl = token.lower()
            if tl in _STOP_WORDS or tl in seen:
                continue
            result = self.resolve_token(token)
            if not result:
                continue
            loc_type, canonical, conf = result
            seen.add(tl)
            if conf == HIGH:
                high_matches.append((token, canonical, loc_type))
            else:
                medium_matches.append((token, canonical, loc_type))

        # Build augmented question from HIGH matches only
        augmented = question
        if high_matches:
            annotations = [f"{canonical} ({loc_type})" for _, canonical, loc_type in high_matches]
            augmented = question + f"  [location context: {', '.join(annotations)}]"

        return augmented, medium_matches
