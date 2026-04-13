from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

from .models import LocationContext

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
    # Articles, prepositions, conjunctions
    "a", "an", "the", "in", "of", "for", "and", "or", "at", "on", "to",
    "into", "from", "by", "with", "about", "above", "below", "before",
    "after", "between", "per", "each", "every", "across", "through",
    # Pronouns
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those",
    # Auxiliary / common verbs — the primary false-positive source
    "is", "are", "was", "were", "be", "been", "being",
    "am", "have", "has", "had", "having",
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "do", "did", "does", "doing",
    "go", "going", "get", "getting", "make", "making",
    "ask", "asking", "asked", "tell", "telling", "told",
    "say", "saying", "said", "think", "know", "see", "use",
    "find", "give", "want", "look", "come", "take", "need", "try",
    # Question words
    "what", "how", "many", "which", "where", "who", "when", "why",
    "whose", "much",
    # Domain-specific common words (not location names)
    "show", "me", "give", "all", "state", "states", "county", "counties",
    "data", "population", "rent", "burden", "rate", "ratio",
    "percent", "percentage", "number", "count", "total",
    "most", "least", "top", "bottom", "highest", "lowest",
    "more", "less", "than", "average", "avg", "sum", "list",
    "compare", "comparison", "severe", "burdened",
    "renters", "renter", "housing", "units", "census", "block",
    "group", "tract", "city", "town", "area", "region", "national",
    # Common adverbs / misc
    "here", "there", "now", "just", "also", "even", "very", "too",
    "so", "up", "out", "if", "not", "but", "then", "when", "while",
    "again", "further", "once", "both", "only", "same", "such",
    "own", "other", "another", "any", "some", "no", "nor",
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


# ----------------------------------------------------------------
# Location state manager
# ----------------------------------------------------------------

def _extract_state_from_county(county_name: str) -> Optional[str]:
    """
    Extract state name from county strings like 'Los Angeles County, California'.
    Returns None if no comma-separated state suffix is present.
    """
    if "," in county_name:
        return county_name.split(",")[-1].strip()
    return None


class LocationStateManager:
    """
    Persistent location state for one chat session.

    Replaces ad-hoc fuzzy search on every message with an explicit
    LocationContext that is updated by a defined set of rules and
    carried forward across turns that mention no new location.

    Fuzzy search only runs for qa_agent route messages; conversational
    messages bypass this class entirely.

    Update rules:
      - HIGH-confidence match or user-confirmed MEDIUM:
          * New state mention → set state; reset county UNLESS the existing
            county belongs to that same state.
          * New county mention → set county; derive + set state from county
            name if possible.
      - No location found in new message → carry existing state forward,
        annotate the question silently.
      - MEDIUM match → surface as clarification; do NOT update state until
        the user confirms.
    """

    def __init__(self, resolver: FuzzyLocationResolver) -> None:
        self._resolver = resolver
        self.ctx = LocationContext()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_message(
        self, question: str
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """
        Scan a new user message for location tokens.

        Returns:
          resolved_question  — question annotated with the current LocationContext
                               (updated by any HIGH matches found)
          clarifications     — list of (original_token, canonical, loc_type)
                               for MEDIUM-confidence matches that need confirmation
        """
        high_matches, medium_matches = self._scan(question)

        # Update state for every HIGH match found in this message
        for _token, canonical, loc_type in high_matches:
            self._update(loc_type, canonical, "high_fuzzy")

        # Annotate question from the (now up-to-date) LocationContext.
        # If the message introduced HIGH matches they're already in ctx;
        # if no new location was found the existing ctx is carried forward.
        resolved_question = self._annotate(question)
        return resolved_question, medium_matches

    def confirm_medium(self, loc_type: str, canonical: str) -> None:
        """
        Called after the user confirms a MEDIUM-confidence match.
        Updates LocationContext with source='user_confirmed'.
        """
        self._update(loc_type, canonical, "user_confirmed")

    def annotate_from_state(self, question: str) -> str:
        """Annotate a question string using the current LocationContext only."""
        return self._annotate(question)

    def reset(self) -> None:
        """Clear the location state (e.g. when user starts a new topic)."""
        self.ctx = LocationContext()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan(
        self, question: str
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Tokenise question, apply stop-list, run resolver.
        Returns (high_matches, medium_matches) as (token, canonical, loc_type).
        """
        words = re.findall(r"[a-zA-Z]+", question)
        candidates: List[str] = list(words)
        for i in range(len(words) - 1):
            candidates.append(f"{words[i]} {words[i + 1]}")

        seen: set = set()
        high: List[Tuple[str, str, str]] = []
        medium: List[Tuple[str, str, str]] = []

        for token in candidates:
            tl = token.lower()
            if tl in _STOP_WORDS or tl in seen:
                continue
            result = self._resolver.resolve_token(token)
            if not result:
                continue
            loc_type, canonical, conf = result
            seen.add(tl)
            if conf == HIGH:
                high.append((token, canonical, loc_type))
            else:
                medium.append((token, canonical, loc_type))

        return high, medium

    def _update(self, loc_type: str, canonical: str, source: str) -> None:
        """Apply update rules to LocationContext."""
        confirmed = source in ("user_confirmed", "explicit")

        if loc_type == "state":
            if self.ctx.county:
                # Keep existing county only if it belongs to this state
                county_state = _extract_state_from_county(self.ctx.county)
                if not county_state or county_state.lower() != canonical.lower():
                    self.ctx.county = None  # Different state — drop old county
            self.ctx.state = canonical
            # Upgrade grain only if not already at county level for same state
            if not self.ctx.county:
                self.ctx.grain = "state"
            self.ctx.source = source
            self.ctx.confirmed = confirmed

        elif loc_type == "county":
            self.ctx.county = canonical
            # Derive state from county name when possible
            state_from_county = _extract_state_from_county(canonical)
            if state_from_county:
                self.ctx.state = state_from_county
            self.ctx.grain = "county"
            self.ctx.source = source
            self.ctx.confirmed = confirmed

    def _annotate(self, question: str) -> str:
        """Append current LocationContext as inline annotation if not already present."""
        if "[location context:" in question:
            return question
        annotation = self.ctx.as_annotation()
        if not annotation:
            return question
        return question + annotation
