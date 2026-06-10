from __future__ import annotations

import abc
import math
import re

from lc_shift.config import RouterConfig
from lc_shift.models import ShiftRequest

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")

_REASONING_KEYWORDS = frozenset({
    "analyze", "compare", "contrast", "explain", "evaluate",
    "synthesize", "critique", "assess", "justify", "derive",
    "prove", "reason", "why", "how does", "what if",
    "trade-off", "implications",
})

_MULTI_PART_RE = re.compile(
    r"\b\d+[.\)]\s|\b(?:first|second|third|finally|additionally)\b",
    re.IGNORECASE,
)


def estimate_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def compute_complexity(prompt: str) -> float:
    """Heuristic scoring of prompt complexity (0.0 to 1.0)."""
    score = 0.0
    tokens = estimate_token_count(prompt)
    lower = prompt.lower()

    if tokens > 500:
        score += 0.3
    elif tokens > 200:
        score += 0.2
    elif tokens > 50:
        score += 0.1

    if _CODE_BLOCK_RE.search(prompt):
        score += 0.25

    keyword_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
    score += min(0.25, keyword_hits * 0.05)

    parts = len(_MULTI_PART_RE.findall(prompt))
    score += min(0.2, parts * 0.05)

    return min(1.0, score)


class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        """Returns tuple of (tier_name, reason)."""
        ...


class ComplexityStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        score = compute_complexity(request.prompt)
        tiers_by_cost = sorted(
            config.tiers.items(),
            key=lambda t: t[1].cost_per_1k_input,
        )
        if score >= config.complexity_threshold:
            name = tiers_by_cost[-1][0]
            reason = f"complexity={score:.2f} >= threshold={config.complexity_threshold}"
        else:
            name = tiers_by_cost[0][0]
            reason = f"complexity={score:.2f} < threshold={config.complexity_threshold}"
        return name, reason


class CostAwareStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        budget = config.cost_budget_usd
        tiers_by_cost = sorted(
            config.tiers.items(),
            key=lambda t: t[1].cost_per_1k_input,
        )

        if budget is not None:
            remaining = budget - spent_usd
            if remaining <= 0:
                return tiers_by_cost[0][0], f"budget exhausted (spent=${spent_usd:.4f})"

            fraction_used = spent_usd / budget if budget > 0 else 1.0
            if fraction_used < 0.5:
                return tiers_by_cost[-1][0], f"budget healthy ({fraction_used:.0%} used)"
            elif fraction_used < 0.8:
                mid = len(tiers_by_cost) // 2
                return tiers_by_cost[mid][0], f"budget moderate ({fraction_used:.0%} used)"
            else:
                return tiers_by_cost[0][0], f"budget tight ({fraction_used:.0%} used)"

        return config.default_tier, "no budget constraint set"


class CascadeStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        cheapest = min(config.tiers.items(), key=lambda t: t[1].cost_per_1k_input)
        return cheapest[0], "cascade: start with cheapest tier"


class LatencyStrategy(BaseStrategy):
    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        target = config.latency_target_ms
        if target is None:
            return config.default_tier, "no latency target set, using default"

        eligible = [
            (name, tier)
            for name, tier in config.tiers.items()
            if tier.avg_latency_ms <= target
        ]

        if eligible:
            best = max(eligible, key=lambda t: t[1].cost_per_1k_input)
            return best[0], f"best tier meeting latency target {target}ms"

        fastest = min(config.tiers.items(), key=lambda t: t[1].avg_latency_ms)
        return fastest[0], (
            f"no tier meets {target}ms target; using fastest ({fastest[1].avg_latency_ms}ms)"
        )


_TOKEN_RE = re.compile(r"\b\w\w+\b")  # words of length 2 or more

_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
})


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LocalTFIDF:
    __slots__ = ("vocab", "idf")

    def __init__(self) -> None:
        self.vocab: set[str] = set()
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> None:
        tokenized_docs = [tokenize(doc) for doc in documents]
        doc_count = len(documents)
        df: dict[str, int] = {}
        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                if term not in _STOP_WORDS:
                    df[term] = df.get(term, 0) + 1
        
        self.vocab = set(df.keys())
        self.idf = {
            term: math.log((1 + doc_count) / (1 + count)) + 1.0
            for term, count in df.items()
        }

    def transform(self, document: str) -> dict[str, float]:
        tokens = tokenize(document)
        if not tokens:
            return {}
        
        tf: dict[str, int] = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1
                
        total_tokens = len(tokens)
        tfidf: dict[str, float] = {}
        for token, count in tf.items():
            tfidf[token] = (count / total_tokens) * self.idf[token]
            
        return tfidf

    @staticmethod
    def cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
        dot_product = sum(v1[t] * v2.get(t, 0.0) for t in v1)
        norm1 = math.sqrt(sum(val ** 2 for val in v1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in v2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)


class SemanticStrategy(BaseStrategy):
    __slots__ = ("_tfidf", "_route_vectors", "_fitted")

    def __init__(self) -> None:
        self._tfidf = LocalTFIDF()
        self._route_vectors: dict[str, list[dict[str, float]]] = {}
        self._fitted = False

    def _fit_routes(self, config: RouterConfig) -> None:
        if self._fitted or not config.semantic_routes:
            return
        
        all_utterances: list[str] = []
        for utterances in config.semantic_routes.values():
            all_utterances.extend(utterances)
            
        if not all_utterances:
            return
            
        self._tfidf.fit(all_utterances)
        
        self._route_vectors = {}
        for tier_name, utterances in config.semantic_routes.items():
            self._route_vectors[tier_name] = [
                self._tfidf.transform(utt) for utt in utterances if utt.strip()
            ]
        self._fitted = True

    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        if not config.semantic_routes:
            return config.default_tier, "semantic_routes not configured; using default tier"
            
        self._fit_routes(config)
        
        prompt_vector = self._tfidf.transform(request.prompt)
        if not prompt_vector:
            return config.default_tier, "empty prompt; using default tier"
            
        best_tier = config.default_tier
        best_sim = -1.0
        
        for tier_name, vectors in self._route_vectors.items():
            for vec in vectors:
                sim = self._tfidf.cosine_similarity(prompt_vector, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_tier = tier_name
                    
        if best_sim <= 0.0:
            return config.default_tier, "no semantic match found (similarity=0.0); using default tier"
            
        return best_tier, f"semantic match found (best similarity={best_sim:.3f})"


class ClassifierStrategy(BaseStrategy):
    __slots__ = ()

    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        weights = config.classifier_weights
        intercept = config.classifier_intercept
        threshold = config.classifier_threshold
        
        if not weights:
            return config.default_tier, "classifier weights not configured; using default tier"
            
        tokens = tokenize(request.prompt)
        score = intercept + sum(weights.get(token, 0.0) for token in tokens)
        
        try:
            prob = 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            prob = 0.0 if score < 0 else 1.0
            
        tiers_by_cost = sorted(
            config.tiers.items(),
            key=lambda t: t[1].cost_per_1k_input,
        )
        
        if prob >= threshold:
            name = tiers_by_cost[-1][0]
            reason = f"classifier probability {prob:.3f} >= threshold {threshold} (score={score:.3f})"
        else:
            name = tiers_by_cost[0][0]
            reason = f"classifier probability {prob:.3f} < threshold {threshold} (score={score:.3f})"
            
        return name, reason


class KNNStrategy(BaseStrategy):
    """k-Nearest-Neighbours router over the local TF-IDF space.

    Stores labelled example prompts per tier and, for each request, finds the
    ``k`` most similar examples and casts a similarity-weighted vote for their
    tiers. Unlike :class:`SemanticStrategy` (single best match), kNN aggregates
    several neighbours, which is far more robust to noisy or ambiguous prompts.

    Recent research ("When Simple kNN Beats Complex Learned Routers",
    arXiv:2505.12601) shows this non-parametric approach rivals heavyweight
    learned routers — while keeping lc-shift's zero-dependency, sub-1ms profile.

    Supports **online learning**: call :meth:`learn` (via ``RouterShifter.learn``)
    to add a labelled example at runtime; the index is rebuilt lazily on the next
    decision.
    """

    __slots__ = ("_tfidf", "_examples", "_vectors", "_runtime", "_fitted")

    def __init__(self) -> None:
        self._tfidf = LocalTFIDF()
        self._examples: list[tuple[str, str]] = []  # (prompt, tier_name)
        self._vectors: list[dict[str, float]] = []
        self._runtime: list[tuple[str, str]] = []
        self._fitted = False

    def learn(self, prompt: str, tier_name: str) -> None:
        """Add a labelled example at runtime and invalidate the fitted index."""
        if prompt.strip():
            self._runtime.append((prompt, tier_name))
            self._fitted = False

    def _fit(self, config: RouterConfig) -> None:
        if self._fitted:
            return

        examples: list[tuple[str, str]] = []
        for tier_name, prompts in (config.knn_examples or {}).items():
            examples.extend((p, tier_name) for p in prompts if p.strip())
        examples.extend(self._runtime)

        self._examples = examples
        if examples:
            self._tfidf.fit([p for p, _ in examples])
            self._vectors = [self._tfidf.transform(p) for p, _ in examples]
        else:
            self._vectors = []
        self._fitted = True

    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        if not config.knn_examples and not self._runtime:
            return config.default_tier, "knn_examples not configured; using default tier"

        self._fit(config)
        if not self._examples:
            return config.default_tier, "no knn examples available; using default tier"

        prompt_vec = self._tfidf.transform(request.prompt)
        if not prompt_vec:
            return config.default_tier, "empty prompt; using default tier"

        scored = [
            (self._tfidf.cosine_similarity(prompt_vec, vec), tier)
            for vec, (_, tier) in zip(self._vectors, self._examples)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(config.knn_k, len(scored))
        neighbours = scored[:k]

        if neighbours[0][0] <= 0.0:
            return config.default_tier, "no knn neighbour matched (similarity=0.0); using default tier"

        votes: dict[str, float] = {}
        for sim, tier in neighbours:
            if sim > 0.0:
                votes[tier] = votes.get(tier, 0.0) + sim

        best_tier = max(votes, key=lambda t: votes[t])
        return best_tier, (
            f"knn vote over {k} neighbours (weight={votes[best_tier]:.3f}, "
            f"top_sim={neighbours[0][0]:.3f})"
        )


class EnsembleStrategy(BaseStrategy):
    """Weighted multi-signal router that combines several base strategies.

    Each configured member casts a vote for its chosen tier; votes are summed by
    the member's weight and the highest-scoring tier wins. This hedges the blind
    spots of any single signal (e.g. complexity heuristics vs. semantic intent).
    """

    __slots__ = ("_members",)

    _AVAILABLE: dict[str, type[BaseStrategy]] = {
        "complexity": ComplexityStrategy,
        "classifier": ClassifierStrategy,
        "semantic": SemanticStrategy,
    }

    def __init__(self) -> None:
        self._members: dict[str, BaseStrategy] = {}

    async def decide(
        self,
        request: ShiftRequest,
        config: RouterConfig,
        spent_usd: float,
    ) -> tuple[str, str]:
        weights = config.ensemble_weights or {}
        if not weights:
            return config.default_tier, "ensemble_weights not configured; using default tier"

        votes: dict[str, float] = {}
        for name, weight in weights.items():
            strat = self._members.get(name)
            if strat is None:
                cls = self._AVAILABLE.get(name)
                if cls is None:
                    continue
                strat = cls()
                self._members[name] = strat
            tier, _ = await strat.decide(request, config, spent_usd)
            votes[tier] = votes.get(tier, 0.0) + weight

        if not votes:
            return config.default_tier, "no ensemble members voted; using default tier"

        best_tier = max(votes, key=lambda t: votes[t])
        breakdown = ", ".join(f"{t}={w:.2f}" for t, w in sorted(votes.items()))
        return best_tier, f"ensemble vote ({breakdown})"


STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    "complexity": ComplexityStrategy,
    "cost_aware": CostAwareStrategy,
    "cascade": CascadeStrategy,
    "latency": LatencyStrategy,
    "semantic": SemanticStrategy,
    "classifier": ClassifierStrategy,
    "knn": KNNStrategy,
    "ensemble": EnsembleStrategy,
}
