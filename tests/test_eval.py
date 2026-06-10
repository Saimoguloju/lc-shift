from __future__ import annotations

from pathlib import Path

import pytest

from lc_shift.cli import main
from lc_shift.config import RouterConfig, Strategy
from lc_shift.eval import BenchmarkResult, EvalRecord, evaluate, load_dataset
from lc_shift.router import RouterShifter

_DATASET = [
    EvalRecord(prompt="hi there", ideal_tier="economy"),
    EvalRecord(prompt="say hello", ideal_tier="economy"),
    EvalRecord(
        prompt=(
            "Analyze and compare the trade-off and explain why. "
            "```python\ndef f(): pass\n```\n1. First 2. Second 3. Third"
        ),
        ideal_tier="performance",
    ),
]


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_metrics_shape(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config.model_copy(
            update={"strategy": Strategy.COMPLEXITY, "complexity_threshold": 0.3}
        ))
        result = await evaluate(router, _DATASET)

        assert isinstance(result, BenchmarkResult)
        assert result.total == 3
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.quality_preserved <= 1.0
        assert result.cost_savings_pct <= 100.0
        assert sum(result.routed_distribution.values()) == 3

    @pytest.mark.asyncio
    async def test_cascade_max_savings_but_under_routes(
        self, three_tier_config: RouterConfig
    ) -> None:
        # Cascade always picks cheapest -> max savings, but under-routes hard prompts.
        router = RouterShifter(three_tier_config.model_copy(update={"strategy": Strategy.CASCADE}))
        result = await evaluate(router, _DATASET)
        assert result.routed_distribution["economy"] == 3
        assert result.cost_savings_pct > 0
        assert result.under_route_rate > 0  # the performance prompt got under-routed

    @pytest.mark.asyncio
    async def test_empty_dataset_raises(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        with pytest.raises(ValueError, match="empty"):
            await evaluate(router, [])

    @pytest.mark.asyncio
    async def test_unknown_ideal_tier_raises(self, three_tier_config: RouterConfig) -> None:
        router = RouterShifter(three_tier_config)
        with pytest.raises(ValueError, match="not in router tiers"):
            await evaluate(router, [EvalRecord(prompt="x", ideal_tier="ghost")])

    def test_report_renders(self) -> None:
        result = BenchmarkResult(
            strategy="complexity", total=10, accuracy=0.8, under_route_rate=0.1,
            over_route_rate=0.1, quality_preserved=0.9, cost_total_usd=0.01,
            cost_all_strong_usd=0.05, cost_savings_pct=80.0, avg_overhead_ms=0.02,
        )
        report = result.format_report()
        assert "Cost savings" in report
        assert "80.0%" in report


class TestDatasetLoading:
    def test_load_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"prompt": "hi", "ideal_tier": "economy"}\n\n'
            '{"prompt": "prove it", "ideal_tier": "performance"}\n',
            encoding="utf-8",
        )
        records = load_dataset(f)
        assert len(records) == 2
        assert records[0].ideal_tier == "economy"


class TestCLI:
    def test_route_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["route", "hello there", "--preset", "mixed-frontier", "--strategy", "complexity"])
        assert code == 0
        assert "Tier:" in capsys.readouterr().out

    def test_providers_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["providers"])
        assert code == 0
        assert "providers" in capsys.readouterr().out

    def test_bench_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"prompt": "hi there", "ideal_tier": "economy"}\n', encoding="utf-8")
        code = main(["bench", str(f), "--preset", "anthropic-3tier", "--strategy", "cascade"])
        assert code == 0
        assert "Cost savings" in capsys.readouterr().out

    def test_unknown_preset_exits(self) -> None:
        with pytest.raises(SystemExit):
            main(["route", "hi", "--preset", "does-not-exist"])

    def test_route_with_config_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text(
            '{"tiers": {"big": {"name": "Big", "provider": "x", "model_id": "x",'
            ' "cost_per_1k_input": 0.01, "cost_per_1k_output": 0.02, "avg_latency_ms": 100},'
            ' "small": {"name": "Small", "provider": "x", "model_id": "y",'
            ' "cost_per_1k_input": 0.001, "cost_per_1k_output": 0.002, "avg_latency_ms": 50}},'
            ' "default_tier": "small", "strategy": "knn", "knn_k": 1,'
            ' "knn_examples": {"big": ["prove the theorem"], "small": ["say hello"]}}',
            encoding="utf-8",
        )
        code = main(["route", "prove the theorem now", "--config", str(cfg)])
        assert code == 0
        assert "big" in capsys.readouterr().out
