"""
Unit tests for SmartScraperGraph._create_graph structure.

Verifies that the pipeline-based graph builder produces the expected
node / edge topology for every combination of the ``html_mode``,
``reasoning`` and ``reattempt`` flags.

These tests use mocked node classes and a captured ``BaseGraph`` so they
run without any network, LLM, or browser dependencies.
"""

from itertools import product
from unittest.mock import MagicMock, patch

import pytest

from scrapegraphai.graphs.abstract_graph import AbstractGraph

# ---------------------------------------------------------------------------
# Minimal fake nodes that mirror the public interface _create_graph relies on.
# ---------------------------------------------------------------------------


class _FakeNode:
    _default_name = "FakeNode"
    node_type = "node"

    def __init__(self, input=None, output=None, node_config=None, node_name=None):
        self.input = input
        self.output = output
        self.node_config = node_config
        self.node_name = node_name or self._default_name

    def update_config(self, params, overwrite=False):
        pass


class _FakeFetchNode(_FakeNode):
    _default_name = "Fetch"


class _FakeParseNode(_FakeNode):
    _default_name = "ParseNode"


class _FakeReasoningNode(_FakeNode):
    _default_name = "PromptRefiner"


class _FakeGenerateAnswerNode(_FakeNode):
    _default_name = "GenerateAnswer"


class _FakeConditionalNode(_FakeNode):
    _default_name = "Cond"
    node_type = "conditional_node"


class _CapturedGraph:
    """Stand-in for BaseGraph that just records its constructor args."""

    def __init__(self, nodes, edges, entry_point, graph_name="Custom", **kw):
        self.nodes = list(nodes)
        self.raw_edges = list(edges)
        self.entry_point = entry_point
        self.graph_name = graph_name


# Expected pipeline shape for every (html_mode, reasoning, reattempt) combo.
#   node_classes  – ordered list of fake-class names in the graph
#   edge_indexes  – (from_idx, to_idx) pairs; ``None`` as to_idx means the
#                   false branch of a ConditionalNode terminates the graph.
_EXPECTED = {
    (False, False, False): (
        ["_FakeFetchNode", "_FakeParseNode", "_FakeGenerateAnswerNode"],
        [(0, 1), (1, 2)],
    ),
    (False, False, True): (
        [
            "_FakeFetchNode",
            "_FakeParseNode",
            "_FakeGenerateAnswerNode",
            "_FakeConditionalNode",
            "_FakeGenerateAnswerNode",
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, None)],
    ),
    (False, True, False): (
        [
            "_FakeFetchNode",
            "_FakeParseNode",
            "_FakeReasoningNode",
            "_FakeGenerateAnswerNode",
        ],
        [(0, 1), (1, 2), (2, 3)],
    ),
    (False, True, True): (
        [
            "_FakeFetchNode",
            "_FakeParseNode",
            "_FakeReasoningNode",
            "_FakeGenerateAnswerNode",
            "_FakeConditionalNode",
            "_FakeGenerateAnswerNode",
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, None)],
    ),
    (True, False, False): (
        ["_FakeFetchNode", "_FakeGenerateAnswerNode"],
        [(0, 1)],
    ),
    (True, False, True): (
        [
            "_FakeFetchNode",
            "_FakeGenerateAnswerNode",
            "_FakeConditionalNode",
            "_FakeGenerateAnswerNode",
        ],
        [(0, 1), (1, 2), (2, 3), (2, None)],
    ),
    (True, True, False): (
        ["_FakeFetchNode", "_FakeReasoningNode", "_FakeGenerateAnswerNode"],
        [(0, 1), (1, 2)],
    ),
    (True, True, True): (
        [
            "_FakeFetchNode",
            "_FakeReasoningNode",
            "_FakeGenerateAnswerNode",
            "_FakeConditionalNode",
            "_FakeGenerateAnswerNode",
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, None)],
    ),
}

_FLAG_NAMES = ("html_mode", "reasoning", "reattempt")


_SSG_MODULE = "scrapegraphai.graphs.smart_scraper_graph"

_NODE_PATCHES = {
    "FetchNode": _FakeFetchNode,
    "ParseNode": _FakeParseNode,
    "ReasoningNode": _FakeReasoningNode,
    "GenerateAnswerNode": _FakeGenerateAnswerNode,
    "ConditionalNode": _FakeConditionalNode,
    "BaseGraph": _CapturedGraph,
}


def _build_graph(flags):
    """Instantiate SmartScraperGraph with mocked deps and return the captured graph."""
    patchers = [
        patch(f"{_SSG_MODULE}.{name}", fake) for name, fake in _NODE_PATCHES.items()
    ]
    patchers.append(
        patch.object(
            AbstractGraph, "_create_llm", create=True, return_value=MagicMock()
        )
    )
    for p in patchers:
        p.start()
    try:
        # Import *after* patching so the module-level names are already swapped.
        from scrapegraphai.graphs.smart_scraper_graph import SmartScraperGraph

        ssg = SmartScraperGraph(
            prompt="test prompt",
            source="https://example.com",
            config={"llm": {"model": "mock"}, **flags},
        )
        # model_token is usually set by _create_llm; our mock bypasses that,
        # so provide it manually and rebuild.
        ssg.model_token = 4096
        return ssg._create_graph()
    finally:
        for p in patchers:
            p.stop()


def _topology(graph):
    """Return (node_class_names, edge_index_pairs) for the captured graph."""
    node_classes = [type(n).__name__ for n in graph.nodes]
    idx = {id(n): i for i, n in enumerate(graph.nodes)}
    edge_idx = [
        (idx[id(a)], None if b is None else idx[id(b)]) for a, b in graph.raw_edges
    ]
    return node_classes, edge_idx


@pytest.mark.unit
@pytest.mark.parametrize("combo", list(product([False, True], repeat=3)))
def test_graph_topology_per_flag_combo(combo):
    """Each boolean (html_mode, reasoning, reattempt) combo yields the expected shape."""
    flags = dict(zip(_FLAG_NAMES, combo))
    graph = _build_graph(flags)

    exp_nodes, exp_edges = _EXPECTED[combo]
    act_nodes, act_edges = _topology(graph)

    assert act_nodes == exp_nodes, f"node mismatch for {flags}"
    assert act_edges == exp_edges, f"edge mismatch for {flags}"
    assert graph.entry_point is graph.nodes[0]
    assert graph.graph_name == "SmartScraperGraph"


@pytest.mark.unit
def test_default_flags_match_all_false():
    """Absent flags behave exactly like ``False`` (no optional stages)."""
    g_default = _build_graph({})
    g_false = _build_graph({"html_mode": False, "reasoning": False, "reattempt": False})
    assert _topology(g_default) == _topology(g_false)


@pytest.mark.unit
@pytest.mark.parametrize("combo", list(product([False, True], repeat=3)))
def test_flags_compose_independently(combo):
    """
    Each flag toggles exactly one stage without influencing any other.

    Demonstrates that a fourth flag could be added without touching the
    expectations for the existing three – the core extensibility goal of
    replacing the combinatorial lookup table.
    """
    html_mode, reasoning, reattempt = combo
    flags = dict(zip(_FLAG_NAMES, combo))
    graph = _build_graph(flags)
    node_classes, _ = _topology(graph)

    # parse_node presence ⇔ html_mode disabled
    assert ("_FakeParseNode" in node_classes) is (not html_mode)
    # reasoning_node presence ⇔ reasoning enabled
    assert ("_FakeReasoningNode" in node_classes) is reasoning
    # conditional tail presence ⇔ reattempt enabled
    assert ("_FakeConditionalNode" in node_classes) is reattempt
    # fixed stages are always present
    assert "_FakeFetchNode" in node_classes
    assert "_FakeGenerateAnswerNode" in node_classes


@pytest.mark.unit
def test_reattempt_branch_edge_ordering():
    """
    ConditionalNode's first outgoing edge must be the *true* branch (regen)
    and the second the *false* branch (None/stop). BaseGraph relies on this
    ordering to populate ``true_node_name`` / ``false_node_name``.
    """
    graph = _build_graph({"reattempt": True})

    cond_idx = next(
        i
        for i, n in enumerate(graph.nodes)
        if type(n).__name__ == "_FakeConditionalNode"
    )
    cond_edges = [e for e in graph.raw_edges if e[0] is graph.nodes[cond_idx]]

    assert len(cond_edges) == 2
    # first edge: regen (GenerateAnswer), second edge: None (stop)
    assert type(cond_edges[0][1]).__name__ == "_FakeGenerateAnswerNode"
    assert cond_edges[1][1] is None


@pytest.mark.unit
def test_node_configs_propagated():
    """Non-flag config options still reach their target nodes unchanged."""
    graph = _build_graph(
        {
            "reasoning": True,
            "reattempt": True,
            "additional_info": "extra ctx",
            "force": True,
            "cut": False,
            "loader_kwargs": {"timeout": 30},
        }
    )

    by_class = {type(n).__name__: n for n in graph.nodes}

    fetch_cfg = by_class["_FakeFetchNode"].node_config
    assert fetch_cfg["force"] is True
    assert fetch_cfg["cut"] is False
    assert fetch_cfg["loader_kwargs"] == {"timeout": 30}

    reason_cfg = by_class["_FakeReasoningNode"].node_config
    assert reason_cfg["additional_info"] == "extra ctx"

    # generate_answer (first occurrence) gets user additional_info
    gen = next(n for n in graph.nodes if type(n).__name__ == "_FakeGenerateAnswerNode")
    assert gen.node_config["additional_info"] == "extra ctx"

    # regen (last node) gets the REGEN_ADDITIONAL_INFO prompt
    regen = graph.nodes[-1]
    from scrapegraphai.prompts import REGEN_ADDITIONAL_INFO

    assert regen.node_config["additional_info"] == REGEN_ADDITIONAL_INFO
    assert regen.input == "user_prompt & answer"

    # conditional node keeps its explicit name and condition
    cond = by_class["_FakeConditionalNode"]
    assert cond.node_name == "ConditionalNode"
    assert cond.node_config["condition"] == 'not answer or answer=="NA"'


# ---------------------------------------------------------------------------
# Tests for the hosted ScrapeGraphAI API client branch
# ---------------------------------------------------------------------------


class TestScrapegraphaiClientBranch:
    """Tests for the ``scrapegraphai/smart-scraper`` hosted API path."""

    @pytest.mark.unit
    def test_client_branch_calls_api_correctly(self):
        """
        When llm_model is "scrapegraphai/smart-scraper", _create_graph should
        delegate to _handle_scrapegraphai_client which calls the remote API.
        """
        import sys

        # Create mock client and module
        mock_client_instance = MagicMock()
        mock_client_instance.smartscraper.return_value = {
            "request_id": "test-req-123",
            "result": {"data": "scraped content"},
        }
        mock_client_class = MagicMock(return_value=mock_client_instance)
        mock_sgai_logger = MagicMock()

        # Insert mock modules into sys.modules so the import inside
        # _handle_scrapegraphai_client finds them
        mock_scrapegraph_py = MagicMock()
        mock_scrapegraph_py.Client = mock_client_class
        mock_scrapegraph_py_logger = MagicMock()
        mock_scrapegraph_py_logger.sgai_logger = mock_sgai_logger

        sys.modules["scrapegraph_py"] = mock_scrapegraph_py
        sys.modules["scrapegraph_py.logger"] = mock_scrapegraph_py_logger

        try:
            with patch.object(
                AbstractGraph, "_create_llm", create=True, return_value=MagicMock()
            ):
                from scrapegraphai.graphs.smart_scraper_graph import SmartScraperGraph

                ssg = SmartScraperGraph(
                    prompt="Extract all products",
                    source="https://example.com/products",
                    config={
                        "llm": {"model": "scrapegraphai/smart-scraper"},
                        "api_key": "test-key",
                    },
                )
                ssg.llm_model = "scrapegraphai/smart-scraper"

                result = ssg._create_graph()

            # Verify client was instantiated with the api_key from config
            mock_client_class.assert_called_once_with(api_key="test-key")

            # Verify smartscraper was called with correct args
            mock_client_instance.smartscraper.assert_called_once_with(
                website_url="https://example.com/products",
                user_prompt="Extract all products",
            )

            # Verify client was closed
            mock_client_instance.close.assert_called_once()

            # Verify the response is returned directly (not a BaseGraph)
            assert result == {
                "request_id": "test-req-123",
                "result": {"data": "scraped content"},
            }
        finally:
            sys.modules.pop("scrapegraph_py", None)
            sys.modules.pop("scrapegraph_py.logger", None)

    @pytest.mark.unit
    def test_client_branch_raises_import_error_when_package_missing(self):
        """
        When scrapegraph_py is not installed, _handle_scrapegraphai_client
        should raise ImportError with a helpful message.
        """
        import sys

        # Ensure scrapegraph_py is NOT in sys.modules
        sys.modules.pop("scrapegraph_py", None)
        sys.modules.pop("scrapegraph_py.logger", None)

        with patch.object(
            AbstractGraph, "_create_llm", create=True, return_value=MagicMock()
        ):
            from scrapegraphai.graphs.smart_scraper_graph import SmartScraperGraph

            ssg = SmartScraperGraph(
                prompt="test",
                source="https://example.com",
                config={"llm": {"model": "scrapegraphai/smart-scraper"}},
            )
            ssg.llm_model = "scrapegraphai/smart-scraper"

            with pytest.raises(ImportError) as exc_info:
                ssg._handle_scrapegraphai_client()

            assert "scrapegraph_py is not installed" in str(exc_info.value)
            assert "pip install scrapegraph-py" in str(exc_info.value)

    @pytest.mark.unit
    def test_client_branch_logs_warning_on_missing_keys(self):
        """
        When the API response lacks expected keys, a warning should be logged.
        """
        import sys

        # Create mock client that returns response without expected keys
        mock_client_instance = MagicMock()
        mock_client_instance.smartscraper.return_value = {"unexpected": "data"}
        mock_client_class = MagicMock(return_value=mock_client_instance)
        mock_sgai_logger = MagicMock()

        mock_scrapegraph_py = MagicMock()
        mock_scrapegraph_py.Client = mock_client_class
        mock_scrapegraph_py_logger = MagicMock()
        mock_scrapegraph_py_logger.sgai_logger = mock_sgai_logger

        sys.modules["scrapegraph_py"] = mock_scrapegraph_py
        sys.modules["scrapegraph_py.logger"] = mock_scrapegraph_py_logger

        try:
            with (
                patch(f"{_SSG_MODULE}.logger") as mock_module_logger,
                patch.object(
                    AbstractGraph, "_create_llm", create=True, return_value=MagicMock()
                ),
            ):
                from scrapegraphai.graphs.smart_scraper_graph import SmartScraperGraph

                ssg = SmartScraperGraph(
                    prompt="test",
                    source="https://example.com",
                    config={
                        "llm": {"model": "scrapegraphai/smart-scraper"},
                        "api_key": "k",
                    },
                )
                ssg.llm_model = "scrapegraphai/smart-scraper"

                ssg._create_graph()

                # Verify warning was logged
                mock_module_logger.warning.assert_called_once_with(
                    "Missing expected keys in response."
                )
        finally:
            sys.modules.pop("scrapegraph_py", None)
            sys.modules.pop("scrapegraph_py.logger", None)
