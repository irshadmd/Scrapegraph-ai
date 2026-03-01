"""
Unit tests for SmartScraperGraph._create_graph structure.

Verifies that the pipeline-based graph builder produces the expected
node / edge topology for every combination of the ``html_mode``,
``reasoning`` and ``reattempt`` flags.

These tests use mocked node classes and a captured ``BaseGraph`` so they
run without any network, LLM, or browser dependencies.
"""

import sys
from itertools import product
from types import ModuleType
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
# Hosted ScrapeGraphAI client branch (llm_model="scrapegraphai/smart-scraper")
#
# When this sentinel model name is configured, _create_graph skips the local
# pipeline entirely and delegates to the scrapegraph_py SDK. That SDK import
# is local to _handle_scrapegraphai_client, so we stub ``scrapegraph_py`` in
# sys.modules before triggering the branch.
# ---------------------------------------------------------------------------


def _make_fake_scrapegraph_py(response):
    """
    Build fake ``scrapegraph_py`` / ``scrapegraph_py.logger`` modules so the
    local import inside ``_handle_scrapegraphai_client`` succeeds without the
    real SDK installed.

    Returns the fake ``Client`` class so tests can inspect call arguments.
    """
    fake_client = MagicMock(name="ScrapegraphClient")
    fake_client.return_value.smartscraper.return_value = response

    fake_pkg = ModuleType("scrapegraph_py")
    fake_pkg.Client = fake_client
    fake_pkg.__path__ = []  # mark as package for the submodule import

    fake_logger_mod = ModuleType("scrapegraph_py.logger")
    fake_logger_mod.sgai_logger = MagicMock(name="sgai_logger")

    return fake_pkg, fake_logger_mod, fake_client


def _make_sgai_scraper(extra_config=None):
    """Instantiate SmartScraperGraph with llm_model forced to the sentinel string."""
    from scrapegraphai.graphs.smart_scraper_graph import SmartScraperGraph

    return SmartScraperGraph(
        prompt="test prompt",
        source="https://example.com",
        config={"llm": {"model": "mock"}, **(extra_config or {})},
    )


@pytest.fixture
def sgai_scraper():
    """
    SmartScraperGraph instance whose llm_model resolves to the sentinel
    string ``"scrapegraphai/smart-scraper"`` so the client branch is taken.

    Node classes and BaseGraph are patched to harmless fakes so the initial
    ``__init__`` -> ``_create_graph`` call (which runs *before* we can force
    the sentinel llm_model) doesn't touch real LLMs or browsers.
    """
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
        ssg = _make_sgai_scraper({"api_key": "sgai-test-key"})
        ssg.llm_model = "scrapegraphai/smart-scraper"
        yield ssg
    finally:
        for p in patchers:
            p.stop()


@pytest.mark.unit
def test_scrapegraphai_client_delegates_and_returns_response(sgai_scraper, caplog):
    """
    Sentinel llm_model bypasses the local pipeline and returns the SDK
    response; the request is made with the right URL, prompt and API key,
    the logger is configured, and the client is closed.
    """
    import logging

    response = {"request_id": "req-abc123", "result": {"answer": "42"}}
    fake_pkg, fake_logger_mod, fake_client = _make_fake_scrapegraph_py(response)

    with patch.dict(
        sys.modules,
        {"scrapegraph_py": fake_pkg, "scrapegraph_py.logger": fake_logger_mod},
    ):
        with caplog.at_level(logging.INFO, logger=_SSG_MODULE):
            result = sgai_scraper._create_graph()

    # The raw SDK response is returned in place of a BaseGraph.
    assert result is response

    # Client is constructed with the api_key from the config.
    fake_client.assert_called_once_with(api_key="sgai-test-key")
    # Request uses the instance's source URL and prompt.
    fake_client.return_value.smartscraper.assert_called_once_with(
        website_url="https://example.com",
        user_prompt="test prompt",
    )
    # SDK logger is configured and the client is closed after use.
    fake_logger_mod.sgai_logger.set_logging.assert_called_once_with(level="INFO")
    fake_client.return_value.close.assert_called_once()

    # Happy path logs request_id and result at INFO.
    assert any("req-abc123" in r.message for r in caplog.records)


@pytest.mark.unit
def test_scrapegraphai_client_warns_on_missing_keys(sgai_scraper, caplog):
    """Responses lacking ``request_id`` / ``result`` trigger a warning, not a crash."""
    import logging

    response = {"unexpected": "shape"}  # missing both expected keys
    fake_pkg, fake_logger_mod, _ = _make_fake_scrapegraph_py(response)

    with patch.dict(
        sys.modules,
        {"scrapegraph_py": fake_pkg, "scrapegraph_py.logger": fake_logger_mod},
    ):
        with caplog.at_level(logging.WARNING, logger=_SSG_MODULE):
            result = sgai_scraper._create_graph()

    assert result is response
    assert any(
        "Missing expected keys" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


@pytest.mark.unit
def test_scrapegraphai_client_import_error(sgai_scraper):
    """
    A missing ``scrapegraph_py`` SDK surfaces as an ``ImportError`` with a
    helpful install hint rather than a bare ``ModuleNotFoundError``.
    """
    # Setting a module entry to ``None`` is Python's import-system negative
    # cache: any subsequent ``import scrapegraph_py`` (or ``from``-import of
    # a submodule) raises ImportError immediately without touching finders,
    # so this is reliable even when the real SDK is installed on disk.
    # ``patch.dict`` handles the save/restore so no manual cleanup is needed.
    with patch.dict(
        sys.modules, {"scrapegraph_py": None, "scrapegraph_py.logger": None}
    ):
        with pytest.raises(ImportError, match="pip install scrapegraph-py"):
            sgai_scraper._handle_scrapegraphai_client()


@pytest.mark.unit
def test_non_sentinel_llm_model_builds_local_pipeline(sgai_scraper):
    """
    Any llm_model other than the exact sentinel string falls through to the
    local pipeline – the client handler must not be touched.
    """
    # Something close-but-not-equal to the sentinel.
    sgai_scraper.llm_model = "scrapegraphai/smart-scraper-v2"
    sgai_scraper.model_token = 4096

    with patch.object(sgai_scraper, "_handle_scrapegraphai_client") as mock_handler:
        graph = sgai_scraper._create_graph()

    mock_handler.assert_not_called()
    assert isinstance(graph, _CapturedGraph)
    assert [type(n).__name__ for n in graph.nodes] == [
        "_FakeFetchNode",
        "_FakeParseNode",
        "_FakeGenerateAnswerNode",
    ]
