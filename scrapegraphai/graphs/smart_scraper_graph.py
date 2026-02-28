"""
SmartScraperGraph Module
"""

import logging
from typing import List, Optional, Tuple, Type

from pydantic import BaseModel

from ..nodes import (
    ConditionalNode,
    FetchNode,
    GenerateAnswerNode,
    ParseNode,
    ReasoningNode,
)
from ..prompts import REGEN_ADDITIONAL_INFO
from .abstract_graph import AbstractGraph
from .base_graph import BaseGraph

# Initialize logger
logger = logging.getLogger(__name__)


class SmartScraperGraph(AbstractGraph):
    """
    SmartScraper is a scraping pipeline that automates the process of
    extracting information from web pages
    using a natural language model to interpret and answer prompts.

    Attributes:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (BaseModel): The schema for the graph output.
        llm_model: An instance of a language model client, configured for generating answers.
        embedder_model: An instance of an embedding model client,
        configured for generating embeddings.
        verbose (bool): A flag indicating whether to show print statements during execution.
        headless (bool): A flag indicating whether to run the graph in headless mode.

    Args:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (BaseModel): The schema for the graph output.

    Example:
        >>> smart_scraper = SmartScraperGraph(
        ...     "List me all the attractions in Chioggia.",
        ...     "https://en.wikipedia.org/wiki/Chioggia",
        ...     {"llm": {"model": "openai/gpt-3.5-turbo"}}
        ... )
        >>> result = smart_scraper.run()
        )
    """

    def __init__(
        self,
        prompt: str,
        source: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
    ):
        super().__init__(prompt, config, source, schema)

        self.input_key = "url" if source.startswith("http") else "local_dir"

        # for detailed logging of the SmartScraper API set it to True
        self.verbose = config.get("verbose", False)

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping.

        The graph is assembled as a **linear pipeline** where each optional
        stage is included or skipped based on an independent boolean flag.
        This replaces the previous combinatorial lookup table (keyed by
        ``(html_mode, reasoning, reattempt)``) which required 2^N entries
        and grew exponentially with every new flag.

        Pipeline stages, in order::

            fetch_node                 (always)
            └─ parse_node              (unless ``html_mode`` is set)
               └─ reasoning_node       (only if ``reasoning`` is set)
                  └─ generate_answer_node   (always)
                     └─ cond_node + regen_node   (only if ``reattempt`` is set)

        Adding a new optional stage is now a matter of:
        1. building the node when its flag is truthy, and
        2. inserting it at the right position in the ``pipeline`` list below.

        Returns:
            BaseGraph: A graph instance representing the web scraping workflow.
        """
        # Delegate to the hosted ScrapeGraphAI API instead of building a
        # local node pipeline when the sentinel model name is configured.
        if self.llm_model == "scrapegraphai/smart-scraper":
            return self._handle_scrapegraphai_client()

        # ---- Read flags (each flag toggles exactly one pipeline stage) ----
        # bool() coercion means any truthy value enables the stage, keeping
        # the three checks consistent with one another.
        html_mode = bool(self.config.get("html_mode", False))
        reasoning = bool(self.config.get("reasoning", False))
        reattempt = bool(self.config.get("reattempt", False))

        # ---- Build individual stages --------------------------------------
        fetch_node = self._build_fetch_node()
        parse_node = self._build_parse_node() if not html_mode else None
        reasoning_node = self._build_reasoning_node() if reasoning else None
        generate_answer_node = self._build_generate_answer_node()

        # ---- Compose the linear part of the pipeline ----------------------
        # Each entry is a node in execution order; ``None`` values (disabled
        # optional stages) are filtered out. Sequential edges between the
        # remaining nodes are derived automatically, so adding/removing an
        # optional linear stage here needs no change elsewhere.
        pipeline = [
            fetch_node,
            parse_node,
            reasoning_node,
            generate_answer_node,
        ]
        nodes: List = [n for n in pipeline if n is not None]
        edges: List[Tuple] = list(zip(nodes, nodes[1:]))

        # ---- Optional branching tail: reattempt ---------------------------
        # This stage is non-linear: a ConditionalNode has two outgoing edges
        # (true -> regenerate, false -> stop) so it's appended explicitly
        # instead of being part of the zipped chain above.
        if reattempt:
            cond_node = self._build_cond_node()
            regen_node = self._build_regen_node()
            nodes.extend((cond_node, regen_node))
            edges.extend(
                (
                    (generate_answer_node, cond_node),
                    # NB: order matters for ConditionalNode – first edge is
                    #     the *true* branch, second is the *false* branch.
                    (cond_node, regen_node),  # true: answer invalid -> retry
                    (cond_node, None),  # false: answer accepted -> stop
                )
            )

        return BaseGraph(
            nodes=nodes,
            edges=edges,
            entry_point=fetch_node,
            graph_name=self.__class__.__name__,
        )

    def _handle_scrapegraphai_client(self):
        """
        Invoke the hosted ScrapeGraphAI SmartScraper API directly and
        return its raw response in place of a local :class:`BaseGraph`.

        This path is taken when ``self.llm_model`` is set to the sentinel
        string ``"scrapegraphai/smart-scraper"``; the request is executed
        immediately (no graph is built) and the API response dict is
        returned so the caller can store it on ``self.graph``.

        Raises:
            ImportError: If the optional ``scrapegraph_py`` dependency is
                not installed.
        """
        try:
            from scrapegraph_py import Client
            from scrapegraph_py.logger import sgai_logger
        except ImportError:
            raise ImportError(
                "scrapegraph_py is not installed. Please install it using 'pip install scrapegraph-py'."
            )

        sgai_logger.set_logging(level="INFO")

        # Initialize the client with explicit API key
        sgai_client = Client(api_key=self.config.get("api_key"))

        # SmartScraper request
        response = sgai_client.smartscraper(
            website_url=self.source,
            user_prompt=self.prompt,
        )

        # Use logging instead of print for better production practices
        if "request_id" in response and "result" in response:
            logger.info(f"Request ID: {response['request_id']}")
            logger.info(f"Result: {response['result']}")
        else:
            logger.warning("Missing expected keys in response.")

        sgai_client.close()

        return response

    # ------------------------------------------------------------------
    # Node builders
    #
    # Each builder is responsible for a single pipeline stage. Keeping the
    # node construction separate from the pipeline assembly means adding a
    # new stage amounts to:
    #   1. writing one builder method, and
    #   2. inserting the node at the desired spot in ``_create_graph``.
    # ------------------------------------------------------------------

    def _build_fetch_node(self) -> FetchNode:
        """Build the fetch stage (always present)."""
        return FetchNode(
            input="url | local_dir",
            output=["doc"],
            node_config={
                "llm_model": self.llm_model,
                "force": self.config.get("force", False),
                "cut": self.config.get("cut", True),
                "loader_kwargs": self.config.get("loader_kwargs", {}),
                "browser_base": self.config.get("browser_base"),
                "scrape_do": self.config.get("scrape_do"),
                "storage_state": self.config.get("storage_state"),
            },
        )

    def _build_parse_node(self) -> ParseNode:
        """Build the parse stage (skipped when ``html_mode`` is set)."""
        return ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "llm_model": self.llm_model,
                "chunk_size": self.model_token,
            },
        )

    def _build_reasoning_node(self) -> ReasoningNode:
        """Build the reasoning stage (enabled by ``reasoning``)."""
        return ReasoningNode(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": self.config.get("additional_info"),
                "schema": self.schema,
            },
        )

    def _build_generate_answer_node(self) -> GenerateAnswerNode:
        """Build the answer-generation stage (always present)."""
        return GenerateAnswerNode(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": self.config.get("additional_info"),
                "schema": self.schema,
            },
        )

    def _build_cond_node(self) -> ConditionalNode:
        """Build the conditional check for the reattempt stage."""
        return ConditionalNode(
            input="answer",
            output=["answer"],
            node_name="ConditionalNode",
            node_config={
                "key_name": "answer",
                "condition": 'not answer or answer=="NA"',
            },
        )

    def _build_regen_node(self) -> GenerateAnswerNode:
        """Build the retry answer-generation stage (enabled by ``reattempt``)."""
        return GenerateAnswerNode(
            input="user_prompt & answer",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": REGEN_ADDITIONAL_INFO,
                "schema": self.schema,
            },
        )

    def run(self) -> str:
        """
        Executes the scraping process and returns the answer to the prompt.

        Returns:
            str: The answer to the prompt.
        """

        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return self.final_state.get("answer", "No answer found.")
