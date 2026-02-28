"""
SmartScraperGraph Module
"""

import logging
from typing import Optional, Type

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

        Returns:
            BaseGraph: A graph instance representing the web scraping workflow.
        """
        if self.llm_model == "scrapegraphai/smart-scraper":
            return self._create_scrapegraphai_client_graph()

        return self._build_graph_from_config()

    def _create_scrapegraphai_client_graph(self):
        """Handles the scrapegraphai/smart-scraper model case using external API."""
        try:
            from scrapegraph_py import Client
            from scrapegraph_py.logger import sgai_logger
        except ImportError:
            raise ImportError(
                "scrapegraph_py is not installed. Please install it using 'pip install scrapegraph-py'."
            )

        sgai_logger.set_logging(level="INFO")

        sgai_client = Client(api_key=self.config.get("api_key"))

        response = sgai_client.smartscraper(
            website_url=self.source,
            user_prompt=self.prompt,
        )

        if "request_id" in response and "result" in response:
            logger.info(f"Request ID: {response['request_id']}")
            logger.info(f"Result: {response['result']}")
        else:
            logger.warning("Missing expected keys in response.")

        sgai_client.close()

        return response

    def _build_graph_from_config(self) -> BaseGraph:
        """
        Builds the graph by composing nodes based on individual config flags.

        The pipeline follows this structure:
            fetch -> [parse] -> [reasoning] -> generate_answer -> [reattempt]

        Where optional stages are included based on config flags:
            - parse: included when html_mode is False
            - reasoning: included when reasoning is True
            - reattempt: included when reattempt is True
        """
        html_mode = self.config.get("html_mode", False)
        reasoning = self.config.get("reasoning", False)
        reattempt = self.config.get("reattempt", False)

        # Build nodes
        fetch_node = self._create_fetch_node()
        parse_node = None if html_mode else self._create_parse_node()
        reasoning_node = self._create_reasoning_node() if reasoning else None
        generate_answer_node = self._create_generate_answer_node()
        cond_node, regen_node = (
            self._create_reattempt_nodes() if reattempt else (None, None)
        )

        # Compose the pipeline
        nodes = []
        edges = []

        nodes.append(fetch_node)
        prev_node = fetch_node

        if parse_node:
            nodes.append(parse_node)
            edges.append((prev_node, parse_node))
            prev_node = parse_node

        if reasoning_node:
            nodes.append(reasoning_node)
            edges.append((prev_node, reasoning_node))
            prev_node = reasoning_node

        nodes.append(generate_answer_node)
        edges.append((prev_node, generate_answer_node))
        prev_node = generate_answer_node

        if cond_node and regen_node:
            nodes.extend([cond_node, regen_node])
            edges.append((prev_node, cond_node))
            edges.append((cond_node, regen_node))  # true branch
            edges.append((cond_node, None))  # false branch (end)

        return BaseGraph(
            nodes=nodes,
            edges=edges,
            entry_point=fetch_node,
            graph_name=self.__class__.__name__,
        )

    def _create_fetch_node(self) -> FetchNode:
        """Creates the FetchNode for retrieving content."""
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

    def _create_parse_node(self) -> ParseNode:
        """Creates the ParseNode for parsing fetched content."""
        return ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "llm_model": self.llm_model,
                "chunk_size": self.model_token,
            },
        )

    def _create_reasoning_node(self) -> ReasoningNode:
        """Creates the ReasoningNode for additional reasoning before answer generation."""
        return ReasoningNode(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": self.config.get("additional_info"),
                "schema": self.schema,
            },
        )

    def _create_generate_answer_node(self) -> GenerateAnswerNode:
        """Creates the GenerateAnswerNode for generating the final answer."""
        return GenerateAnswerNode(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": self.config.get("additional_info"),
                "schema": self.schema,
            },
        )

    def _create_reattempt_nodes(self) -> tuple:
        """Creates the ConditionalNode and regeneration node for reattempt logic."""
        cond_node = ConditionalNode(
            input="answer",
            output=["answer"],
            node_name="ConditionalNode",
            node_config={
                "key_name": "answer",
                "condition": 'not answer or answer=="NA"',
            },
        )
        regen_node = GenerateAnswerNode(
            input="user_prompt & answer",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": REGEN_ADDITIONAL_INFO,
                "schema": self.schema,
            },
        )
        return cond_node, regen_node

    def run(self) -> str:
        """
        Executes the scraping process and returns the answer to the prompt.

        Returns:
            str: The answer to the prompt.
        """

        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return self.final_state.get("answer", "No answer found.")
