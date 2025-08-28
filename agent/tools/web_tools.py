from __future__ import annotations
import re, html, os, json
from typing import Optional, List, Dict, Tuple
from ..registry import tool
from ..state import State
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
#from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.utilities import SerpAPIWrapper


"""Util that calls Wikipedia."""

import logging
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

WIKIPEDIA_MAX_QUERY_LENGTH = 300




class WikipediaAPIWrapper(BaseModel):
    """Wrapper around WikipediaAPI.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    """

    wiki_client: Any  #: :meta private:
    top_k_results: int = 1
    lang: str = "en"
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 40000

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia

            lang =  "en"
            wikipedia.set_lang(lang)
            values["wiki_client"] = wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values



    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
        )
        page_info = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                # print(wiki_page)
                page_info.append(wiki_page.content)
        return page_info


    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def _page_to_document(self, page_title: str, wiki_page: Any) -> Document:
        main_meta = {
            "title": page_title,
            "summary": wiki_page.summary,
            "source": wiki_page.url,
        }
        add_meta = (
            {
                "categories": wiki_page.categories,
                "page_url": wiki_page.url,
                "image_urls": wiki_page.images,
                "related_titles": wiki_page.links,
                "parent_id": wiki_page.parent_id,
                "references": wiki_page.references,
                "revision_id": wiki_page.revision_id,
                "sections": wiki_page.sections,
            }
            if self.load_all_available_meta
            else {}
        )
        doc = Document(
            page_content=wiki_page.content[: self.doc_content_chars_max],
            metadata={
                **main_meta,
                **add_meta,
            },
        )
        return doc

    def _fetch_page(self, page: str) -> Optional[str]:
        try:
            return self.wiki_client.page(title=page, auto_suggest=False)
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            return None



    def load(self, query: str) -> List[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        """
        return list(self.lazy_load(query))




    def lazy_load(self, query: str) -> Iterator[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        """
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results
        )
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if doc := self._page_to_document(page_title, wiki_page):
                    yield doc



# ---- Tools --------------------------------------------------------------------

@tool("wiki_lookup")
def wiki_lookup_tool(state: State, title_or_query: Optional[str] = None, **kwargs) -> str:
    """
    Pull content from English Wikipedia and answer the current question from it.
    Uses the official REST and action APIs for cleaner plaintext, search + redirects,
    with an HTML fallback as last resort. Output unchanged.
    """
    q = (title_or_query or state["question"]).strip()
    wikipedia = WikipediaAPIWrapper()
    text = wikipedia.run(q)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": f"""Question: {q} 
            "Answer the question in 1-2 word or number only, no explanations or even tags like 'Answer:'."""},
            {
                "role": "user",
                "content": f"Use the following Wikipedia text to answer the question: \n\n{text}\n\n",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content

@tool("web_lookup")
def web_lookup_tool(state: State, title_or_query: Optional[str] = None, **kwargs) -> str:
    """
    search the web and answer the current question from it.
    Uses the Google search api
    """
    params = {
    "engine": "google",
    "gl": "us",
    "hl": "en",
    }
    search = SerpAPIWrapper(params=params)
    q = (title_or_query or state["question"]).strip()
    text = search.run(q)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": f"""Question: {q} 
            "Answer the question in 1-2 word or number only, no explanations or even tags like 'Answer:'."""},
            {
                "role": "user",
                "content": f"Use the following Wikipedia text to answer the question: \n\n{text}\n\n",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content

