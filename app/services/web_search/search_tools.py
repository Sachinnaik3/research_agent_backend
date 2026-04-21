import os
import time
import traceback
from dotenv import load_dotenv
from typing import List
from app.utils.logger import logger

# LangChain Imports
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()

def _firecrawl_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search using Firecrawl. Extracts cleaner markdown and metadata dates.
    """
    start_time = time.time()
    try:
        logger.info({
            "event": "firecrawl_search_started",
            "query": query,
            "max_results": max_results
        })

        loader = FireCrawlLoader(
            query=query,
            mode="search",
            params={"limit": max_results}
        )

        docs = loader.load()
        out: List[dict] = []

        for doc in docs:
            # Firecrawl stores dates in metadata
            out.append({
                "title": doc.metadata.get("title") or "No Title",
                "url": doc.metadata.get("url") or "",
                "snippet": doc.page_content[:500] + "...", 
                "published_at": doc.metadata.get("published_date") or doc.metadata.get("date"),
                "source": "firecrawl",
            })

        duration = round(time.time() - start_time, 2)
        logger.success({
            "event": "firecrawl_search_success",
            "query": query,
            "results_count": len(out),
            "duration_sec": duration
        })
        return out

    except Exception as e:
        logger.error({"event": "firecrawl_search_failed", "error": str(e)})
        logger.error(traceback.format_exc())
        return []


def _duckduckgo_search(query: str, max_results: int = 5, search_type: str = "news") -> List[dict]:
    """
    Search using DuckDuckGo. 
    Defaulted to search_type='news' because 'text' search rarely provides dates.
    """
    start_time = time.time()
    try:
        logger.info({
            "event": f"duckduckgo_{search_type}_search_started",
            "query": query,
            "max_results": max_results
        })

        # Use Wrapper to define the source (news or text)
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        tool = DuckDuckGoSearchResults(
            api_wrapper=wrapper,
            source=search_type, 
            output_format='list'
        )

        results = tool.invoke({"query": query})
        out: List[dict] = []

        for r in results or []:
            out.append({
                "title": r.get("title") or "No Title",
                "url": r.get("link") or "",
                "snippet": r.get("snippet") or r.get("body") or "",
                # The 'news' backend provides a 'date' field
                "published_at": r.get("date"), 
                "source": f"duckduckgo_{search_type}",
            })

        duration = round(time.time() - start_time, 2)
        logger.success({
            "event": f"duckduckgo_{search_type}_success",
            "query": query,
            "results_count": len(out),
            "duration_sec": duration
        })
        return out

    except Exception as e:
        logger.error({"event": "duckduckgo_search_failed", "error": str(e)})
        logger.error(traceback.format_exc())
        return []
