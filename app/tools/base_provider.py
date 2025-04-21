import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class CodeSearchProvider:
    """Base class for code search providers"""
    async def search(self, query: str, language: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        # Added **kwargs to allow subclasses to accept more specific args like per_page
        raise NotImplementedError 