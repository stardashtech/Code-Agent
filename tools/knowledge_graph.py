from neo4j import GraphDatabase, basic_auth
import os
import logging
from config import settings # Import settings object

logger = logging.getLogger(__name__)

class KnowledgeGraphTool:
    """
    Tool for executing knowledge graph queries using Neo4j.
    Reads connection details from the global settings object.
    """
    def __init__(self):
        # Use settings object
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.driver = None # Initialize driver as None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            # Optional: Add a connection check here if needed
            # self.driver.verify_connectivity()
            logger.info(f"Neo4j driver initialized for URI: {self.uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver for URI {self.uri}: {e}")
            # Keep self.driver as None

    def query(self, query_text: str) -> str:
        if self.driver is None:
            logger.error("KnowledgeGraph query failed: Neo4j driver not initialized.")
            return "Error: Neo4j driver not initialized. Check connection settings and logs."
        try:
            # Use driver.execute_query for modern Neo4j drivers if possible/preferred
            # Sticking with session.run for now based on original code
            with self.driver.session() as session:
                result = session.run(query_text)
                # Process results robustly
                records = []
                try:
                    records = [record.data() for record in result]
                except Exception as parse_e:
                    logger.warning(f"Failed to parse all Neo4j records fully: {parse_e}")
                    # Attempt to get summary if data parsing fails
                    summary = result.consume().summary
                    return f"KnowledgeGraph query executed (parsing failed): {summary}"
                    
                summary = result.consume().summary # Consume the result fully
                logger.debug(f"Neo4j query executed successfully: {summary.query}")
                return f"KnowledgeGraph query result: {records}"
        except Exception as e:
            logger.exception("Error executing KnowledgeGraph query: %s", e)
            return f"Error: KnowledgeGraph query execution error: {str(e)}" 