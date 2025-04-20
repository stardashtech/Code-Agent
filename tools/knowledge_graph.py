from neo4j import GraphDatabase, basic_auth
import os
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphTool:
    """
    Tool for executing knowledge graph queries using Neo4j.
    Environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    """
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
        except Exception as e:
            logger.exception("Failed to establish Neo4j connection: %s", e)
            self.driver = None

    def query(self, query_text: str) -> str:
        if self.driver is None:
            return "Error: Could not establish Neo4j connection."
        try:
            with self.driver.session() as session:
                result = session.run(query_text)
                records = [record.data() for record in result]
                return f"KnowledgeGraph query result: {records}"
        except Exception as e:
            logger.exception("Error in KnowledgeGraph query: %s", e)
            return f"Error: KnowledgeGraph query error: {str(e)}" 