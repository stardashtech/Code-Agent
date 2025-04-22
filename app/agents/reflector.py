import logging
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Any
import re
import json
from enum import Enum

if TYPE_CHECKING:
    from app.services.code_agent import LLMProvider

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Different types of queries that can be processed."""
    GENERAL = "general"
    CODE_SPECIFIC = "code_specific"
    ERROR_ANALYSIS = "error_analysis"
    DEPENDENCY = "dependency"
    FEATURE_REQUEST = "feature_request"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ARCHITECTURE = "architecture"

class QueryComplexity(Enum):
    """Complexity levels for queries."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Reflector:
    """
    Agent responsible for reflecting on the user query to better understand it.
    
    Features:
    - Advanced context understanding for better query assessment
    - Multiple prompt templates for different query types
    - Adaptive keyword extraction based on query content
    - Intelligent query decomposition
    - Enhanced clarity assessment with detailed reasoning
    - Query complexity evaluation
    """

    def __init__(self, provider: Optional['LLMProvider'] = None):
        """Initialize the Reflector agent.

        Args:
            provider: Optional instance of an LLMProvider. Required to perform reflection.
        """
        self.provider = provider
        
        # Base system prompt for clarity assessment
        self._base_clarity_system_prompt = """You are an expert query analyzer specifically focused on assessing technical programming queries.
Your primary responsibility is to evaluate how clear, specific, and actionable a user's query is.
You can detect ambiguities, vague language, and missing context that would make it difficult to provide an accurate response.
Provide a detailed assessment that helps determine if clarification is needed before proceeding."""
        
        # Prompt templates for different query types
        self._general_prompt_template = """
        Assess the clarity of this general programming query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Goal clarity: Is the objective clearly stated?
        2. Context sufficiency: Is there enough background information?
        3. Specificity: Is it specific enough to be actionable?
        4. Ambiguity: Are there multiple ways to interpret the request?
        5. Completeness: Does it include all necessary information?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._code_specific_prompt_template = """
        Assess the clarity of this code-specific query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Code component clarity: Are relevant files/functions/classes clearly identified?
        2. Intended behavior: Is the desired code behavior or modification clearly specified?
        3. Context sufficiency: Is there enough code context provided?
        4. Environment details: Is language/framework/version information provided if relevant?
        5. Expected output: Is the desired outcome clearly defined?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "code_components": [<key code elements mentioned>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._error_analysis_prompt_template = """
        Assess the clarity of this error-related query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Error specificity: Is the complete error message included?
        2. Error context: Is the scenario when the error occurs described?
        3. Troubleshooting history: Are previous attempts mentioned?
        4. Code context: Are relevant code snippets provided?
        5. Expected behavior: Is the expected behavior clearly stated?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "error_type": "<identified error type if available>",
            "affected_files": [<probable files involved>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._dependency_prompt_template = """
        Assess the clarity of this dependency-related query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Dependency specificity: Are the exact dependencies identified?
        2. Issue clarity: Is the dependency issue or requirement clearly explained?
        3. Version constraints: Are version requirements specified?
        4. Package ecosystem: Is the package manager or ecosystem identified?
        5. Project context: Is the project's dependency structure explained?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "packages": [<identified package names>],
            "package_manager": "<identified package manager>",
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._feature_request_prompt_template = """
        Assess the clarity of this feature request query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Feature description: Is the desired functionality clearly described?
        2. Integration details: Is it clear where/how the feature should be integrated?
        3. Design preferences: Are implementation preferences specified?
        4. Constraints: Are constraints or requirements mentioned?
        5. Acceptance criteria: Are success criteria defined?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "feature_requirements": [<core requirements identified>],
            "integration_points": [<integration locations identified>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """

        self._performance_prompt_template = """
        Assess the clarity of this performance-related query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Performance issue: Is the performance problem clearly described?
        2. Metrics: Are measurable performance metrics provided?
        3. System context: Is the system environment described?
        4. Bottleneck identification: Are suspected bottlenecks specified?
        5. Expected improvement: Are performance targets defined?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "performance_issues": [<identified performance issues>],
            "bottleneck_areas": [<suspected bottleneck code areas>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._security_prompt_template = """
        Assess the clarity of this security-related query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Security concern: Is the security issue clearly described?
        2. Vulnerability details: Are specific vulnerabilities identified?
        3. System exposure: Is the attack surface or exposure explained?
        4. Criticality: Is the severity or risk level indicated?
        5. Current protections: Are existing security measures mentioned?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "security_concerns": [<identified security issues>],
            "vulnerable_components": [<potentially vulnerable areas>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
        
        self._architecture_prompt_template = """
        Assess the clarity of this architecture-related query:
        
        QUERY: {query}
        
        {context}
        
        Evaluate the query's clarity on these dimensions:
        1. Architectural scope: Is the scope of architectural changes clear?
        2. Current structure: Is the current architecture described?
        3. Design goals: Are architectural goals and requirements specified?
        4. Constraints: Are system constraints or limitations mentioned?
        5. Success criteria: Are evaluation criteria for the architecture defined?
        
        Respond with a detailed JSON containing:
        {{
            "clarity_score": <float 0.0-1.0>,
            "is_clear": <boolean>,
            "ambiguities": [<strings listing specific ambiguities>],
            "missing_context": [<strings listing missing information>],
            "clarity_reasoning": "<explanation of your assessment>",
            "architecture_components": [<identified architectural components>],
            "design_principles": [<relevant design principles>],
            "complexity": "<low|medium|high|very_high>",
            "complexity_reasoning": "<explanation of complexity assessment>"
        }}
        """
    
    def _determine_query_type(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryType:
        """
        Determine the type of query to select the appropriate prompt template.
        
        Args:
            query: The user query to analyze
            conversation_history: Optional list of previous messages
            
        Returns:
            The query type as a QueryType enum value
        """
        query_lower = query.lower()
        
        # Tool kullanımına dair belirteçler - Araç odaklı sorguları daha iyi tanımak için
        tool_usage_indicators = [
            "how to use", "using", "example of", "examples of", "show me", "demonstrate", 
            "how do i use", "can you show", "how can i", "search tool", "github search", 
            "stackoverflow search", "web search", "tool", "example", "api", "function"
        ]
        
        # Bir tool/API hakkında soru soruyorsa - genel olarak işle
        if any(indicator in query_lower for indicator in tool_usage_indicators):
            logger.info("Query classified as tool usage related.")
            return QueryType.GENERAL
        
        # More comprehensive indicators for each query type
        error_indicators = [
            "error", "exception", "fail", "bug", "issue", "traceback", "not working", 
            "broken", "crash", "debug", "fix", "problem", "doesn't work", "incorrect", 
            "wrong", "unexpected", "traceback", "stacktrace", "segmentation fault"
        ]
        
        dependency_indicators = [
            "dependency", "package", "library", "module", "import", "install", 
            "requirements", "pip", "npm", "yarn", "composer", "cargo", "gem", 
            "nuget", "version conflict", "compatibility", "upgrade", "downgrade"
        ]
        
        feature_indicators = [
            "add", "implement", "create", "new feature", "enhance", "extend", 
            "integrate", "build", "develop", "feature request", "new functionality", 
            "add capability", "enhancement", "add support for", "feature development"
        ]
        
        code_specific_indicators = [
            "function", "class", "method", "variable", "code", "implementation", 
            "syntax", "file", "refactor", "snippet", "struct", "object", "interface",
            "architecture", "design pattern", "algorithm", "data structure", "optimization"
        ]
        
        performance_indicators = [
            "performance", "slow", "speed", "optimize", "efficiency", "latency",
            "throughput", "bottleneck", "profiling", "memory usage", "cpu usage",
            "benchmark", "response time", "lag", "resource consumption", "scaling"
        ]
        
        security_indicators = [
            "security", "vulnerability", "exploit", "attack", "threat", "hack",
            "authentication", "authorization", "encryption", "injection", "xss",
            "csrf", "malicious", "permission", "secure", "risk", "sensitive data"
        ]
        
        architecture_indicators = [
            "architecture", "design", "structure", "system", "component", "module",
            "pattern", "microservice", "monolith", "decoupling", "separation of concerns",
            "layered", "mvc", "mvvm", "organization", "blueprint", "framework"
        ]
        
        # Github, npm, docker gibi belirli tool isimlerini kontrol et
        if "github" in query_lower or "git" in query_lower.split():
            logger.info("Query contains GitHub reference. Classified as GENERAL.")
            return QueryType.GENERAL
            
        if "npm" in query_lower or "pypi" in query_lower or "package" in query_lower:
            logger.info("Query contains package manager reference. Classified as DEPENDENCY.")
            return QueryType.DEPENDENCY
            
        if "docker" in query_lower or "container" in query_lower:
            logger.info("Query contains Docker reference. Classified as CODE_SPECIFIC.")
            return QueryType.CODE_SPECIFIC
            
        if "search" in query_lower:
            logger.info("Query contains search reference. Classified as GENERAL.")
            return QueryType.GENERAL
        
        # Check for indicators in the query with more sophisticated matching
        if any(indicator in query_lower for indicator in error_indicators):
            return QueryType.ERROR_ANALYSIS
        
        if any(indicator in query_lower for indicator in dependency_indicators):
            return QueryType.DEPENDENCY
        
        if any(indicator in query_lower for indicator in feature_indicators):
            return QueryType.FEATURE_REQUEST
        
        if any(indicator in query_lower for indicator in performance_indicators):
            return QueryType.PERFORMANCE
        
        if any(indicator in query_lower for indicator in security_indicators):
            return QueryType.SECURITY
            
        if any(indicator in query_lower for indicator in architecture_indicators):
            return QueryType.ARCHITECTURE
        
        if any(indicator in query_lower for indicator in code_specific_indicators) or "```" in query:
            return QueryType.CODE_SPECIFIC
        
        # Check conversation history for contextual clues
        if conversation_history:
            # Look at the last 3 messages for context
            recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            # Combine recent messages for analysis
            combined_history = " ".join([msg.get("content", "") for msg in recent_messages])
            combined_history_lower = combined_history.lower()
            
            # Check for code blocks or specific indicators in conversation history
            if "```" in combined_history or "traceback" in combined_history:
                return QueryType.CODE_SPECIFIC
                
            if any(indicator in combined_history_lower for indicator in error_indicators):
                return QueryType.ERROR_ANALYSIS
                
            if any(indicator in combined_history_lower for indicator in performance_indicators):
                return QueryType.PERFORMANCE
                
            if any(indicator in combined_history_lower for indicator in security_indicators):
                return QueryType.SECURITY
                
            if any(indicator in combined_history_lower for indicator in architecture_indicators):
                return QueryType.ARCHITECTURE
        
        # Default to general if no specific type is identified
        return QueryType.GENERAL

    async def assess_query_clarity(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None, code_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess the clarity of a query to determine if more information is needed.
        
        Args:
            query: The user query to assess
            conversation_history: Optional list of previous messages
            code_context: Optional code context related to the query
            
        Returns:
            A dictionary containing:
            - clarity_score: A float between 0.0 and 1.0
            - is_clear: Boolean indicating if the query is clear
            - ambiguities: List of specific ambiguities
            - missing_context: List of missing context items
            - clarity_reasoning: Explanation of the assessment
        """
        if not self.provider:
            logger.error("LLM Provider not available for clarity assessment.")
            return {
                "clarity_score": 0.5,
                "is_clear": False,
                "ambiguities": ["Cannot assess query clarity without LLM provider"],
                "missing_context": ["LLM provider configuration"],
                "clarity_reasoning": "Clarity assessment requires a configured LLM provider."
            }
            
        # Determine query type to select appropriate prompt template
        query_type = self._determine_query_type(query, conversation_history)
        logger.info(f"Query type determined as: {query_type.value}")
        
        # Select prompt template based on query type
        if query_type == QueryType.CODE_SPECIFIC:
            prompt_template = self._code_specific_prompt_template
        elif query_type == QueryType.ERROR_ANALYSIS:
            prompt_template = self._error_analysis_prompt_template
        elif query_type == QueryType.DEPENDENCY:
            prompt_template = self._dependency_prompt_template
        elif query_type == QueryType.FEATURE_REQUEST:
            prompt_template = self._feature_request_prompt_template
        elif query_type == QueryType.PERFORMANCE:
            prompt_template = self._performance_prompt_template
        elif query_type == QueryType.SECURITY:
            prompt_template = self._security_prompt_template
        elif query_type == QueryType.ARCHITECTURE:
            prompt_template = self._architecture_prompt_template
        else:
            prompt_template = self._general_prompt_template
            
        # Format conversation history for context
        context = ""
        if conversation_history and len(conversation_history) > 0:
            formatted_history = self._format_conversation_history(conversation_history)
            context += f"CONVERSATION HISTORY:\n{formatted_history}\n\n"
            
        # Add code context if provided
        if code_context:
            context += f"CODE CONTEXT:\n{code_context}\n\n"
            
        # Generate the complete prompt
        prompt = prompt_template.format(query=query, context=context)
        
        # Create messages for LLM request
        messages = [
            {"role": "system", "content": self._base_clarity_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Get response from LLM
            response = await self.provider.generate_completion(messages, temperature=0.3)
            
            # Extract and parse JSON from response
            try:
                # Look for JSON pattern in the response
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    
                    # Ensure required fields are present
                    if "clarity_score" not in result:
                        result["clarity_score"] = 0.5
                    if "is_clear" not in result:
                        result["is_clear"] = False
                    if "ambiguities" not in result:
                        result["ambiguities"] = []
                    if "missing_context" not in result:
                        result["missing_context"] = []
                    if "clarity_reasoning" not in result:
                        result["clarity_reasoning"] = "No reasoning provided."
                        
                    # Add query type to result
                    result["query_type"] = query_type.value
                    
                    logger.info(f"Clarity assessment: score={result['clarity_score']}, clear={result['is_clear']}")
                    return result
                else:
                    logger.warning("Failed to extract JSON from clarity assessment response.")
                    return {
                        "clarity_score": 0.5,
                        "is_clear": False,
                        "ambiguities": ["Cannot parse clarity assessment response"],
                        "missing_context": [],
                        "clarity_reasoning": "Failed to parse LLM response for clarity assessment.",
                        "query_type": query_type.value
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing clarity assessment JSON: {e}")
                return {
                    "clarity_score": 0.5,
                    "is_clear": False,
                    "ambiguities": ["Invalid format in clarity assessment response"],
                    "missing_context": [],
                    "clarity_reasoning": "Failed to parse LLM response for clarity assessment.",
                    "query_type": query_type.value
                }
        except Exception as e:
            logger.error(f"Error during clarity assessment: {e}", exc_info=True)
            return {
                "clarity_score": 0.5,
                "is_clear": False,
                "ambiguities": [f"Error during assessment: {e}"],
                "missing_context": [],
                "clarity_reasoning": "Error occurred during LLM-based clarity assessment.",
                "query_type": query_type.value
            }
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for inclusion in prompts.
        
        Args:
            conversation_history: List of message dictionaries
            
        Returns:
            Formatted string of conversation history
        """
        if not conversation_history:
            return "Conversation History: None"
        
        formatted = "Conversation History:\n"
        # Only include the last 5 messages to avoid excessive token usage
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        for i, message in enumerate(recent_history):
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "... [message truncated]"
            
            formatted += f"{i+1}. {role}: {content}\n\n"
            
        return formatted

    async def extract_keywords(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        Extract important keywords from a query.
        
        Args:
            query: The user query to extract keywords from
            conversation_history: Optional list of previous messages
            
        Returns:
            A list of extracted keywords
        """
        if not self.provider:
            return []
        
        # Determine query type for contextual keyword extraction
        query_type = self._determine_query_type(query, conversation_history)
        
        # Create a prompt that emphasizes different types of keywords based on query type
        system_prompt = """You are an expert in extracting meaningful keywords from programming queries.
Your task is to identify the most important technical terms, concepts, and identifiers that capture 
the essence of the query. Focus on terms that would be useful for semantic search or understanding the query intent."""
        
        user_prompt = f"""
        Extract the most important keywords from this query:
        
        QUERY: {query}
        
        Query Type: {query_type.value}
        
        {self._format_conversation_history(conversation_history) if conversation_history else ""}
        
        Focus on extracting:
        """
        
        # Customize keyword focus based on query type
        if query_type == QueryType.ERROR_ANALYSIS:
            user_prompt += """
            - Error types and exception names
            - Function/method names where errors occur
            - Package/library names
            - Error codes or messages
            - Programming language and runtime details
            - Affected components or systems
            """
        elif query_type == QueryType.DEPENDENCY:
            user_prompt += """
            - Package/library names and exact versions
            - Package managers
            - Dependency requirements
            - Compatibility terms
            - Build systems or environments
            - Installation contexts
            """
        elif query_type == QueryType.FEATURE_REQUEST:
            user_prompt += """
            - Feature names and functionality
            - Component or module names
            - Integration points
            - Implementation requirements
            - Design patterns or architecture terms
            - User needs or use cases
            - Similar existing features
            """
        elif query_type == QueryType.CODE_SPECIFIC:
            user_prompt += """
            - Function/method/class names
            - File names or paths
            - Programming language
            - Code concepts and patterns
            - Implementation details
            - APIs or interfaces mentioned
            - Technical concepts being implemented
            """
        elif query_type == QueryType.PERFORMANCE:
            user_prompt += """
            - Performance metrics
            - Bottleneck components
            - Optimization techniques
            - Benchmark values
            - System resources mentioned
            - Performance-critical operations
            - Scaling concepts
            """
        elif query_type == QueryType.SECURITY:
            user_prompt += """
            - Security vulnerability types
            - Authentication/authorization concepts
            - Threat vectors
            - Security protocols
            - Sensitive data types
            - Security frameworks or tools
            - Compliance standards
            """
        elif query_type == QueryType.ARCHITECTURE:
            user_prompt += """
            - Architecture patterns
            - System components
            - Design principles
            - Integration patterns
            - Scale considerations
            - System boundaries
            - Technical constraints
            """
        else:
            user_prompt += """
            - Technical concepts
            - Tools or technologies
            - Programming languages
            - Actions or operations
            - Key domain terms
            - Questions or requirements
            """
            
        user_prompt += """
        Return ONLY a JSON array of strings. Each string should be a single keyword or key phrase.
        Include at most 15 keywords, prioritizing the most important and specific terms.
        Each term should be directly relevant to the query and useful for understanding or searching.
        
        Example: ["python", "file handling", "os.path", "permission error", "read-only", "docker container"]
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.provider.generate_completion(messages, temperature=0.1)
            
            # Extract and parse the JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            
            if json_match:
                keywords = json.loads(json_match.group(0))
                if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                    logging.info(f"Extracted {len(keywords)} keywords: {', '.join(keywords[:5])}...")
                    return keywords
            
            # Fallback extraction if JSON parsing fails
            logging.warning("JSON parsing failed for keywords, falling back to simple extraction")
            # Simple extraction: take any word in quotes as a keyword
            fallback_keywords = re.findall(r'"([^"]*)"', response)
            if fallback_keywords:
                return fallback_keywords
                
            return []
            
        except Exception as e:
            logging.error(f"Error during keyword extraction: {str(e)}", exc_info=True)
            return []

    async def decompose_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: The user query to decompose
            conversation_history: Optional list of previous messages
            
        Returns:
            A list of dictionaries, each containing:
            - sub_query: The text of the sub-query
            - purpose: The purpose of this sub-query
            - dependencies: List of indices of dependent sub-queries
            - priority: Priority level of this sub-query
        """
        if not self.provider:
            return []
        
        # Determine query type and complexity to guide decomposition
        query_type = self._determine_query_type(query, conversation_history)
        
        system_prompt = """
        You are an expert at breaking down complex programming queries into logical, actionable sub-queries.
        Your task is to analyze a query and determine if it contains multiple questions, requires multiple steps,
        or should be broken down for clearer handling.
        
        When decomposing queries:
        1. Identify distinct questions or requests within the main query
        2. Break multi-step processes into individual steps
        3. Separate information gathering from implementation tasks
        4. Create a logical sequence with clear dependencies between sub-queries
        5. Assign appropriate priority levels to each sub-query
        """
        
        user_prompt = f"""
        Analyze this query and determine if it needs to be broken down:
        
        QUERY: {query}
        
        Query Type: {query_type.value}
        
        {self._format_conversation_history(conversation_history) if conversation_history else ""}
        
        If the query is simple and focused on a single task, respond with an empty array: []
        
        If the query is complex (contains multiple requests, requires multiple steps, or has distinct components), 
        break it down into logical sub-queries.
        
        Return a JSON array of objects, where each object has:
        - "sub_query": The text of the sub-query as a clear, standalone question
        - "purpose": What this sub-query accomplishes
        - "dependencies": Array of indices (0-based) of any sub-queries this one depends on
        - "priority": One of ["high", "medium", "low"] indicating importance
        
        Example for a complex query:
        [
          {{
            "sub_query": "What version of React is used in this project?",
            "purpose": "Identify React version",
            "dependencies": [],
            "priority": "high"
          }},
          {{
            "sub_query": "What React components need to be updated to use hooks instead of class components?",
            "purpose": "Identify components for modernization",
            "dependencies": [0],
            "priority": "high"
          }},
          {{
            "sub_query": "How should I convert class components to functional components with hooks?",
            "purpose": "Determine conversion strategy",
            "dependencies": [0, 1],
            "priority": "medium"
          }},
          {{
            "sub_query": "Update the React components to use hooks instead of class components",
            "purpose": "Implement component modernization",
            "dependencies": [0, 1, 2],
            "priority": "high"
          }}
        ]
        
        Return ONLY the JSON array.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.provider.generate_completion(messages, temperature=0.2)
            
            # Extract and parse the JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            
            if json_match:
                decomposed = json.loads(json_match.group(0))
                if isinstance(decomposed, list):
                    if decomposed:  # If not empty
                        logging.info(f"Decomposed query into {len(decomposed)} sub-queries")
                        
                        # Validate and fix potential issues with the decomposition
                        for i, sub_query in enumerate(decomposed):
                            # Ensure dependencies only reference earlier sub-queries
                            if "dependencies" in sub_query:
                                sub_query["dependencies"] = [d for d in sub_query["dependencies"] if d < i]
                                
                            # Ensure priority is valid
                            if "priority" not in sub_query or sub_query["priority"] not in ["high", "medium", "low"]:
                                sub_query["priority"] = "medium"
                    else:
                        logging.info("Query assessed as not needing decomposition")
                    return decomposed
            
            logging.warning("Failed to parse decomposition response, returning empty list")
            return []
            
        except Exception as e:
            logging.error(f"Error during query decomposition: {str(e)}", exc_info=True)
            return []

    async def analyze_intent(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Analyze the intent behind a query.
        
        Args:
            query: The user query to analyze
            conversation_history: Optional list of previous messages
            
        Returns:
            A dictionary containing:
            - primary_intent: The main intent behind the query
            - secondary_intents: Additional intents if present
            - confidence: Confidence level in the intent assessment
            - entities: Extracted entities from the query
            - context_requirements: Requirements for context to fulfill the query
            - planning_insights: Insights useful for planning execution
            - complexity: Assessment of query complexity
        """
        if not self.provider:
            return {
                "primary_intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "context_requirements": {},
                "planning_insights": {}
            }
        
        # Determine query type to guide intent analysis
        query_type = self._determine_query_type(query, conversation_history)
        
        system_prompt = """
        You are an expert at analyzing programming and development queries to identify the user's intent.
        Your analysis helps determine the best approach to resolve the query.
        
        When analyzing intent:
        1. Identify the primary goal the user wants to accomplish
        2. Extract relevant entities mentioned in the query
        3. Determine what context would be needed to fully address the query
        4. Provide insights that would help plan an effective response
        5. Consider both explicit and implicit intents
        """
        
        user_prompt = f"""
        Analyze the intent behind this query:
        
        QUERY: {query}
        
        Query Type: {query_type.value}
        
        {self._format_conversation_history(conversation_history) if conversation_history else ""}
        
        Return a JSON object with:
        
        1. "primary_intent": Select the most applicable from:
           - "fix_error": Fixing a bug or error
           - "implement_feature": Adding new functionality
           - "refactor_code": Improving existing code without changing functionality
           - "optimize_performance": Making code perform better
           - "explain_code": Understanding what code does
           - "add_dependency": Adding or managing dependencies
           - "improve_security": Addressing security concerns
           - "design_architecture": Designing system architecture
           - "other": Other intents (with explanation)
        
        2. "secondary_intents": Array of additional intents from the same list above
        
        3. "confidence": A float 0.0-1.0 indicating confidence in intent assessment
        
        4. "entities": An object with:
           - "languages": Array of programming languages mentioned
           - "files": Array of file names/paths mentioned
           - "functions": Array of function/method names mentioned
           - "packages": Array of packages/libraries mentioned
           - "errors": Array of error types/messages mentioned
           - "concepts": Array of technical concepts mentioned
        
        5. "context_requirements": An object with:
           - "code_examples": "needed" | "helpful" | "not_needed"
           - "error_details": "needed" | "helpful" | "not_needed"
           - "version_info": "needed" | "helpful" | "not_needed"
           - "implementation_details": "needed" | "helpful" | "not_needed"
           - "existing_code_base": "needed" | "helpful" | "not_needed"
           - "environment_info": "needed" | "helpful" | "not_needed"
        
        6. "planning_insights": An object with:
           - "search_priority": "code" | "web" | "github" | "stackoverflow" | combinations with | separator
           - "analysis_depth_required": "shallow" | "moderate" | "deep"
           - "external_resources_needed": boolean
           - "estimated_complexity": "simple" | "moderate" | "complex" | "very_complex"
           - "suggested_approach": Brief description of recommended approach
        
        Return ONLY the JSON object.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.provider.generate_completion(messages, temperature=0.1)
            
            # Extract and parse the JSON object
            json_match = re.search(r'{.*}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Ensure required fields have defaults
                if "primary_intent" not in result:
                    result["primary_intent"] = "other"
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "secondary_intents" not in result:
                    result["secondary_intents"] = []
                    
                # Add query_type to the result
                result["query_type"] = query_type.value
                
                # Log intent analysis
                intent_str = f"{result['primary_intent']} ({result['confidence']:.2f})"
                if result.get("secondary_intents"):
                    intent_str += f" + secondary: {', '.join(result['secondary_intents'])}"
                logging.info(f"Intent analysis: {intent_str}")
                
                return result
            
            # Fallback if JSON parsing fails
            logging.warning("Failed to parse intent analysis response, using default")
            return {
                "primary_intent": "other",
                "secondary_intents": [],
                "confidence": 0.3,
                "entities": {},
                "context_requirements": {
                    "code_examples": "helpful",
                    "error_details": "helpful",
                    "version_info": "helpful",
                    "implementation_details": "helpful",
                    "existing_code_base": "helpful",
                    "environment_info": "helpful"
                },
                "planning_insights": {
                    "search_priority": "code|web",
                    "analysis_depth_required": "moderate",
                    "external_resources_needed": True,
                    "estimated_complexity": "moderate",
                    "suggested_approach": "Standard analysis and response"
                },
                "query_type": query_type.value
            }
            
        except Exception as e:
            logging.error(f"Error during intent analysis: {str(e)}", exc_info=True)
            return {
                "primary_intent": "unknown",
                "secondary_intents": [],
                "confidence": 0.0,
                "entities": {},
                "context_requirements": {},
                "planning_insights": {},
                "query_type": query_type.value
            } 