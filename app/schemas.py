from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator

# Agent request/response models
class AgentRequest(BaseModel):
    """Request model for agent run"""
    query: str = Field(..., description="The query to process", min_length=1)
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class AgentResponse(BaseModel):
    """Response model for agent run"""
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of the query")
    code_snippets: Optional[List[Dict[str, Any]]] = Field(None, description="Relevant code snippets")
    code_fix: Optional[Dict[str, Any]] = Field(None, description="Generated code fix")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    @root_validator(pre=True)
    def validate_fields(cls, values):
        """Validate that required fields are present or error is set"""
        # If there's an error message, other fields can be None
        if "error" in values and values["error"] is not None:
            return values
            
        # If no error, require analysis, code_snippets, and code_fix
        if not values.get("analysis"):
            values["analysis"] = {"error_type": "validation_error", "description": "Missing analysis"}
        
        if not values.get("code_snippets") and values.get("analysis", {}).get("error_type") not in ["initialization_error", "api_error"]:
            values["code_snippets"] = []
        
        if not values.get("code_fix") and values.get("analysis", {}).get("error_type") not in ["initialization_error", "api_error"]:
            values["code_fix"] = {"explanation": "No code fix available"}
            
        return values

# MCP request/response models
class MCPRequest(BaseModel):
    """Request model for MCP"""
    message: str = Field(..., description="The message to process")
    context: Dict[str, Any] = Field(default_factory=dict, description="MCP context")

class MCPResponse(BaseModel):
    """Response model for MCP"""
    result: Dict[str, Any] = Field(..., description="The result of the message processing")
    mcp_context: Dict[str, Any] = Field(..., description="Updated MCP context")

# Code management models
class CodeUploadResponse(BaseModel):
    """Response model for code upload"""
    status: str
    message: str
    file_name: str
    language: str

class CodeProcessResponse(BaseModel):
    """Response model for directory processing"""
    status: str
    message: str
    directory: str
    supported_languages: List[str]

class CodeSearchQuery(BaseModel):
    """Request model for code search"""
    query: str = Field(..., description="The search query", min_length=1)
    limit: int = Field(default=5, description="Maximum number of results", ge=1, le=20)
    filter_language: Optional[str] = Field(default=None, description="Filter by language")

# Health check model
class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    components: Dict[str, str] 