from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import time
from typing import Dict, Any, Optional
import traceback

from app.schemas import AgentRequest, AgentResponse, MCPRequest, MCPResponse
from app.services.code_agent import CodeAgent
from app.api.dependencies import get_agent
from app.metrics import AGENT_REQUESTS, AGENT_REQUEST_DURATION
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    agent: CodeAgent = Depends(get_agent)
):
    """
    Run the code agent with the given query
    """
    # Add debug logging
    logger.info(f"Settings OpenAI API Key set: {settings.openai_api_key is not None}")
    
    start_time = time.time()
    AGENT_REQUESTS.inc()  # Increment the counter

    try:
        logger.info(f"Received agent request: {request.query[:100]}...")
        
        # Run the agent
        result = await agent.run(request.query)
        
        # Measure duration and record metric
        duration = time.time() - start_time
        AGENT_REQUEST_DURATION.observe(duration)
        
        logger.info(f"Agent execution completed in {duration:.2f}s")
        
        # Convert to AgentResponse
        return result
    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        
        # Measure duration even for errors
        duration = time.time() - start_time
        AGENT_REQUEST_DURATION.observe(duration)
        
        # Return error response
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error running agent: {str(e)}"
            )

@router.post("/mcp", response_model=MCPResponse)
async def handle_mcp(
    request: MCPRequest,
    agent: CodeAgent = Depends(get_agent)
):
    """
    Handle a Model Context Protocol (MCP) request
    """
    start_time = time.time()
    AGENT_REQUESTS.inc()  # Increment the counter

    try:
        logger.info(f"Received MCP request: {request.message[:100]}...")
        
        # Process MCP request
        result = await agent.process_mcp_request(request.message, request.context)
        
        # Measure duration and record metric
        duration = time.time() - start_time
        AGENT_REQUEST_DURATION.observe(duration)
        
        logger.info(f"MCP execution completed in {duration:.2f}s")
        
        # Convert to MCPResponse
        return {
            "result": result,
            "mcp_context": result.get("mcp_context", {})
        }
    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error processing MCP request: {str(e)}", exc_info=True)
        
        # Measure duration even for errors
        duration = time.time() - start_time
        AGENT_REQUEST_DURATION.observe(duration)
        
        # Return error response
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing MCP request: {str(e)}"
            ) 