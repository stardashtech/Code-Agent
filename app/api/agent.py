from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.schemas import AgentRequest, AgentResponse, SubgoalResult, MCPRequest, MCPResponse
from app.services.code_agent import run_agent
from typing import Dict, Any, List
import logging

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/run", response_model=AgentResponse, status_code=status.HTTP_200_OK)
async def agent_run(req: AgentRequest):
    """
    Run the agent with the provided query
    
    - **query**: The user query to process
    
    Returns the agent's response including plan, subgoals, execution trace, and final answer
    """
    try:
        logger.info(f"Processing agent request with query: {req.query}")
        result = await run_agent(req.query)
        
        # Transform result into Pydantic model structure
        trace: List[SubgoalResult] = []
        for item in result["execution_trace"]:
            subgoal = item["subgoal"]
            result_data = item["result"]
            
            # Extract tool_used from result if available
            tool_used = result_data.get("tool_used", "llm") if isinstance(result_data, dict) else "unknown"
            
            trace.append(SubgoalResult(
                subgoal=subgoal,
                tool_used=tool_used,
                result=result_data
            ))
        
        # Build response object
        response = AgentResponse(
            plan=result["plan"],
            subgoals=result["subgoals"],
            execution_trace=trace,
            final_answer=result["final_answer"],
            chain_of_thought=result.get("chain_of_thought")
        )
        
        logger.info("Successfully processed agent request")
        return response
        
    except Exception as e:
        logger.error(f"Error processing agent request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing agent request: {str(e)}"
        )

# Model Context Protocol endpoint
@router.post("/mcp", response_model=MCPResponse, status_code=status.HTTP_200_OK)
async def model_context_protocol(req: MCPRequest):
    """
    Process a request using Model Context Protocol
    
    - **message**: The user message to process
    - **context**: Optional context information
    
    Returns the response and updated context
    """
    try:
        logger.info(f"Processing MCP request with message: {req.message}")
        
        # Use the agent to process the query
        result = await run_agent(req.message)
        
        # Extract the relevant information for MCP response
        response = MCPResponse(
            response=result["final_answer"],
            context={
                "plan": result["plan"],
                "subgoals": result["subgoals"],
                "chain_of_thought": result.get("chain_of_thought"),
                # Include any context from the request
                **(req.context or {})
            }
        )
        
        logger.info("Successfully processed MCP request")
        return response
        
    except Exception as e:
        logger.error(f"Error processing MCP request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing MCP request: {str(e)}"
        ) 