from fastapi import APIRouter

from app.api.endpoints import agent, code

api_router = APIRouter()

# Add all endpoint routers
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(code.router, prefix="/code", tags=["code"]) 