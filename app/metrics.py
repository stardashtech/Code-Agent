"""
Metrics definitions for the application
"""
from prometheus_client import Counter, Histogram

# Define metrics
AGENT_REQUESTS = Counter(
    "agent_requests_total", 
    "Total number of agent requests"
)

AGENT_REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds", 
    "Duration of agent requests in seconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600]
) 