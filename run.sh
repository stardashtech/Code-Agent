#!/bin/bash
set -e

# Create required directories
mkdir -p logs data

# Default values
MODE="dev"
WORKERS=4
RELOAD=true
HOST="0.0.0.0"
PORT=8000

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --prod)
            MODE="prod"
            RELOAD=false
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Please create it based on .env.example."
    exit 1
fi

# Run in development mode (default)
if [ "$MODE" = "dev" ]; then
    echo "Starting in DEVELOPMENT mode"
    exec python -m uvicorn app.main:app --host $HOST --port $PORT --reload
fi

# Run in production mode
if [ "$MODE" = "prod" ]; then
    echo "Starting in PRODUCTION mode with $WORKERS workers"
    exec gunicorn \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        app.main:app
fi 