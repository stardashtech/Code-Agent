#!/bin/bash
set -e

echo "===== Testing Agent API ====="

# 1. Ensure server is running
echo "Checking if server is running..."
curl -s http://localhost:8000/healthz > /dev/null || {
    echo "Error: Server is not running. Start the server with './run.sh'"
    exit 1
}

# 2. Process test directory
echo "Processing test-data directory..."
curl -s -X POST http://localhost:8000/api/code/process-directory \
    -F "directory=./test-data" \
    -F "languages=py" > /dev/null || {
    echo "Error: Failed to process directory"
    exit 1
}

echo "Waiting for processing to complete (5 seconds)..."
sleep 5

# 3. Test agent with different errors
echo "Testing Division by Zero Error..."
response=$(curl -s -X POST http://localhost:8000/api/agent/run \
    -H "Content-Type: application/json" \
    -d '{"query": "calculate_average fonksiyonum boş liste ile çağrıldığında bir hata veriyor. Lütfen bu hatayı düzelt."}')

echo "Agent Response:"
echo "$response" | jq .

# 4. Test Agent with NameError
echo -e "\nTesting NameError for undefined variable..."
response=$(curl -s -X POST http://localhost:8000/api/agent/run \
    -H "Content-Type: application/json" \
    -d '{"query": "get_birth_year metodumda bir hata alıyorum. current_year değişkeni tanımlanmamış."}')

echo "Agent Response:"
echo "$response" | jq .

# 5. Test MCP Interface
echo -e "\nTesting MCP Interface..."
response=$(curl -s -X POST http://localhost:8000/api/agent/mcp \
    -H "Content-Type: application/json" \
    -d '{
        "message": "process_data fonksiyonunda AttributeError alıyorum",
        "context": {
            "session_id": "test-session",
            "code_context": [
                {
                    "file_path": "test-data/error_code.py",
                    "language": "python"
                }
            ]
        }
    }')

echo "MCP Response:"
echo "$response" | jq .

echo -e "\n===== Test Complete =====" 