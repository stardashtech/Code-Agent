import requests
import json
import sys

def test_mcp_api():
    """Test the Model Context Protocol (MCP) API endpoint."""
    print("Testing MCP API...")
    
    # Test message
    message = "Bir dizideki en büyük sayıyı bulan kod nasıl yazılır?"
    context = {
        "previous_messages": [],
        "language_preference": "python"
    }
    
    # Make request to API
    try:
        response = requests.post(
            "http://localhost:8000/api/agent/mcp",
            json={"message": message, "context": context},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response received successfully!")
            
            # Print result
            print("\nResult:")
            print(json.dumps(result.get("result", {}), indent=2))
            
            # Print MCP context
            print("\nMCP Context:")
            print(json.dumps(result.get("mcp_context", {}), indent=2))
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_mcp_api()
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1) 