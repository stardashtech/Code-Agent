import requests
import json
import sys

def test_agent_api():
    """Test the agent API by sending a simple query and checking the response."""
    print("Testing Agent API...")
    
    # Test query
    query = "Check the syntax of this Python function: def calculate_average(numbers): return sum(numbers) / len(numbers)"
    
    # Make request to API
    try:
        response = requests.post(
            "http://localhost:8000/api/agent/run",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response received successfully!")
            
            # Print response fields
            print("\nAnalysis:")
            print(json.dumps(result.get("analysis", {}), indent=2))
            
            print("\nCode Fix:")
            print(json.dumps(result.get("code_fix", {}), indent=2))
            
            print("\nCode Snippets:")
            snippets = result.get("code_snippets", [])
            print(f"Found {len(snippets)} snippets")
            
            # Check if there are any errors in the response
            if "error_type" in result.get("analysis", {}):
                error_type = result["analysis"]["error_type"]
                print(f"\nError detected: {error_type}")
                print(f"Description: {result['analysis'].get('description', 'No description')}")
                
                # Consider 'syntax_error' as a valid result for our mock
                if error_type in ["initialization_error", "api_error", "processing_error"]:
                    return False
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_agent_api()
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1) 