#!/usr/bin/env python3
"""
CLI tool for interacting with the code agent API.
Usage:
    python agent-cli.py "Fonksiyonum boş liste ile çağrıldığında hata veriyor"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin
import re
import logging

import requests

DEFAULT_API_URL = "http://localhost:8000"

# Setup logger for the CLI script itself
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s') # Basic config

# Color definitions globally
colors = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m"
}

def print_colorized(key, value, indent=0):
    """Print key-value pairs with colors for better readability."""
    indent_str = " " * indent
    
    if isinstance(value, dict):
        print(f"{indent_str}{colors['bold']}{colors['blue']}{key}:{colors['reset']}")
        for k, v in value.items():
            print_colorized(k, v, indent + 2)
    elif isinstance(value, list):
        print(f"{indent_str}{colors['bold']}{colors['blue']}{key}:{colors['reset']}")
        for i, item in enumerate(value):
            if isinstance(item, dict):
                print(f"{indent_str}  {colors['bold']}[{i}]:{colors['reset']}")
                for k, v in item.items():
                    print_colorized(k, v, indent + 4)
            else:
                print(f"{indent_str}  [{i}]: {item}")
    else:
        if key == "error" and value:
            value_str = f"{colors['red']}{value}{colors['reset']}"
        elif key == "explanation":
            value_str = f"{colors['green']}{value}{colors['reset']}"
        elif key in ["file_path", "language"]:
            value_str = f"{colors['cyan']}{value}{colors['reset']}"
        elif "original" in key:
            value_str = f"{colors['red']}{value}{colors['reset']}"
        elif "replacement" in key:
            value_str = f"{colors['green']}{value}{colors['reset']}"
        else:
            value_str = str(value)
            
        print(f"{indent_str}{colors['bold']}{key}:{colors['reset']} {value_str}")

def run_agent_query(query, api_url=DEFAULT_API_URL):
    """Send a query to the agent API and return the result."""
    endpoint = urljoin(api_url, "api/agent/run")
    # Removed debug print of endpoint, less noisy
    # print(f"DEBUG: Using endpoint URL: {endpoint}") 
    
    try:
        response = requests.post(
            endpoint,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=60 # Add a reasonable timeout for the API call itself
        )
        response.raise_for_status() # Raises HTTPError for 4xx/5xx responses
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"API call to {endpoint} timed out.")
         return {"error": f"API call timed out. The agent might be busy or unresponsive at {api_url}."}
    except requests.exceptions.ConnectionError as e:
         logger.error(f"API call to {endpoint} failed: Connection Error: {e}")
         return {"error": f"Could not connect to the agent API at {api_url}. Is the server running?"}
    except requests.exceptions.HTTPError as e:
        # Handle errors specifically raised by raise_for_status()
        error_msg = f"API returned an error: {e.response.status_code} {e.response.reason}"
        logger.error(f"API call to {endpoint} failed: {error_msg}")
        try:
             # Try to get more details from the response body
             detail = e.response.json().get("detail", "No details provided.")
             error_msg += f" - Detail: {detail}"
        except json.JSONDecodeError:
             error_msg += f" - Response body: {e.response.text[:100]}..."
        return {"error": error_msg} 
    except requests.exceptions.RequestException as e:
        # Catch other potential request errors
        logger.error(f"API call to {endpoint} failed: {e}")
        return {"error": f"An unexpected error occurred while contacting the agent API: {e}"} 
    except Exception as e:
        # Catch unexpected errors (e.g., JSON decoding of success response)
        logger.error(f"Error processing API response from {endpoint}: {e}", exc_info=True)
        return {"error": f"Failed to process the response from the agent API: {e}"}

def process_directory(directory, languages=None, api_url=DEFAULT_API_URL):
    """Process a directory of code files."""
    endpoint = urljoin(api_url, "api/code/process-directory")
    
    if languages is None:
        languages = ["py", "js", "cs", "java", "cpp"]
    
    try:
        response = requests.post(
            endpoint,
            data={"directory": directory},
            files={"languages": (None, json.dumps(languages))}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error processing directory: {str(e)}")
        if hasattr(e, "response") and e.response:
            try:
                error_data = e.response.json()
                return {"error": error_data.get("detail", str(e))}
            except:
                return {"error": str(e)}
        return {"error": str(e)}

def print_final_answer(answer_text):
    """Prints the final answer, highlighting markdown code blocks."""
    code_block_pattern = r"```(?:[a-zA-Z0-9\+\-\_]*)\n(.*?)\n```"
    matches = list(re.finditer(code_block_pattern, answer_text, re.DOTALL))
    
    last_end = 0
    if not matches:
        # No code blocks found, print the whole answer normally
        print(answer_text)
    else:
        for match in matches:
            start, end = match.span()
            # Print text before the code block
            print(answer_text[last_end:start].strip())
            
            # Print the code block highlighted
            code_content = match.group(1)
            print(f"{colors['magenta']}--- Suggested Code ---{colors['reset']}")
            print(f"{colors['cyan']}{code_content.strip()}{colors['reset']}")
            print(f"{colors['magenta']}--------------------{colors['reset']}")
            
            last_end = end
        # Print any text after the last code block
        if last_end < len(answer_text):
            print(answer_text[last_end:].strip())

def main():
    parser = argparse.ArgumentParser(description="Interact with the code agent API")
    parser.add_argument("query", nargs="?", help="Query to send to the agent")
    parser.add_argument("--process", "-p", help="Process a directory of code files")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full response including plan and trace")

    args = parser.parse_args()

    if args.process:
        print(f"Processing directory: {args.process}")
        result = process_directory(args.process, api_url=args.api_url)
        print("\nResult:")
        print_colorized("process_result", result)
        return

    def handle_response(result, verbose=False):
        """Handles printing the response based on status."""
        print("\nAgent Response:")
        status = result.get("status")
        error = result.get("error")
        message = result.get("message")
        final_answer = result.get("final_answer")
        plan = result.get("plan")
        trace = result.get("execution_trace")

        if error:
            print_colorized("error", error)
            return # Stop further processing if there's a top-level error
            
        if status == "clarification_needed":
            print(f"{colors['yellow']}*** Clarification Needed ***{colors['reset']}")
            print_final_answer(message or "The agent requires clarification to proceed.")
            print(f"\n{colors['yellow']}Please refine your query based on the message above and try again.{colors['reset']}")
            if verbose:
                print("\n--- Context & Partial Execution --- ")
                if plan:
                    print_colorized("initial_plan", plan)
                if trace:
                    print("\nExecution Summary:")
                    for i, step in enumerate(trace):
                        tool = step.get("result", {}).get("tool_used", "unknown") # Get tool from result dict
                        step_result = step.get("result", {}).get("result", "No result recorded")
                        summary = str(step_result)[:100] + ("..." if len(str(step_result)) > 100 else "")
                        print(f"  Step {i+1}: {step.get('subgoal', 'Unknown subgoal')[:60]}... [{colors['blue']}{tool}{colors['reset']}] -> {colors['cyan']}{summary}{colors['reset']}")
                        
        elif status == "completed":
            print(f"{colors['green']}*** Task Completed ***{colors['reset']}")
            if final_answer:
                print_final_answer(final_answer)
            else:
                print("(No final answer provided)")

            if verbose:
                print("\n--- Execution Details --- ")
                if plan:
                    print_colorized("initial_plan", plan)
                if trace:
                    print("\nExecution Summary:")
                    for i, step in enumerate(trace):
                        tool = step.get("result", {}).get("tool_used", "unknown")
                        step_result = step.get("result", {}).get("result", "No result recorded")
                        summary = str(step_result)[:100] + ("..." if len(str(step_result)) > 100 else "")
                        print(f"  Step {i+1}: {step.get('subgoal', 'Unknown subgoal')[:60]}... [{colors['blue']}{tool}{colors['reset']}] -> {colors['cyan']}{summary}{colors['reset']}")
        else:
            print(f"{colors['red']}*** Unknown Status ***{colors['reset']}")
            print("The agent returned an unexpected response structure.")
            print_colorized("full_response", result)

    if not args.query:
        print("Interactive mode. Type 'exit' or 'quit' to exit.")
        while True:
            try:
                query = input("\n> ")
                if query.lower() in ["exit", "quit"]:
                    break
                if not query.strip():
                    continue

                print("Sending query to agent...")
                result = run_agent_query(query, api_url=args.api_url)
                handle_response(result, args.verbose)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        print(f"Sending query: {args.query}")
        result = run_agent_query(args.query, api_url=args.api_url)
        handle_response(result, args.verbose)

if __name__ == "__main__":
    main() 