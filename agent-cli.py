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

import requests

DEFAULT_API_URL = "http://localhost:8000"

def print_colorized(key, value, indent=0):
    """Print key-value pairs with colors for better readability."""
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "red": "\033[91m",
    }
    
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
    print(f"DEBUG: Using endpoint URL: {endpoint}")
    
    try:
        response = requests.post(
            endpoint,
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        if hasattr(e, "response") and e.response:
            try:
                error_data = e.response.json()
                return {"error": error_data.get("detail", str(e))}
            except:
                return {"error": str(e)}
        return {"error": str(e)}

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

def main():
    parser = argparse.ArgumentParser(description="Interact with the code agent API")
    parser.add_argument("query", nargs="?", help="Query to send to the agent")
    parser.add_argument("--process", "-p", help="Process a directory of code files")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint URL")
    
    args = parser.parse_args()
    
    if args.process:
        print(f"Processing directory: {args.process}")
        result = process_directory(args.process, api_url=args.api_url)
        print("\nResult:")
        print_colorized("process_result", result)
        return
    
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
                print("\nAgent Response:")
                
                if "error" in result and result["error"]:
                    print_colorized("error", result["error"])
                else:
                    if "analysis" in result:
                        print_colorized("analysis", result["analysis"])
                    
                    if "code_fix" in result:
                        print_colorized("code_fix", result["code_fix"])
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        print(f"Sending query: {args.query}")
        result = run_agent_query(args.query, api_url=args.api_url)
        
        print("\nAgent Response:")
        if "error" in result and result["error"]:
            print_colorized("error", result["error"])
        else:
            if "analysis" in result:
                print_colorized("analysis", result["analysis"])
            
            if "code_fix" in result:
                print_colorized("code_fix", result["code_fix"])

if __name__ == "__main__":
    main() 