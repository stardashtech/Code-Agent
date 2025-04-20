from agent.agent import CodeAgent

def main():
    # Example user query
    user_query = (
        "I installed a new library and I'm getting a TypeError. "
        "Please determine the steps to fix my code."
    )
    agent = CodeAgent()
    result = agent.run(user_query)
    print("=== Final Agent Output ===")
    print(result)

if __name__ == "__main__":
    main() 