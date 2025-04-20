class ShortTermMemory:
    """
    Session-based short-term memory. Chain-of-thought data is kept hidden here.
    """
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key, None)

    def update_chain_of_thought(self, text: str):
        chain = self.data.get("chain_of_thought", "")
        chain += f"\n{text}"
        self.data["chain_of_thought"] = chain

class LongTermMemory:
    """
    Persistent data and previous query/answers can be stored here.
    """
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key, None) 