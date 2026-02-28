"""Prompt templates shared across chain builders."""

memory_prompt_template = """
<s>[INST] You are an AI chatbot having a conversation with a human. Answer questions clearly.
Previous conversation: {history}
Human: {human_input}
AI: [/INST]
""".strip()
