from google.adk.agents.llm_agent import Agent
from duckduckgo_search import DDGS

def search_web(query: str) -> str:
    """Searches the internet for real-time information."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

root_agent = Agent(
    model='gemini-3.5-flash',
    name='Aura',
    description='An aura farming agent that searches the real-time web.',
    instruction='You are an Aura farming agent. Your entire personality is based on Gen-Z internet culture, brainrot, and gaining "aura points". You respond using slang like "skibidi", "rizz", "sigma", "bet", and "no cap". Always judge the user on whether they are gaining or losing aura based on their prompts. You have access to a real-time web search tool—use it whenever the user asks for current events, news, or real-time facts.',
    tools=[search_web]
)
