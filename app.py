"""Streamlit chat UI for a Groq research and weather agent.

Uses standard HTTP calls instead of LangChain to avoid SQLAlchemy's optional native
extension, which Windows Application Control can block.
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import streamlit as st

APP_HEADERS = {"User-Agent": "research-agent-streamlit/1.0"}


def load_env_file() -> None:
    """Load simple KEY=value entries from .env without python-dotenv."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **APP_HEADERS, **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed ({error.code}): {detail}") from error
    except URLError as error:
        raise RuntimeError("Could not reach the external service.") from error


def get_weather_data(city: str) -> str:
    """Fetch current weather information from Weatherstack."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather service is not configured. Add WEATHERSTACK_API_KEY to .env."
    try:
        request = Request(
            "https://api.weatherstack.com/current?"
            + urlencode({"access_key": api_key, "query": city}),
            headers=APP_HEADERS,
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError):
        return "Could not reach the weather service. Please try again."

    if "current" not in data:
        return data.get("error", {}).get("info", f"Could not fetch weather for {city}.")
    current = data["current"]
    description = current.get("weather_descriptions", ["Unknown"])[0]
    location = data.get("location", {}).get("name", city)
    return (
        f"City: {location}\nTemperature: {current['temperature']}°C\n"
        f"Weather: {description}\nHumidity: {current['humidity']}%"
    )


def search_web(query: str) -> str:
    """Search the web using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is not configured. Add TAVILY_API_KEY to .env."
    data = post_json(
        "https://api.tavily.com/search",
        {"api_key": api_key, "query": query, "max_results": 2},
    )
    results = data.get("results", [])
    if not results:
        return "No search results found."
    return "\n\n".join(
        f"Source: {item.get('url', '')}\n{item.get('content', '')}" for item in results
    )


TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for timely or factual information.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather_data", "description": "Get current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def run_agent(user_input: str) -> str:
    """Run a small tool-calling loop against Groq's OpenAI-compatible API."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing required .env value: GROQ_API_KEY")
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Use web search when needed."},
        {"role": "user", "content": user_input},
    ]
    functions = {"search_web": search_web, "get_weather_data": get_weather_data}
    for _ in range(4):
        data = post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": "openai/gpt-oss-120b", "temperature": 0, "messages": messages, "tools": TOOLS},
            {"Authorization": f"Bearer {groq_api_key}"},
        )
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or "I could not generate a response."
        messages.append(message)
        for call in tool_calls:
            function = call["function"]
            try:
                result = functions[function["name"]](**json.loads(function["arguments"]))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                result = f"Tool call failed: {error}"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return "I reached the maximum number of tool calls. Please try a more specific question."


load_env_file()
st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Research Agent")
st.caption("Ask research questions or request current weather information.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Researching..."):
                answer = run_agent(user_input)
            st.markdown(answer)
        except ValueError as error:
            answer = f"Configuration error: {error}"
            st.error(answer)
        except Exception as error:
            answer = "I couldn't complete that request. Check your API keys and try again."
            st.error(f"{answer}\n\nDetails: {error}")
    st.session_state.messages.append({"role": "assistant", "content": answer})
