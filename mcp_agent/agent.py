import datetime
import os
import sys
from zoneinfo import ZoneInfo
import opik
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from opik.integrations.adk import OpikTracer

# --- Constants and Centralized Data ---

AGENT_MODEL = "gemini-1.5-flash"
AGENT_NAME = "weather_time_city_agent"

# Centralized data store for all city-related information.
# This avoids data duplication and recreation on each function call.
CITY_DATA = {
    "newyork": {
        "display_name": "New York",
        "weather_report": "The weather in New York is sunny with a temperature of 45°F.",
        "timezone": "America/New_York",
    },
    "london": {
        "display_name": "London",
        "weather_report": "It's cloudy in London with a temperature of 55°F.",
        "timezone": "Europe/London",
    },
    "tokyo": {
        "display_name": "Tokyo",
        "weather_report": "Tokyo is experiencing light rain and a temperature of 72°F.",
        "timezone": "Asia/Tokyo",
    },
}

# --- Configuration Validation ---

# Proactively check for the required API key to prevent downstream errors.
# The script will exit with a clear message if the key is missing.
Maps_API_KEY = os.environ.get("Maps_PLATFORM_API_KEY")
if not Maps_API_KEY:
    sys.exit("Error: Environment variable 'Maps_PLATFORM_API_KEY' is not set.")

# --- Tool Functions ---

def get_weather(city: str) -> dict:
    """
    Retrieves mock weather information for a given city.

    Args:
        city: The name of the city.

    Returns:
        A dictionary with the status and weather report or an error message.
    """
    print(f'. [TOOL] "get_weather" - City: {city}.')
    city_normalized = city.lower().replace(" ", "")

    if city_data := CITY_DATA.get(city_normalized):
        return {"status": "success", "report": city_data["weather_report"]}
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }


def get_current_time(city: str) -> dict:
    """
    Calculates the current time in a given city using its timezone.

    Args:
        city: The name of the city.

    Returns:
        A dictionary with the status and time report or an error message.
    """
    print(f'. [TOOL] "get_current_time" - City: {city}.')
    city_normalized = city.lower().replace(" ", "")

    if not (city_data := CITY_DATA.get(city_normalized)):
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have timezone information for '{city}'.",
        }

    try:
        tz = ZoneInfo(city_data["timezone"])
        now = datetime.datetime.now(tz)
        display_name = city_data["display_name"]
        report = f"The current time in {display_name} is {now.strftime('%H:%M')}."
        return {"status": "success", "report": report}
    except Exception as e:
        # Catch potential errors from ZoneInfo or datetime processing.
        return {
            "status": "error",
            "error_message": f"Could not retrieve time for '{city}'. Reason: {e}",
        }

# --- Agent Initialization ---

opik.configure(use_local=False)
opik_tracer = OpikTracer()

root_agent = LlmAgent(
    name=AGENT_NAME,
    model=AGENT_MODEL,
    description=(
        "Agent to answer questions about the time and weather in a city and "
        "provide directions between two cities."
    ),
    instruction=(
        "You are a helpful assistant. When the user asks for "
        "a specific city, use the 'get_weather' and the "
        "'get_current_time' tools to find the weather and current time "
        "information. If the tools return an error, inform the user. "
        "If the tools are successful, present the report clearly."
    ),
    tools=[
        get_weather,
        get_current_time,
        MCPToolset(
            connection_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-google-maps",
                ],
                env={"Maps_API_KEY": Maps_API_KEY},
            ),
        ),
    ],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)
