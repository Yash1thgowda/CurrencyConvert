import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# --- 1. Define Tool ---

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value from one supported unit to another."""

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,

        ("m", "feet"): lambda x: x * 3.28084,
        ("feet", "m"): lambda x: x * 0.3048,

        ("cm", "inches"): lambda x: x * 0.393701,
        ("inches", "cm"): lambda x: x * 2.54,

        ("kg", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kg"): lambda x: x * 0.453592,

        ("g", "kg"): lambda x: x / 1000,
        ("kg", "g"): lambda x: x * 1000,

        ("seconds", "minutes"): lambda x: x / 60,
        ("minutes", "seconds"): lambda x: x * 60,

        ("minutes", "hours"): lambda x: x / 60,
        ("hours", "minutes"): lambda x: x * 60,

        ("hours", "days"): lambda x: x / 24,
        ("days", "hours"): lambda x: x * 24,
    }

    # Temperature conversions
    if from_unit in ["celsius", "c"] and to_unit in ["fahrenheit", "f"]:
        result = (value * 9 / 5) + 32
    elif from_unit in ["fahrenheit", "f"] and to_unit in ["celsius", "c"]:
        result = (value - 32) * 5 / 9
    elif from_unit in ["celsius", "c"] and to_unit in ["kelvin", "k"]:
        result = value + 273.15
    elif from_unit in ["kelvin", "k"] and to_unit in ["celsius", "c"]:
        result = value - 273.15
    elif (from_unit, to_unit) in conversions:
        result = conversions[(from_unit, to_unit)](value)
    else:
        return f"Conversion from {from_unit} to {to_unit} is not supported."

    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"


tools = [unit_converter]


# --- 2. Initialize Model & Agent ---

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a Unit Conversion Agent. "
        "Your job is to convert values between supported units. "
        "Use the unit_converter tool whenever a conversion is requested. "
        "Supported conversions include length, weight, temperature, and time. "
        "If the requested conversion is not supported, clearly tell the user."
    )
)


# --- 3. Input / Output Formatting ---

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))

        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return str(content)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)


# --- 4. FastAPI App ---

app = FastAPI(
    title="Unit Converter Agent",
    version="1.0",
    description="A LangChain agent powered by Gemini with a unit conversion tool."
)


@app.get("/")
def root():
    return {
        "message": "Server is running. Visit /agent/playground/ to chat, or /docs for the API."
    }


add_routes(app, formatted_agent_chain, path="/agent")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
