import json
import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMATTING_INSTRUCTIONS


load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
pydantic_output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
chain = agent_executor
formatting_prompt = PromptTemplate(
    template="""
    Convert the following response into a structured JSON format.
    
    Original Response: {response}
    
    Format it according to this schema:
    {format_instructions}
    
    Return ONLY the JSON object, no additional text or markdown formatting.
    """,
    input_variables=["response"],
    partial_variables={
        "format_instructions": pydantic_output_parser.get_format_instructions()
    },
)
formatting_chain = formatting_prompt | llm | pydantic_output_parser


def main():
    react_response = chain.invoke(
        {
            "input": "search for 3 job postings for an engineering manager in cairo,Egypt on linkedin and return the results in a table"
        }
    )
    print("Raw ReAct Response:")
    print(react_response["output"])
    print("\n" + "=" * 50 + "\n")

    # Format it for UI
    try:
        structured_response = formatting_chain.invoke(
            {"response": react_response["output"]}
        )
        print("Structured Response for UI:")
        print(structured_response)

        # Convert to dict for JSON serialization
        ui_data = structured_response.dict()
        print("\nJSON for UI:")
        print(json.dumps(ui_data, indent=2))

    except Exception as e:
        print(f"Formatting error: {e}")
        # Fallback to raw response
        ui_data = {"response": react_response["output"], "sources": []}
        print("Fallback UI data:")
        print(json.dumps(ui_data, indent=2))


if __name__ == "__main__":
    main()
