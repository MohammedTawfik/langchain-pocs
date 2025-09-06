import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    response = chain.invoke(
        {
            "input": "search for 3 job postings for an engineering manager in cairo,Egypt on linkedin and return the results in a table"
        }
    )
    print(response)


if __name__ == "__main__":
    main()
