from pydantic import BaseModel, Field
from typing import List


class Source(BaseModel):
    """The schema of the source used by the agent"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """The schema of the response used by the agent"""

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the response",
    )
    response: str = Field(description="The response of the agent")
