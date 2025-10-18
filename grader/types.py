from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class Trajectory(BaseModel):
    id: Optional[str] = None
    messages: List[Message]


class GradingResult(BaseModel):
    trajectory_id: str
    grader_model: str
    score: float
