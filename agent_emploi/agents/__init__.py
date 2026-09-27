"""Agents LLM : les seuls modules qui appellent un modèle.

Chacun a une responsabilité unique, reçoit son routeur par injection et rend un
objet Pydantic validé. Ils sont donc testables sans réseau, en passant un
routeur scripté.
"""

from agent_emploi.agents.fit import FitAgent
from agent_emploi.agents.gate import GateAgent
from agent_emploi.agents.letter import LetterAgent
from agent_emploi.agents.review import ReviewAgent

__all__ = ["FitAgent", "GateAgent", "LetterAgent", "ReviewAgent"]
