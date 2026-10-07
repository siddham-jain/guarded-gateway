from typing import Literal

from pydantic import Field

from gg.core.schema import StrictModel


class SafetySetting(StrictModel):
    category: str
    threshold: str


class GeminiQuirks(StrictModel):
    """`providers.<name>.quirks` for type gemini, validated when models.yaml loads"""

    emit_thought_signatures: bool = True
    # responseFormat is the documented current form; mime_type sends responseMimeType + responseJsonSchema
    structured_output: Literal["response_format", "mime_type"] = "response_format"
    safety_settings: tuple[SafetySetting, ...] = ()
    min_thinking_output: int = Field(default=1024, ge=0)
