from typing import Optional

from pydantic import BaseModel


class REFDocument(BaseModel):
    id: str
    type: str
    uoa: Optional[str] = None
    title: Optional[str] = None
    names: Optional[list[str]] = None
    research_start: Optional[int] = None
    research_end: Optional[int] = None
    impact_start: Optional[int] = None
    impact_end: Optional[int] = None
    summary: Optional[str] = None
    details: Optional[str] = None
    sources: Optional[str] = None
    text: Optional[str] = None
    text_summary: Optional[str] = None
    compressed: Optional[bytes] = None
    file: str

    def set_field(self, field: str, value: str):
        """
        Set a field of the document.

        :param field: field name
        :param value: value to set
        """
        setattr(self, field, value)
