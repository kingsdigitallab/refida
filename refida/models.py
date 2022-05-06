from typing import Optional, Union

from pydantic import BaseModel


class REFDocument(BaseModel):
    id: str
    type: str
    uoa_n: Optional[int] = None
    uoa: Optional[str] = None
    title: Optional[str] = None
    names: Optional[list[str]] = None
    research_start: Optional[int] = None
    research_end: Optional[int] = None
    impact_start: Optional[int] = None
    impact_end: Optional[int] = None
    summary: Optional[str] = None
    research: Optional[str] = None
    details: Optional[str] = None
    sources: Optional[str] = None
    text: Optional[str] = None
    compressed: Optional[bytes] = None
    file: str

    def set_field(self, field: str, value: Union[int, str]):
        """
        Set a field of the document.

        :param field: field name
        :param value: value to set
        """
        setattr(self, field, value)
