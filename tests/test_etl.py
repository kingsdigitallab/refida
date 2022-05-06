import pytest

from refida import etl
from settings import PARAGRAPH_TYPE_EXCLUDE_PATTERN


def test_is_impact_case_study():
    assert etl.is_impact_case_study("impact case study") is True
    assert etl.is_impact_case_study("environment case study") is False


@pytest.fixture
def paragraph_type_exclude():
    return ["microsoft word", "uoa"]


@pytest.fixture
def paragraph_type_include():
    return ["impact case study (ref3)", "unit-level environment template (ref5b)"]


def test_include_paragraph_type(paragraph_type_exclude, paragraph_type_include):
    for content in paragraph_type_exclude:
        assert etl.include_paragraph(content, PARAGRAPH_TYPE_EXCLUDE_PATTERN) is False

    for content in paragraph_type_include:
        assert etl.include_paragraph(content, PARAGRAPH_TYPE_EXCLUDE_PATTERN) is True


@pytest.fixture
def paragraph_content_exclude():
    return [
        "microsoft word - some file name.docx",
        "page 10",
        "impact case study (ref3)",
        "unit-level environment template (ref5b)",
    ]


@pytest.fixture
def paragraph_content_include():
    return [
        "the microsoft word document",
        "the statement in page 10 explains",
        "in this impact case study",
        "in this unit-level environment template",
    ]


def test_include_paragraph_content(
    paragraph_content_exclude, paragraph_content_include
):
    for content in paragraph_content_exclude:
        assert etl.include_paragraph(content) is False

    for content in paragraph_content_include:
        assert etl.include_paragraph(content) is True


@pytest.fixture
def period_section_name():
    return "period when the underpinning research was undertaken"


@pytest.fixture
def summary_section_name():
    return "1. summary of the impact"


@pytest.fixture
def research_section_name():
    return "2. underpinning research"


@pytest.fixture
def paragraphs_empty():
    return []


@pytest.fixture
def paragraphs_content_invalid(
    period_section_name, summary_section_name, research_section_name
):
    return [
        "impact case study (ref3)",
        "unit of assessment: 99",
        "title of case study:",
        f"{period_section_name}:",
        "names(s):",
        "roles(s):",
        f"{summary_section_name}",
        f"{research_section_name}",
    ]


@pytest.fixture
def paragraphs_content_at_start(
    period_section_name, summary_section_name, research_section_name
):
    return [
        "impact case study (ref3)",
        "unit of assessment: 10 Maths",
        "title of case study: case study title",
        f"{period_section_name}: 2022 - 2022" "name(s):",
        "name 1",
        "name 2",
        "role(s):",
        f"{summary_section_name}",
        "impact summary",
        f"{research_section_name}",
        "research description",
    ]


@pytest.fixture
def paragraphs_content_at_middle(
    period_section_name, summary_section_name, research_section_name
):
    return [
        "impact case study (ref3)",
        "something unit of assessment: 10 Maths",
        "something title of case study: case study title",
        f"something {period_section_name}: 2022 - 2022",
        "something name(s):",
        "name 1",
        "name 2",
        "role(s):",
        f"something {summary_section_name}",
        "impact summary",
        f"impact summary {research_section_name}",
        "research description",
    ]


def test_get_uoa(
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert etl.get_uoa(paragraphs_empty) is None
    with pytest.raises(KeyError):
        etl.get_uoa(paragraphs_content_invalid)

    assert etl.get_uoa(paragraphs_content_at_start) == (10, "Maths")
    assert etl.get_uoa(paragraphs_content_at_middle) == (10, "Maths")


def test_get_paragraph_index(
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert etl.get_paragraph_index(paragraphs_empty, "unit of assessment:") == (-1, -1)
    assert etl.get_paragraph_index(
        paragraphs_content_invalid, "unit of assessment:"
    ) == (1, 0)
    assert etl.get_paragraph_index(
        paragraphs_content_at_start, "unit of assessment:"
    ) == (1, 0)
    assert etl.get_paragraph_index(
        paragraphs_content_at_middle, "unit of assessment:"
    ) == (1, 10)


def test_get_title(
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert etl.get_title(paragraphs_empty) is None
    assert etl.get_title(paragraphs_content_invalid) is None
    assert etl.get_title(paragraphs_content_at_start) == "case study title"
    assert etl.get_title(paragraphs_content_at_middle) == "case study title"


def test_get_names(
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert etl.get_names(paragraphs_empty) is None
    assert etl.get_names(paragraphs_content_invalid) is None
    assert etl.get_names(paragraphs_content_at_start) == ["name 1", "name 2"]
    assert etl.get_names(paragraphs_content_at_middle) == ["name 1", "name 2"]


def test_get_period(
    period_section_name,
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert etl.get_period(paragraphs_empty, period_section_name) is None
    assert etl.get_period(paragraphs_content_invalid, period_section_name) == []
    assert etl.get_period(paragraphs_content_at_start, period_section_name) == [
        2022,
        2022,
    ]
    assert etl.get_period(paragraphs_content_at_middle, period_section_name) == [
        2022,
        2022,
    ]


def test_get_section(
    summary_section_name,
    research_section_name,
    paragraphs_empty,
    paragraphs_content_invalid,
    paragraphs_content_at_start,
    paragraphs_content_at_middle,
):
    assert (
        etl.get_section(paragraphs_empty, summary_section_name, research_section_name)
        is None
    )
    assert (
        etl.get_section(
            paragraphs_content_invalid, summary_section_name, research_section_name
        )
        == ""
    )
    assert (
        etl.get_section(
            paragraphs_content_at_start, summary_section_name, research_section_name
        )
        == "impact summary"
    )
    assert (
        etl.get_section(
            paragraphs_content_at_middle, summary_section_name, research_section_name
        )
        == "impact summary\nimpact summary"
    )
