import re
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Optional

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from joblib import Memory

PROJECT_TITLE = "REF Impact Data Analysis"

ROOT_DIR = Path(".")

DATA_DIR = ROOT_DIR.joinpath("data")
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True)

CACHE_DIR = DATA_DIR.joinpath(".cache")
if not CACHE_DIR.is_dir():
    CACHE_DIR.mkdir(parents=True)

memory = Memory(CACHE_DIR, verbose=0)

# =====================================================================================
# field names to access the data
FIELD_ID = "id"
DATA_UOA = "uoa"
DATA_SUMMARY = "summary"
DATA_RESEARCH = "research"
DATA_DETAILS = "details"
DATA_SOURCES = "sources"
DATA_TEXT = "text"
DATA_ENTITY_SECTIONS: list[str] = [
    DATA_SUMMARY,
    DATA_DETAILS,
    DATA_SOURCES,
]

FEATURE_TOPIC_TOPIC = "topic"
FEATURE_TOPIC_SCORE = "score"

FEATURE_SUMMARY = "summary"

FEATURE_ENTITY_ENTITY = "entity"
FEATURE_ENTITY_LABEL = "label"
FEATURE_ENTITY_TEXT = "text"

FEATURE_GEO_DISPLAY_NAME = "display_name"
FEATURE_GEO_LAT = "lat"
FEATURE_GEO_LON = "lon"
FEATURE_GEO_CATEGORY = "category"
FEATURE_GEO_PLACE = "place"
FEATURE_GEO_PLACE_LAT = "place_lat"
FEATURE_GEO_PLACE_LON = "place_lon"
FEATURE_GEO_GEOJSON = "geojson"


# =====================================================================================
# etl module settings
PARAGRAPH_TYPE_EXCLUDE_PATTERN: re.Pattern = re.compile(
    r"^(Microsoft Word)|(UoA)", re.IGNORECASE
)
PARAGRAPH_CONTENT_EXCLUDE_PATTERN: re.Pattern = re.compile(
    (
        r"^(Microsoft Word.*?docx)|"
        r"(Page \d+)|"
        r"(Impact case study \(REF3\))|"
        r"(Unit-level environment template \(REF5b\))|"
        r"(\(indicative maximum.*?\))$"
    ),
    re.IGNORECASE,
)
PARAGRAPH_CONTENT_REMOVE: re.Pattern = re.compile(
    r"\(indicative maximum.*?words\)",
    re.IGNORECASE,
)


def get_sections_impact_case_study(end: str) -> list[tuple[str, str, str]]:
    return [
        (DATA_SUMMARY, "1. Summary of the impact", "2. Underpinning research"),
        (DATA_RESEARCH, "2. Underpinning research", "3. References to the research"),
        (
            DATA_DETAILS,
            "4. Details of the impact",
            "5. Sources to corroborate the impact",
        ),
        (DATA_SOURCES, "5. Sources to corroborate the impact", end),
    ]


def get_sections_environment(end: str) -> list[tuple[str, str, str]]:
    return [
        (
            DATA_DETAILS,
            "1. Unit context and structure, research and impact strategy",
            "2. People",
        ),
        (
            DATA_SOURCES,
            "4. Collaboration and contribution to the research base, economy and society",
            end,
        ),
    ]


UOA_PATTERN: re.Pattern = re.compile(r"(\d+)")
UOA = {
    "1": "Clinical Medicine",
    "2": "Public Health",
    "3": "Dentistry, Nursing, Pharmacy",
    "4": "Psychology, Psychiatry and Neuroscience",
    "5": "Biological Sciences",
    "8": "Chemistry",
    "9": "Physics",
    "10": "Maths",
    "11": "Informatics",
    "12": "Engineering",
    "14": "Geography",
    "17": "Business and Management Studies",
    "18": "Law",
    "19": "Politics and International Studies",
    "21": "Sociology",
    "23": "Education",
    "24": "Sports and Exercise Science",
    "26": "Modern languages",
    "27": "English",
    "28": "History",
    "29": "Classics",
    "30": "Philosophy",
    "31": "Theology",
    "33": "Film and Music",
    "34": "Communication, Cultural and Media Studies",
}

# =====================================================================================
# features module settings
# model used for topic modelling
TOPIC_CLASSIFICATION_MODEL: str = "joeddav/bart-large-mnli-yahoo-answers"
# labels used for topic modelling
TOPIC_CLASSIFICATION_TOPICS: list[str] = [
    "Cultural",
    "Economic",
    "Environmental",
    "Global",
    "Health",
    "Legal",
    "Policy",
    "Social",
    "Technological",
]

TOPIC_CLASSIFICATION_FIELDS_OF_RESEARCH: dict[str, list[str]] = {
    "AGRICULTURAL, VETERINARY AND FOOD SCIENCES": [
        "Agricultural biotechnology",
        "Agriculture, land and farm management",
        "Animal production",
        "Crop and pasture production",
        "Fisheries sciences",
        "Food sciences",
        "Forestry sciences",
        "Horticultural production",
        "Veterinary sciences",
        "Other agricultural, veterinary and food sciences",
    ],
    "BIOLOGICAL SCIENCES": [
        "Biochemistry and cell biology",
        "Bioinformatics and computational biology",
        "Ecology",
        "Evolutionary biology",
        "Genetics",
        "Industrial biotechnology",
        "Microbiology",
        "Plant biology",
        "Zoology",
        "Other biological sciences",
    ],
    "BIOMEDICAL AND CLINICAL SCIENCES": [
        "Cardiovascular medicine and haematology",
        "Clinical sciences",
        "Dentistry",
        "Immunology",
        "Medical biochemistry and metabolomics",
        "Medical biotechnology",
        "Medical microbiology",
        "Medical physiology",
        "Neurosciences",
        "Nutrition and dietetics",
        "Oncology and carcinogenesis",
        "Ophthalmology and optometry",
        "Paediatrics",
        "Pharmacology and pharmaceutical sciences",
        "Reproductive medicine",
        "Other biomedical and clinical sciences",
    ],
    "BUILT ENVIRONMENT AND DESIGN": [
        "Architecture",
        "Building",
        "Design",
        "Urban and regional planning",
        "Other built environment and design",
    ],
    "CHEMICAL SCIENCES": [
        "Analytical chemistry",
        "Inorganic chemistry",
        "Macromolecular and materials chemistry",
        "Medicinal and biomolecular chemistry",
        "Organic chemistry",
        "Physical chemistry",
        "Theoretical and computational chemistry",
        "Other chemical sciences",
    ],
    "COMMERCE, MANAGEMENT, TOURISM AND SERVICES": [
        "Accounting, auditing and accountability",
        "Banking, finance and investment",
        "Business systems in context",
        "Commercial services",
        "Human resources and industrial relations",
        "Marketing",
        "Strategy, management and organisational behaviour",
        "Tourism",
        "Transportation, logistics and supply chains",
        "Other commerce, management, tourism and services",
    ],
    "CREATIVE ARTS AND WRITING": [
        "Art history, theory and criticism",
        "Creative and professional writing",
        "Music",
        "Performing arts",
        "Screen and digital media",
        "Visual arts",
        "Other creative arts and writing",
    ],
    "EARTH SCIENCES": [
        "Atmospheric sciences",
        "Climate change science",
        "Geochemistry",
        "Geoinformatics",
        "Geology",
        "Geophysics",
        "Hydrology",
        "Oceanography",
        "Physical geography and environmental geoscience",
        "Other earth sciences",
    ],
    "ECONOMICS": [
        "Applied economics",
        "Econometrics",
        "Economic theory",
        "Other economics",
    ],
    "EDUCATION": [
        "Curriculum and pedagogy",
        "Education policy, sociology and philosophy",
        "Education systems",
        "Specialist studies in education",
        "Other education",
    ],
    "ENGINEERING": [
        "Aerospace engineering",
        "Automotive engineering",
        "Biomedical engineering",
        "Chemical engineering",
        "Civil engineering",
        "Communications engineering",
        "Control engineering, mechatronics and robotics",
        "Electrical engineering",
        "Electronics, sensors and digital hardware",
        "Engineering practice and education",
        "Environmental engineering",
        "Fluid mechanics and thermal engineering",
        "Geomatic engineering",
        "Manufacturing engineering",
        "Maritime engineering",
        "Materials engineering",
        "Mechanical engineering",
        "Nanotechnology",
        "Resources engineering and extractive metallurgy",
        "Other engineering",
    ],
    "ENVIRONMENTAL SCIENCES": [
        "Climate change impacts and adaptation",
        "Ecological applications",
        "Environmental biotechnology",
        "Environmental management",
        "Pollution and contamination",
        "Soil sciences",
        "Other environmental sciences",
    ],
    "HEALTH SCIENCES": [
        "Allied health and rehabilitation science",
        "Epidemiology",
        "Health services and systems",
        "Midwifery",
        "Nursing",
        "Public health",
        "Sports science and exercise",
        "Traditional, complementary and integrative medicine",
        "Other health sciences",
    ],
    "HISTORY, HERITAGE AND ARCHAEOLOGY": [
        "Archaeology",
        "Heritage, archive and museum studies",
        "Historical studies",
        "Other history, heritage and archaeology",
    ],
    "HUMAN SOCIETY": [
        "Anthropology",
        "Criminology",
        "Demography",
        "Development studies",
        "Gender studies",
        "Human geography",
        "Policy and administration",
        "Political science",
        "Social work",
        "Sociology",
        "Other human society",
    ],
    "INDIGENOUS STUDIES": [
        "Indigenous studies",
    ],
    "INFORMATION AND COMPUTING SCIENCES": [
        "Applied computing",
        "Artificial intelligence",
        "Computer vision and multimedia computation",
        "Cybersecurity and privacy",
        "Data management and data science",
        "Distributed computing and systems software",
        "Graphics, augmented reality and games",
        "Human-centred computing",
        "Information systems",
        "Library and information studies",
        "Machine learning",
        "Software engineering",
        "Theory of computation",
        "Other information and computing sciences",
    ],
    "LANGUAGE, COMMUNICATION AND CULTURE": [
        "Communication and media studies",
        "Cultural studies",
        "Language studies",
        "Linguistics",
        "Literary studies",
        "Other language, communication and culture",
    ],
    "LAW AND LEGAL STUDIES": [
        "Commercial law",
        "Environmental and resources law",
        "International and comparative law",
        "Law in context",
        "Legal systems",
        "Private law and civil obligations",
        "Public law",
        "Other law and legal studies",
    ],
    "MATHEMATICAL SCIENCES": [
        "Applied mathematics",
        "Mathematical physics",
        "Numerical and computational mathematics",
        "Pure mathematics",
        "Statistics",
        "Other mathematical sciences",
    ],
    "PHILOSOPHY AND RELIGIOUS STUDIES": [
        "Applied ethics",
        "History and philosophy of specific fields",
        "Philosophy",
        "Religious studies",
        "Theology",
        "Other philosophy and religious studies",
    ],
    "PHYSICAL SCIENCES": [
        "Astronomical sciences",
        "Atomic, molecular and optical physics",
        "Classical physics",
        "Condensed matter physics",
        "Medical and biological physics",
        "Nuclear and plasma physics",
        "Particle and high energy physics",
        "Quantum physics",
        "Space sciences",
        "Synchrotrons and accelerators",
        "Other physical sciences",
    ],
    "PSYCHOLOGY": [
        "Applied and developmental psychology",
        "Biological psychology",
        "Clinical and health psychology",
        "Cognitive and computational psychology",
        "Social and personality psychology",
        "Other psychology",
    ],
}


def get_fields_of_research() -> list[str]:
    return list(chain.from_iterable(TOPIC_CLASSIFICATION_FIELDS_OF_RESEARCH.values()))


TOPIC_CLASSIFICATION_IMPACTS: list[str] = [
    "Publications",
    "Journal articles",
    "Conference proceedings",
    "Reports",
    "Policy briefs",
    "Book",
    "Guide",
    "Collaborations and partnerships",
    "Research grant",
    "Fellowship",
    "Co-funding",
    "Capital funding",
    "Travel funding",
    "Engagement activities",
    "Working group",
    "Expert panel",
    "Talk",
    "Magazine",
    "Event",
    "Open day",
    "Media interaction",
    "Blog",
    "Social media",
    "Broadcast",
    "Influence on policy",
    "Letter to parliament",
    "Training of policymakers",
    "Citation in guidance",
    "Citation in policy",
    "Evidence to government",
    "Consultations",
    "Influence on business",
    "Citation in working procedures",
    "Revisions to guidance docs",
    "Citation in industry report",
    "Article in trade press",
    "Talk at trade event",
    "Research tools and methods",
    "Research databases and models",
    "Intellectual property and licensing",
    "Copyright",
    "Patent application",
    "Trademark",
    "Open source",
    "Artistic and creative products",
    "Image",
    "Artwork",
    "Creative writing",
    "Music score",
    "Animation",
    "Exhibition",
    "Performance",
    "Software and technical products",
    "Software",
    "Web application",
    "Improved technology",
    "Spin-outs",
    "Awards and recognition",
    "Research prize",
    "Honorary membership",
    "Editor of journal",
    "National honour",
    "Use of facilities and resources",
]


# model used for summarisation
SUMMARISATION_MODEL: str = "sshleifer/distilbart-cnn-12-6"

# https://spacy.io/models
SPACY_LANGUAGE_MODEL: str = "en_core_web_trf"

SPACY_EXTRA_STOP_WORDS: list[str] = ["Miss", "Mr", "Mrs", "Ms"]

# https://spacy.io/models/en#en_core_web_sm-labels
SPACY_LOCATION_ENTITY_TYPES: list[str] = ["GPE", "LOC"]
SPACY_ENTITY_TYPES: list[str] = SPACY_LOCATION_ENTITY_TYPES + [
    "ORG",
    "NORP",
    "PRODUCT",
]


nominatim = Nominatim(user_agent="kdl.kcl.ac.uk")
geolocator = RateLimiter(nominatim.geocode, min_delay_seconds=1)


@lru_cache
def get_place_category(name: str, country: str) -> Optional[str]:
    """
    Returns the place category for a given place and country.

    :name: the name of the place
    :country: the country to get the category for.
    """
    if not name:
        return None

    if name == "London":
        return "Local"

    if country == "United Kingdom":
        return "National"

    return "Global"
