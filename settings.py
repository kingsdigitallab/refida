import re
from pathlib import Path

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

ROOT_DIR = Path(".")

DATA_DIR = ROOT_DIR.joinpath("data")
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True)

PROJECT_TITLE = "REF Impact Data Analysis"


PARAGRAPH_EXCLUDE_PATTERN: re.Pattern = re.compile(
    r"^(Page \d+)|(Impact case study \(REF3\))$"
)

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

# model used for summarisation
SUMMARISATION_MODEL: str = "sshleifer/distilbart-cnn-12-6"

# https://spacy.io/models
SPACY_LANGUAGE_MODEL: str = "en_core_web_md"

SPACY_EXTRA_STOP_WORDS: list[str] = ["Miss", "Mr", "Mrs", "Ms"]

# https://spacy.io/models/en#en_core_web_sm-labels
SPACY_LOCATION_ENTITY_TYPES: list[str] = ["GPE", "LOC"]
SPACY_ENTITY_TYPES: list[str] = SPACY_LOCATION_ENTITY_TYPES + [
    "ORG",
    "NORP",
    "PERSON",
]

geolocator = Nominatim(user_agent="kdl.kcl.ac.uk")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# field names to access the data
FIELD_ID = "id"
DATA_SUMMARY = "summary"
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

FEATURE_LAT = "lat"
FEATURE_LON = "lon"
