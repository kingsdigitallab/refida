# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2022-06-14

### Fixed

- Filtering by field of research could produce confusing results because it included
  documents without a FoR.

## [0.5.0] - 2022-06-13

### Added

- Explanation about text summarisation.

### Changed

- Display selected document under the about the data section.
- Update Streamlit and txtai to latest versions.
- Sidebar labels.
- Text search filter interface.

## [0.4.0] - 2022-05-30

### Added

- KDL footer.
- Extract unit of assessment panel.
- Panel filter to the dashboard.
- Entity type glossary.
- Entities filter.
- UK nations distribution.
- Command to extract strengths, areas for improvement and future plans.

### Changed

- Rename types of impact to outputs.
- Sort units of assessment by number.
- Group types of output.

## [0.3.1] - 2022-05-17

### Added

- `explain` module to explain the outputs of the classification models.
- Entity type filters to the dashboard.
- Topic based filters to the dashboard.
- About the data section to the dashboard.

## [0.3.0] - 2022-05-10

### Added

- Product to the list of entity types to extract.
- MPL-2.0 license file.
- Tests for `etl` module.
- [pre-commit](https://pre-commit.com/) configuration.
- Processing for environment studies.
- Help information to the dashboard.

### Changed

- Geo data to count/display only unique mentions per case study.
- Group places by country to display data in the map.
- Move all the chart creation to the visualize module.
- Enable click selection in the data grid.
- `etl` module to extract data from environment statements.
- Reduce the number of columns in the data grid.
- Refactor sidebar options.
- Dynamically calculate the plot heights.
- Move text data to the `y` axis for easier readability.

### Removed

- Person from the list of entity types to extract.

### Fixed

- Labels being cut-off in parallel plots.

## [0.2.1] - 2022-04-28

### Added

- Bubble map to the dashboard.
- Module to create visualisations.

### Changed

- Streamlit cache to use the new recommended method of [caching](https://docs.streamlit.io/library/api-reference/performance/st.experimental_memo).

## [0.2.0] - 2022-04-27

### Added

- Chart to show correlation between topics and units of assessment.
- Docker compose files.
- Extract research section to be used to extract fields of research.
- Streamlit configuration file.
- Alluvial diagram to show relationship between topics and units of assessment.
- Country information to the place entities.
- Country related charts.
- Persistent cache for geocoding.
- Mapbox token.

### Changed

- Topic classification to support extracting from different sections and using different labels.
- Enable multi-label topic classification.
- spaCy language model to a transformer based model for better accuracy.
- Geolocate feature to return GeoJSON alongside with the data.

## [0.1.0] - 2022-04-06

### Added

- Command and module to extract, transform and load data.
- Text summarisation.
- Topic classification.
- Dashboard with data, summary and topics chart.
- Entity extraction.
- Entity geolocation.
