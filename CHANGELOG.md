# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - yyyy-mm-dd

### Changed

- Geo data to count/display only unique mentions per case study.
- Group places by country to display data in the map.
- Move all the chart creation to the visualize module.

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
