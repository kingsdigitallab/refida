# build stage
FROM python:3.9-slim as build

ARG poetry="poetry>=1.1,<1.2"
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  # dependencies for building Python packages
  && apt-get install --no-install-recommends -y build-essential openjdk-11-jdk \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

RUN pip install ${poetry}

WORKDIR /tmp

COPY poetry.lock pyproject.toml /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

RUN python -m venv /venv

RUN . /venv/bin/activate \
    && pip install --no-cache-dir --upgrade -r requirements.txt \
    && python -c "import nltk; nltk.download('punkt');"

# base image
FROM python:3.9-slim as base

RUN addgroup --system refida \
  && adduser --system --ingroup refida refida

WORKDIR /app

COPY --from=build --chown=refida:refida /venv /venv

ENV PATH="/venv/bin:$PATH"

USER refida

EXPOSE 8000

ENTRYPOINT ["streamlit", "run"]

# dev image
FROM base as dev

CMD ["streamlit_app.py"]

# production image
FROM base as prod

COPY --chown=refida:refida . /app/

CMD ["streamlit_app.py"]
