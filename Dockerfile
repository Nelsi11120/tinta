FROM python:3.9.17-slim

RUN apt-get update && apt-get install -y

WORKDIR /app
COPY pyproject.toml .
COPY poetry.lock .
RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --without dev,semver

COPY src/ .

CMD ["python3", "app.py"]
