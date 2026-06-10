# Minimal image for the lc-shift OpenAI-compatible routing proxy.
# Only runtime dependency is pydantic, so the image stays tiny.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

# `docker run lc-shift serve --backend ... --host 0.0.0.0`
ENTRYPOINT ["lc-shift"]
CMD ["serve", "--backend", "http://host.docker.internal:11434/v1", "--host", "0.0.0.0"]
