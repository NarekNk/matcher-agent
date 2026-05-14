FROM python:3.11-slim

WORKDIR /app

# System deps for inference:
# - ffmpeg: required by openai-whisper to decode Spotify preview MP3s.
# - libgomp1: OpenMP runtime; PyTorch / many scientific wheels expect it on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Install matcher runtime package from a remote source so this app can stay isolated.
# Example:
#   --build-arg MATCHER_AGENT_PIP_SPEC="git+https://github.com/<org>/<repo>.git@<sha>"
ARG MATCHER_AGENT_PIP_SPEC=""
RUN if [ -n "$MATCHER_AGENT_PIP_SPEC" ]; then pip install --no-cache-dir "$MATCHER_AGENT_PIP_SPEC"; fi

COPY main.py /app/main.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
