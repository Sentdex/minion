FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Configure the base url and optional the max no turns
# and polling
ENV MINION_BASE_URL=http://<enter your base url here>/v1 \
    MINION_API_KEY=nokey \
    MINION_BASH_POLL_SECONDS=3 \
    MINION_MAX_MODEL_TURNS=300

# Install git (if cloning during build) and any build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    procps \
    golang \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/Sentdex/minion.git .

# Install dependencies
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
        pip install --no-cache-dir .; \
    fi

#Entry point tiny to prevent zombies
ENTRYPOINT ["tini", "--"]

# Default command (adjust if necessary)
#CMD ["python", "minion.py"]
CMD ["bash"]
