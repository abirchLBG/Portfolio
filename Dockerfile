FROM python:3.13

# Install dependencies
WORKDIR /app
COPY pyproject.toml uv.lock ./
# RUN pip install uv && uv sync --no-cache
RUN pip install uv && uv sync

# Copy the source code
COPY src ./src
COPY data ./data

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
# CMD 
