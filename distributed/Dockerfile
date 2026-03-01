FROM python:3.11-slim

# System dependencies
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir GDAL==$(gdal-config --version) && \
    pip install --no-cache-dir -r requirements.txt

# Install the package
COPY pyproject.toml ./
COPY paleoeurope/ ./paleoeurope/
COPY distributed/ ./distributed/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["scripts/run_full_pipeline.py", "--help"]
