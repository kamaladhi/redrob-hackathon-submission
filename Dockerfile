FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the python scripts, pre-computed artifacts, and a sample input file
COPY rank.py .
COPY candidate_index.faiss .
COPY candidate_metadata.pkl .
COPY small_candidates.jsonl .

# Ensure the script can be executed
CMD ["python", "rank.py", "--candidates", "small_candidates.jsonl", "--out", "submission.csv"]
