# Security

## Authentication

API keys for LLM providers must be stored in `.env` and never committed.

For Google Cloud, use Application Default Credentials and least-privilege service accounts when reading from `gs://` buckets.

## Data

Source documents under `docs/` should exclude secrets. Redact tokens before indexing.
