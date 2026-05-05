# Railway FastAPI App (Supabase-backed artifacts)

This folder is a standalone deployment unit for Railway. It:

1. Downloads required inference files from a Supabase bucket on startup.
2. Exposes `/health` and `/recommend` with the same query contract as the current API.
3. Executes the existing recommender command (`matcher_agent.cli.recommend`) via `RECOMMEND_COMMAND`.

## Required environment variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `PREVIEW_RESOLVER_URL` (optional but recommended for audio-feature enrichment)

## Optional environment variables

- `SUPABASE_PREFIX` - bucket folder prefix (example: `models/v2026-05-05`)
- `LOCAL_DATA_ROOT` - local cache root (default: `/app/runtime_data`)
- `REQUIRE_EMBEDDINGS` - `1`/`0` (default `1`)
- `RECOMMEND_COMMAND` - override recommend command (default: `python -m matcher_agent.cli.recommend`)

## Required bucket keys (relative to `SUPABASE_PREFIX`)

- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `data/playlists.parquet`
- `data/historical_matches.parquet`
- `output/training_data.csv`
- `data/embeddings/text_embeddings.parquet` (unless `REQUIRE_EMBEDDINGS=0`)

## Deploy strategy

If you do not want to upload this whole repo to Railway:

1. Upload only this folder as the service source.
2. Provide a package source for `matcher_agent` at build time using:
   - Docker build arg `MATCHER_AGENT_PIP_SPEC`
   - example: `git+https://github.com/<org>/<repo>.git@<commit-sha>`

The `Dockerfile` installs that package if the build arg is non-empty.

## Railway build arg for Option A

Set the Railway build arg `MATCHER_AGENT_PIP_SPEC` so the image installs
`matcher_agent` during build.

- If `matcher-agent` is its own repository:
  - `git+https://github.com/<org>/<matcher-agent-repo>.git@<commit-sha>`
- If it lives in a monorepo subdirectory:
  - `git+https://github.com/<org>/<repo>.git@<commit-sha>#subdirectory=matcher-agent`

Pinning to a commit SHA is recommended to keep deploys reproducible.
