# Group Members : 
Muhammad Hamza Saeed F23607019 
Hassan Ahmed F23607018
Maha Ayub F23607032
Malaika Dawlat F23607011



# Movie Search Engine

A full-stack NLP search engine over a hand-curated movie corpus, built as an assignment for an NLP course. Search 65 movies by plot, director, cast, or genre using three different ranking algorithms and watch how the rankings change.

The project ships with a FastAPI backend, a React frontend, and a Docker Compose setup so the whole thing runs with a single command.

## Features

- **65 hand-curated movies** with rich 4-sentence plot summaries spanning sci-fi, drama, crime, horror, animation, war, fantasy, and more
- **Three ranking algorithms** you can switch between live:
  - **TF-IDF** with cosine similarity (sublinear TF, L2 normalization)
  - **BM25** (Okapi BM25, the standard probabilistic IR baseline)
  - **Hybrid** with a tunable `α` slider that linearly blends min-max-normalized TF-IDF and BM25 scores
- **Classic NLP preprocessing pipeline**: lowercase → punctuation strip → regex tokenize → English stopword removal → Porter stemming
- **Live search** — debounced queries, query-term highlighting in result snippets, score display, example query chips
- **REST API** with JSON responses and Pydantic schemas
- **Single-command deploy** with Docker Compose (Nginx reverse-proxies `/api` to the backend)

## Stack

| Layer    | Tools                                                                       |
| -------- | --------------------------------------------------------------------------- |
| Backend  | Python 3.12, FastAPI, Uvicorn, scikit-learn, rank-bm25, NLTK (Porter only)  |
| Frontend | React 18, Vite 5, plain CSS (no Tailwind)                                   |
| Serving  | Nginx (production), Vite dev server (development)                           |
| DevOps   | Docker, Docker Compose                                                      |

## Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Browser    │ ──────▶ │  Nginx (frontend │ ──────▶ │  FastAPI         │
│              │  HTTP   │  container :80)  │  /api/* │  (backend :8000) │
└──────────────┘         └──────────────────┘         └──────────────────┘
                            serves React SPA              search engine
                            proxies /api/* to backend     loads movies.json
                                                          builds TF-IDF + BM25 indexes
```

In development, Vite's dev server replaces Nginx and proxies `/api` to the local backend on port 8000.

## Quickstart with Docker

Requires Docker Desktop.

```bash
docker compose up --build
```

Then open <http://localhost:8080>.

The frontend container waits on the backend's healthcheck before serving traffic, so the first request after `docker compose up` always works.

To stop:

```bash
docker compose down
```

## Manual setup (without Docker)

### 1. Backend

```bash
pip install -r requirements.txt
cd src
python -m uvicorn api:app --reload --port 8000
```

The API is now at <http://127.0.0.1:8000>. OpenAPI docs are auto-generated at <http://127.0.0.1:8000/docs>.

> **Windows note:** use `python -m uvicorn` rather than bare `uvicorn` — the pip-installed script is not always on PATH.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the backend on port 8000.

## API reference

### `GET /api/stats`

Returns indexing statistics.

```json
{ "movie_count": 65, "vocabulary_size": 2513 }
```

### `GET /api/search`

| Param    | Type                          | Default | Description                                    |
| -------- | ----------------------------- | ------- | ---------------------------------------------- |
| `q`      | string (required)             | —       | Search query                                   |
| `method` | `bm25` \| `tfidf` \| `hybrid` | `bm25`  | Ranking method                                 |
| `top_k`  | int (1–50)                    | `10`    | How many results to return                     |
| `alpha`  | float (0.0–1.0)               | `0.5`   | Hybrid weight: `1.0` = pure TF-IDF, `0.0` = pure BM25 |

Example:

```
GET /api/search?q=dream%20within%20a%20dream&method=hybrid&top_k=5&alpha=0.6
```

Response:

```json
{
  "query": "dream within a dream",
  "method": "hybrid",
  "terms": ["dream", "within", "dream"],
  "results": [
    {
      "id": 1,
      "title": "Inception",
      "year": 2010,
      "director": "Christopher Nolan",
      "genres": ["Science Fiction", "Thriller", "Action"],
      "cast": "Leonardo DiCaprio, ...",
      "plot": "Dom Cobb is a skilled thief...",
      "score": 1.0
    }
  ]
}
```

## NLP pipeline

For each movie, the searchable document is built by concatenating `title + director + genres + cast + plot`. That document goes through the preprocessor in [src/preprocessing.py](src/preprocessing.py):

1. **Lowercase** the entire string
2. **Tokenize** with the regex `[a-z0-9]+` (drops punctuation in one pass)
3. **Filter out** English stopwords (a hardcoded list of 127 common words) and tokens shorter than 2 characters
4. **Stem** each remaining token with the Porter stemmer (e.g. `astronauts → astronaut`, `dreaming → dream`)

The same preprocessor is used at index time and at query time, so a search for *"dreaming astronauts"* will match a plot mentioning *"astronaut"* and *"dream"*.

### Why a hardcoded stopword list instead of `nltk.corpus.stopwords`?

NLTK's stopwords/punkt data files are downloaded on first use. On some Windows configurations the auto-download fails silently. Embedding the stopwords as a Python frozenset removes that fragility — the only NLTK dependency left is `PorterStemmer`, which works without any data files.

## Ranking details

### TF-IDF (`method=tfidf`)

scikit-learn's `TfidfVectorizer` with `sublinear_tf=True` and L2 normalization, scored against the query vector with cosine similarity. Best for short keyword queries where exact term overlap matters.

### BM25 (`method=bm25`)

`rank_bm25.BM25Okapi` with default parameters (`k1=1.5`, `b=0.75`). Better than TF-IDF for longer, more natural-language queries because it models document-length normalization and saturation of term frequency.

### Hybrid (`method=hybrid`)

Both score vectors are min-max normalized to `[0, 1]` and then linearly combined:

```
combined = α · norm(tfidf) + (1 − α) · norm(bm25)
```

The `α` slider in the UI lets you A/B the two endpoints in real time.

## Project structure

```
.
├── data/
│   └── movies.json            # 65-movie corpus (hand-curated)
├── src/
│   ├── preprocessing.py       # tokenize / stopwords / Porter stemmer
│   ├── indexer.py             # MovieSearchEngine: TF-IDF, BM25, hybrid
│   └── api.py                 # FastAPI endpoints
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # main React component
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js         # dev proxy /api → :8000
│   ├── nginx.conf             # production proxy /api → backend:8000
│   ├── Dockerfile             # multi-stage: Vite build + Nginx serve
│   └── package.json
├── Dockerfile.backend         # Python + FastAPI image
├── docker-compose.yml         # backend + frontend services
├── requirements.txt
└── README.md
```

## Try these queries

| Query                                       | What you should see                                  |
| ------------------------------------------- | ---------------------------------------------------- |
| `dream within a dream`                      | Inception ranks first by a wide margin               |
| `mafia family revenge`                      | The Godfather, Goodfellas, John Wick                 |
| `artificial intelligence consciousness`     | Ex Machina, Her, 2001: A Space Odyssey               |
| `dystopian future society`                  | Blade Runner 2049, Mad Max: Fury Road, WALL-E        |
| `serial killer detective investigation`     | Se7en, Zodiac, The Silence of the Lambs, Prisoners   |
| `war soldier survival mission`              | Saving Private Ryan, 1917, Dunkirk, The Pianist      |

Switch between BM25, TF-IDF, and Hybrid to see how the rankings shift.

## License

This project was built for an academic NLP assignment.
