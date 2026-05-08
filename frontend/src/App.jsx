import { useEffect, useMemo, useRef, useState } from "react";

const METHODS = [
  { value: "bm25", label: "BM25" },
  { value: "tfidf", label: "TF-IDF (cosine)" },
  { value: "hybrid", label: "Hybrid" },
];

const EXAMPLES = [
  "dystopian future society",
  "dream within a dream",
  "mafia family revenge",
  "artificial intelligence consciousness",
  "lonely man falls in love",
  "war soldier survival mission",
  "serial killer detective investigation",
];

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function Highlight({ text, terms }) {
  const html = useMemo(() => {
    if (!terms || terms.length === 0) return text;
    const pattern = new RegExp(
      `\\b(${terms.filter((t) => t.length > 1).map(escapeRegex).join("|")})\\b`,
      "gi"
    );
    const parts = text.split(pattern);
    return parts.map((part, i) =>
      pattern.test(part) ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>
    );
  }, [text, terms]);
  return <>{html}</>;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("bm25");
  const [topK, setTopK] = useState(10);
  const [alpha, setAlpha] = useState(0.5);

  const [results, setResults] = useState([]);
  const [terms, setTerms] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchedQuery, setSearchedQuery] = useState("");

  const debounceRef = useRef(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setError("Could not reach the search API. Is the FastAPI server running on port 8000?"));
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) {
      setResults([]);
      setSearchedQuery("");
      setError(null);
      return;
    }
    debounceRef.current = setTimeout(() => {
      runSearch();
    }, 250);
    return () => clearTimeout(debounceRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, method, topK, alpha]);

  async function runSearch() {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        q: query.trim(),
        method,
        top_k: String(topK),
        alpha: String(alpha),
      });
      const res = await fetch(`/api/search?${params}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResults(data.results);
      setTerms(data.terms);
      setSearchedQuery(data.query);
    } catch (e) {
      setError(e.message || "Search failed.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Movie Search Engine</h1>
        <p className="subtitle">
          {stats
            ? `Searching across ${stats.movie_count} movies — vocabulary of ${stats.vocabulary_size} stemmed terms.`
            : "Loading index…"}
        </p>
      </header>

      <div className="controls">
        <input
          className="search-input"
          type="search"
          placeholder="Type a query, e.g. 'space astronaut wormhole'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        <select
          className="select"
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          aria-label="Ranking method"
        >
          {METHODS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <input
          className="number"
          type="number"
          min={1}
          max={20}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value) || 1)}
          aria-label="Number of results"
          style={{ width: 80 }}
        />
      </div>

      {method === "hybrid" && (
        <div className="advanced">
          <label htmlFor="alpha">
            Hybrid weight α (TF-IDF ↔ BM25): <strong>{alpha.toFixed(2)}</strong>
          </label>
          <input
            id="alpha"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
          <span>α=1 → pure TF-IDF, α=0 → pure BM25</span>
        </div>
      )}

      <div className="examples">
        <span className="label">Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => setQuery(ex)}>
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      {loading && <div className="status">Searching…</div>}

      {!loading && !error && searchedQuery && (
        <div className="status">
          {results.length > 0
            ? `${results.length} result${results.length === 1 ? "" : "s"} for "${searchedQuery}"`
            : `No results for "${searchedQuery}"`}
        </div>
      )}

      <div className="results">
        {results.map((r, idx) => (
          <article key={r.id} className="result">
            <div className="result-head">
              <h2 className="result-title">
                <span className="result-rank">{idx + 1}.</span>
                {r.title} <span style={{ color: "var(--text-dim)", fontWeight: 400 }}>({r.year})</span>
              </h2>
              <span className="result-score">score {r.score.toFixed(4)}</span>
            </div>
            <div className="result-meta">
              <strong>{r.director}</strong> · {r.genres.join(", ")} · <em>{r.cast}</em>
            </div>
            <p className="result-plot">
              <Highlight text={r.plot} terms={terms} />
            </p>
          </article>
        ))}
      </div>

      {!query && !error && (
        <div className="empty">
          Start typing or pick an example query above.
        </div>
      )}
    </div>
  );
}
