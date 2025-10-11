import { useRef, useState } from "react";
import { API_BASE } from "./config";

export default function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);

  function onPickFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setError("");
    setResults([]);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }

  async function onSubmit(e) {
    e.preventDefault();
    if (!file) {
      setError("Please select an image first.");
      return;
    }
    try {
      setLoading(true);
      setError("");
      setResults([]);

      const fd = new FormData();
      fd.append("image", file);
      fd.append("top_k", String(topK));

      const resp = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        body: fd,
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${txt.slice(0, 200)}`);
      }
      const data = await resp.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="border-b bg-white">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">PetFace Demo</h1>
          <a
            className="text-sm text-blue-600 hover:underline"
            href="/docs"
            title="Backend API docs (FastAPI)"
          >
            API Docs
          </a>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6">
        {/* Uploader */}
        <form
          onSubmit={onSubmit}
          className="bg-white rounded-2xl shadow-sm border p-4 md:p-6"
        >
          <div className="flex flex-col md:flex-row gap-6">
            {/* Left: Dropzone */}
            <div className="flex-1">
              <label
                htmlFor="file"
                className="block w-full cursor-pointer rounded-xl border-2 border-dashed border-neutral-300 p-6 hover:border-neutral-500 transition"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="text-center space-y-2">
                  <div className="text-4xl">🐶</div>
                  <div className="font-medium">Upload a dog image</div>
                  <div className="text-sm text-neutral-500">
                    JPG/PNG • we’ll search similar identities
                  </div>
                </div>
                <input
                  id="file"
                  type="file"
                  accept="image/*"
                  ref={fileInputRef}
                  onChange={onPickFile}
                  className="hidden"
                />
              </label>

              {preview && (
                <div className="mt-4">
                  <div className="text-sm mb-2 text-neutral-600">
                    Preview
                  </div>
                  <img
                    src={preview}
                    alt="preview"
                    className="w-full max-h-80 object-contain rounded-xl border"
                  />
                </div>
              )}
            </div>

            {/* Right: Controls */}
            <div className="w-full md:w-64">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Top-K</label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value || 5))}
                    className="mt-1 w-full rounded-lg border px-3 py-2"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading || !file}
                  className="w-full rounded-lg bg-blue-600 text-white py-2.5 font-medium hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? "Searching..." : "Search"}
                </button>

                {API_BASE && (
                  <p className="text-xs text-neutral-500">
                    Using API: <span className="break-all">{API_BASE}</span>
                  </p>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
        </form>

        {/* Results */}
        <section className="mt-6">
          {results.length > 0 && (
            <>
              <h2 className="text-lg font-semibold mb-3">
                Results ({results.length})
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {results.map((r, idx) => (
                  <ResultCard key={idx} r={r} />
                ))}
              </div>
            </>
          )}
          {!loading && results.length === 0 && (
            <p className="text-sm text-neutral-500">
              Upload a dog photo to start searching.
            </p>
          )}
        </section>
      </main>

      <footer className="text-center text-xs text-neutral-500 py-8">
        © {new Date().getFullYear()} PetFace Demo
      </footer>
    </div>
  );
}

function ResultCard({ r }) {
  const label =
    Array.isArray(r.label) ? r.label.join(" / ") : String(r.label || "");
  const score = typeof r.score === "number" ? (r.score * 100).toFixed(1) : "";

  return (
    <div className="bg-white border rounded-xl overflow-hidden shadow-sm">
      <div className="aspect-square bg-neutral-100">
        {/* 后端返回了可直链的 url（/images/相对路径） */}
        <img
          src={r.url}
          alt={label}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
      <div className="p-3">
        <div className="text-sm font-medium truncate">{label}</div>
        <div className="text-xs text-neutral-500">Score: {score}%</div>
        {r.path && (
          <div
            className="text-xs text-neutral-400 truncate"
            title={r.path}
          >
            {r.path}
          </div>
        )}
      </div>
    </div>
  );
}
