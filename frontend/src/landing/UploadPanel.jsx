import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

function pickVideos(e) {
  return e.target.files;
}

export default function UploadPanel() {
  const youtubeInput = useRef(null);
  const datasetInput = useRef(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [lastKind, setLastKind] = useState(null);

  async function uploadWithKind(file, kind) {
    if (!file) return;
    setError(null);
    setResult(null);
    setLastKind(kind);
    setLoading(true);
    const body = new FormData();
    body.append("file", file);
    body.append("video_kind", kind);
    try {
      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = data.detail;
        let msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d.map((e) => e.msg ?? JSON.stringify(e)).join(" ")
              : res.statusText;
        throw new Error(msg || "Request failed");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  function onYoutubeChange(e) {
    const files = pickVideos(e);
    if (files?.length) uploadWithKind(files[0], "youtube");
    e.target.value = "";
  }

  function onDatasetChange(e) {
    const files = pickVideos(e);
    if (files?.length) uploadWithKind(files[0], "dataset");
    e.target.value = "";
  }

  return (
    <section style={{ position: "relative", zIndex: 1, padding: "0 6vw 4rem" }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
        style={{
          display: "grid",
          gap: "1.25rem",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          maxWidth: "960px",
        }}
      >
        <input
          ref={youtubeInput}
          type="file"
          accept="video/*"
          hidden
          onChange={onYoutubeChange}
        />
        <input
          ref={datasetInput}
          type="file"
          accept="video/*"
          hidden
          onChange={onDatasetChange}
        />

        <article
          style={{
            background: "var(--bg-elevated)",
            borderRadius: "16px",
            padding: "1.75rem",
            border: "1px solid var(--border)",
            boxShadow: "var(--shadow)",
          }}
        >
          <h2
            style={{
              margin: "0 0 0.5rem",
              fontSize: "1.2rem",
              fontWeight: 600,
              letterSpacing: "-0.02em",
            }}
          >
            YouTube-style clip
          </h2>
          <p
            style={{
              margin: "0 0 1.25rem",
              color: "var(--text-muted)",
              fontSize: "0.95rem",
              lineHeight: 1.55,
            }}
          >
            Natural backgrounds (faces in context). We attempt a face-centric crop
            before resizing so the 3D ResNet can focus on facial motion like in
            the wild.
          </p>
          <motion.button
            type="button"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            disabled={loading}
            onClick={() => youtubeInput.current?.click()}
            style={{
              width: "100%",
              padding: "0.85rem 1rem",
              borderRadius: "10px",
              border: "none",
              background: "var(--accent)",
              color: "#faf9f6",
              fontWeight: 600,
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading && lastKind === "youtube" ? "Analyzing…" : "Upload video"}
          </motion.button>
        </article>

        <article
          style={{
            background: "var(--bg-elevated)",
            borderRadius: "16px",
            padding: "1.75rem",
            border: "1px solid var(--border)",
            boxShadow: "var(--shadow)",
          }}
        >
          <h2
            style={{
              margin: "0 0 0.5rem",
              fontSize: "1.2rem",
              fontWeight: 600,
              letterSpacing: "-0.02em",
            }}
          >
            Dataset / studio clip
          </h2>
          <p
            style={{
              margin: "0 0 1.25rem",
              color: "var(--text-muted)",
              fontSize: "0.95rem",
              lineHeight: 1.55,
            }}
          >
            Clean or white-background footage (RAVDESS-style). Full-frame
            resize matches the preprocessing path in{" "}
            <code style={{ fontSize: "0.9em" }}>src/preprocess.py</code>.
          </p>
          <motion.button
            type="button"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            disabled={loading}
            onClick={() => datasetInput.current?.click()}
            style={{
              width: "100%",
              padding: "0.85rem 1rem",
              borderRadius: "10px",
              border: "1px solid var(--border)",
              background: "#fffefb",
              color: "var(--accent)",
              fontWeight: 600,
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading && lastKind === "dataset" ? "Analyzing…" : "Upload video"}
          </motion.button>
        </article>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{
              marginTop: "1.5rem",
              padding: "1rem 1.25rem",
              maxWidth: "960px",
              borderRadius: "12px",
              background: "#fff5f5",
              border: "1px solid rgba(180, 40, 40, 0.2)",
              color: "#5c1515",
              fontSize: "0.95rem",
            }}
          >
            {error}
          </motion.div>
        )}
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{
              marginTop: "1.5rem",
              padding: "1.35rem 1.5rem",
              maxWidth: "960px",
              borderRadius: "14px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              boxShadow: "var(--shadow)",
            }}
          >
            <p
              style={{
                margin: "0 0 0.35rem",
                fontSize: "0.75rem",
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                color: "var(--text-muted)",
              }}
            >
              Model output
            </p>
            <p
              style={{
                margin: "0 0 0.75rem",
                fontSize: "2rem",
                fontWeight: 600,
                letterSpacing: "-0.03em",
              }}
            >
              {(result.match_probability * 100).toFixed(1)}%
              <span
                style={{
                  fontSize: "0.95rem",
                  fontWeight: 500,
                  color: "var(--text-muted)",
                  marginLeft: "0.5rem",
                }}
              >
                match probability
              </span>
            </p>
            <p style={{ margin: 0, lineHeight: 1.6, color: "var(--text-muted)" }}>
              {result.interpretation}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
