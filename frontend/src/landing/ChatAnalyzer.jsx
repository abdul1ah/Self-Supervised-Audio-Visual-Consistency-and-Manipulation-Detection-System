import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function ChatAnalyzer() {
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const [messages, setMessages] = useState([
    {
      id: "intro",
      role: "assistant",
      content: "Hey! I'm your audio-visual consistency analyzer. Upload a video to get started, and I'll check whether the audio and visuals are perfectly synced.",
      timestamp: new Date(),
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [videoKind, setVideoKind] = useState("youtube");
  const [agentMode, setAgentMode] = useState("tool_calling");

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const newUserMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: `Analyzing: ${file.name}`,
      file_name: file.name,
      file_size_mb: (file.size / (1024 * 1024)).toFixed(2),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("video_kind", videoKind);

    try {
      const response = await fetch(`/api/analyze?agent_mode=${agentMode}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        const detail =
          error.detail ||
          error.message ||
          `Error: ${response.status} ${response.statusText}`;
        throw new Error(detail);
      }

      const data = await response.json();

      if (agentMode === "tool_calling") {
        // Tool-calling mode: stream the conversation
        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: data.final_response,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        // Simple mode: structured response
        const responseText =
          data.agent_summary ||
          `Match probability: ${(data.match_probability * 100).toFixed(1)}%\n\n${data.interpretation}`;

        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: responseText,
          match_probability: data.match_probability,
          risk_level: data.risk_level,
          recommendation: data.recommendation,
          validation_issues: data.validation?.issues || [],
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (err) {
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `⚠️ ${err.message || "Upload failed"}`,
        is_error: true,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      fileInputRef.current.value = "";
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <section
      style={{
        position: "relative",
        zIndex: 1,
        padding: "1rem 1.5rem",
        maxWidth: "900px",
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        height: "100%",
      }}
    >
      {/* Chat Messages Container */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          marginBottom: "2rem",
          paddingRight: "0.5rem",
          scrollBehavior: "smooth",
        }}
      >
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                marginBottom: "1.5rem",
                display: "flex",
                justifyContent:
                  message.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div
                style={{
                  maxWidth: "75%",
                  padding: "1rem 1.25rem",
                  borderRadius: "16px",
                  background:
                    message.role === "user"
                      ? "var(--accent)"
                      : message.is_error
                        ? "#fff5f5"
                        : "var(--bg-elevated)",
                  border:
                    message.role === "user"
                      ? "none"
                      : message.is_error
                        ? "1px solid rgba(180, 40, 40, 0.2)"
                        : "1px solid var(--border)",
                  color:
                    message.role === "user"
                      ? "#faf9f6"
                      : message.is_error
                        ? "#5c1515"
                        : "var(--text-default)",
                  lineHeight: 1.6,
                  wordWrap: "break-word",
                }}
              >
                <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                  {message.content}
                </p>

                {/* File info for user messages */}
                {message.role === "user" && message.file_name && (
                  <p
                    style={{
                      margin: "0.75rem 0 0",
                      fontSize: "0.85rem",
                      opacity: 0.7,
                    }}
                  >
                    {message.file_size_mb} MB
                  </p>
                )}

                {/* Structured metadata for assistant messages */}
                {message.role === "assistant" &&
                  !message.is_error &&
                  message.match_probability !== undefined && (
                    <div
                      style={{
                        marginTop: "1rem",
                        paddingTop: "1rem",
                        borderTop:
                          message.role === "user"
                            ? "none"
                            : "1px solid rgba(0,0,0,0.1)",
                      }}
                    >
                      <div
                        style={{
                          display: "grid",
                          gap: "0.75rem",
                          gridTemplateColumns: "repeat(2, 1fr)",
                        }}
                      >
                        <div>
                          <p
                            style={{
                              margin: "0 0 0.35rem",
                              fontSize: "0.7rem",
                              textTransform: "uppercase",
                              letterSpacing: "0.1em",
                              opacity: 0.6,
                            }}
                          >
                            Probability
                          </p>
                          <p
                            style={{
                              margin: 0,
                              fontSize: "1.4rem",
                              fontWeight: 600,
                            }}
                          >
                            {(message.match_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p
                            style={{
                              margin: "0 0 0.35rem",
                              fontSize: "0.7rem",
                              textTransform: "uppercase",
                              letterSpacing: "0.1em",
                              opacity: 0.6,
                            }}
                          >
                            Risk level
                          </p>
                          <p
                            style={{
                              margin: 0,
                              fontSize: "0.95rem",
                              fontWeight: 600,
                              textTransform: "capitalize",
                            }}
                          >
                            {message.risk_level || "unknown"}
                          </p>
                        </div>
                      </div>

                      {message.recommendation && (
                        <p
                          style={{
                            marginTop: "0.75rem",
                            fontSize: "0.9rem",
                            opacity: 0.8,
                            fontStyle: "italic",
                          }}
                        >
                          💡 {message.recommendation}
                        </p>
                      )}

                      {message.validation_issues?.length > 0 && (
                        <div
                          style={{
                            marginTop: "0.75rem",
                            padding: "0.6rem",
                            borderRadius: "8px",
                            background: "rgba(180, 120, 30, 0.08)",
                            fontSize: "0.85rem",
                          }}
                        >
                          <p style={{ margin: "0 0 0.35rem", fontWeight: 600 }}>
                            ⚠️ Notes:
                          </p>
                          <ul
                            style={{
                              margin: 0,
                              paddingLeft: "1.2rem",
                            }}
                          >
                            {message.validation_issues.map((issue, idx) => (
                              <li key={idx}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              display: "flex",
              justifyContent: "flex-start",
              marginBottom: "1.5rem",
            }}
          >
            <div
              style={{
                padding: "1rem 1.25rem",
                borderRadius: "16px",
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                display: "flex",
                gap: "0.5rem",
                alignItems: "center",
              }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                style={{
                  width: "16px",
                  height: "16px",
                  borderRadius: "50%",
                  border: "2px solid var(--accent)",
                  borderTopColor: "transparent",
                }}
              />
              <span style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
                Analyzing...
              </span>
            </div>
          </motion.div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Section */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          paddingTop: "1rem",
          borderTop: "1px solid var(--border)",
        }}
      >
        {/* Settings */}
        <div
          style={{
            display: "grid",
            gap: "0.75rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          }}
        >
          <div>
            <label
              style={{
                display: "block",
                fontSize: "0.8rem",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: "0.35rem",
                color: "#000",
                fontWeight: 700,
              }}
            >
              Video Type
            </label>
            <select
              value={videoKind}
              onChange={(e) => setVideoKind(e.target.value)}
              disabled={loading}
              style={{
                width: "100%",
                padding: "0.6rem",
                borderRadius: "8px",
                border: "1px solid var(--border)",
                background: "#FFF",
                color: "#000",
                fontSize: "0.9rem",
                cursor: loading ? "wait" : "pointer",
              }}
            >
              <option value="youtube">YouTube-style (natural background)</option>
              <option value="dataset">Dataset (clean background)</option>
            </select>
          </div>

          <div>
            <label
              style={{
                display: "block",
                fontSize: "0.8rem",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: "0.35rem",
                color: "#000",
                fontWeight: 700,
              }}
            >
              Agent Mode
            </label>
            <select
              value={agentMode}
              onChange={(e) => setAgentMode(e.target.value)}
              disabled={loading}
              style={{
                width: "100%",
                padding: "0.6rem",
                borderRadius: "8px",
                border: "1px solid var(--border)",
                background: "#FFF",
                color: "#000",
                fontSize: "0.9rem",
                cursor: loading ? "wait" : "pointer",
              }}
            >
              <option value="simple">Simple (rule-based)</option>
              <option value="tool_calling">
                Tool Calling (LLM-powered)
              </option>
            </select>
          </div>
        </div>

        {/* Upload Button */}
        <motion.button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          style={{
            padding: "0.95rem 1.5rem",
            borderRadius: "12px",
            border: "none",
            background: "var(--accent)",
            color: "#faf9f6",
            fontWeight: 600,
            fontSize: "1rem",
            cursor: loading ? "wait" : "pointer",
            opacity: loading ? 0.7 : 1,
            transition: "opacity 0.2s",
          }}
        >
          {loading ? "Analyzing..." : "📹 Upload Video"}
        </motion.button>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          disabled={loading}
          hidden
        />

        <p
          style={{
            margin: 0,
            fontSize: "0.8rem",
            color: "var(--text-muted)",
            textAlign: "center",
          }}
        >
          Max 200 MB. Supports MP4, WebM, MOV, MKV, AVI
        </p>
      </div>
    </section>
  );
}
