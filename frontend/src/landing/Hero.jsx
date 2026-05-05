import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section
      style={{
        position: "relative",
        zIndex: 1,
        padding: "4rem 6vw 2rem",
        maxWidth: "52rem",
      }}
    >
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        style={{
          fontSize: "0.85rem",
          textTransform: "uppercase",
          letterSpacing: "0.14em",
          color: "var(--text-muted)",
          marginBottom: "1rem",
          fontWeight: 500,
        }}
      >
        Self-supervised audio–visual fusion
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, delay: 0.05 }}
        style={{
          fontSize: "clamp(2.25rem, 4vw, 3.35rem)",
          fontWeight: 600,
          lineHeight: 1.12,
          letterSpacing: "-0.03em",
          margin: "0 0 1rem",
        }}
      >
        Detect mismatches between what you see and what you hear.
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, delay: 0.12 }}
        style={{
          fontSize: "1.08rem",
          lineHeight: 1.65,
          color: "var(--text-muted)",
          maxWidth: "38rem",
          margin: 0,
          fontWeight: 400,
        }}
      >
        A video encoder ingests temporal RGB clips while an audio encoder reads
        log-mel spectrograms. The fusion head scores how well the soundtrack
        belongs to the visuals—the same setup used in your training code in{" "}
        <code style={{ fontSize: "0.9em", color: "var(--text)" }}>src/</code>.
      </motion.p>
    </section>
  );
}
