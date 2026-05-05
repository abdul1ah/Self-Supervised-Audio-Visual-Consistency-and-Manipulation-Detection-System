import { motion } from "framer-motion";

export default function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      style={{
        position: "relative",
        zIndex: 1,
        padding: "2rem 6vw 3rem",
        borderTop: "1px solid var(--border)",
        fontSize: "0.85rem",
        color: "var(--text-muted)",
      }}
    >
      Uploads are processed on your machine; place trained weights at{" "}
      <code style={{ color: "var(--text)" }}>checkpoints/best_model.pth</code>{" "}
      or set <code style={{ color: "var(--text)" }}>CHECKPOINT_PATH</code>.
    </motion.footer>
  );
}
