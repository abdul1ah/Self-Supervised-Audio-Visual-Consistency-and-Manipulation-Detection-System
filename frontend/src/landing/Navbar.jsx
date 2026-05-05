import { motion } from "framer-motion";

const fadeUp = {
  initial: { opacity: 0, y: 22 },
  animate: { opacity: 1, y: 0 },
};

export default function Navbar() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        position: "relative",
        zIndex: 2,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "1.25rem 6vw",
        borderBottom: "1px solid var(--border)",
        background: "rgba(242, 241, 236, 0.72)",
        backdropFilter: "blur(12px)",
      }}
    >
      <motion.span
        variants={fadeUp}
        initial="initial"
        animate="animate"
        transition={{ delay: 0.05 }}
        style={{
          fontWeight: 600,
          letterSpacing: "-0.03em",
          fontSize: "1.05rem",
        }}
      >
        AV Consistency Lab
      </motion.span>
      <span
        style={{
          fontSize: "0.8rem",
          color: "var(--text-muted)",
          fontWeight: 500,
        }}
      >
        Dual-stream · R3D-18 + ResNet18
      </span>
    </motion.header>
  );
}
