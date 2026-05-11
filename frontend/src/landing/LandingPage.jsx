import { Suspense } from "react";
import SceneBackground from "./SceneBackground.jsx";
import Navbar from "./Navbar.jsx";
import ChatAnalyzer from "./ChatAnalyzer.jsx";
import Footer from "./Footer.jsx";

function SceneFallback() {
  return null;
}

export default function LandingPage() {
  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <Suspense fallback={<SceneFallback />}>
        <SceneBackground />
      </Suspense>
      <Navbar />
      <ChatAnalyzer />
      <Footer />
    </div>
  );
}
