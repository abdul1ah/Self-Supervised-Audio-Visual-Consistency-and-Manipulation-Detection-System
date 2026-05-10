import { Suspense } from "react";
import SceneBackground from "./SceneBackground.jsx";
import Navbar from "./Navbar.jsx";
import Hero from "./Hero.jsx";
import ChatAnalyzer from "./ChatAnalyzer.jsx";
import Footer from "./Footer.jsx";

function SceneFallback() {
  return null;
}

export default function LandingPage() {
  return (
    <>
      <Suspense fallback={<SceneFallback />}>
        <SceneBackground />
      </Suspense>
      <Navbar />
      <Hero />
      <ChatAnalyzer />
      <Footer />
    </>
  );
}
