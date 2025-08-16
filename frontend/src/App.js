import React from "react";
import ImageGenerator from "./components/ImageGenerator";

function App() {
  return (
    <>
      {/* Bootstrap CSS CDN */}
      <link
        href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css"
        rel="stylesheet"
      />
      
      {/* Custom Dark Theme Styles */}
      <style jsx>{`
        body {
          background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
          min-height: 100vh;
          color: #ffffff;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .tech-container {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 20px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .glow-effect {
          text-shadow: 0 0 20px rgba(64, 224, 255, 0.5);
          background: linear-gradient(45deg, #40e0ff, #7c3aed);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        
        .particle-bg::before {
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-image: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
          z-index: -1;
        }
      `}</style>
      
      <div className="particle-bg min-vh-100">
        <div className="container py-5">
          <div className="row justify-content-center">
            <div className="col-12 text-center mb-5">
              <h1 className="display-3 fw-bold mb-3 glow-effect">
                AI Image Generator
              </h1>
              <p className="lead text-light opacity-75">
                Transform your ideas into stunning visuals with advanced AI technology
              </p>
            </div>
            <div className="col-lg-8">
              <div className="tech-container p-4">
                <ImageGenerator />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/js/bootstrap.bundle.min.js"></script>
    </>
  );
}

export default App;