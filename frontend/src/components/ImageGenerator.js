import React, { useState } from "react";
import axios from "axios";

const ImageGenerator = () => {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      alert("Please enter a prompt!");
      return;
    }

    setLoading(true);
    setImage(null);

    try {
      const formData = new FormData();
      formData.append("prompt", prompt);

      const response = await axios.post("http://localhost:8000/generate", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        responseType: "blob",
      });

      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
    } catch (error) {
      console.error("Error generating image:", error);
      alert("Failed to generate image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style jsx>{`
        .cyber-input {
          background: rgba(0, 0, 0, 0.4);
          border: 2px solid rgba(64, 224, 255, 0.3);
          border-radius: 15px;
          color: #ffffff;
          transition: all 0.3s ease;
        }
        
        .cyber-input:focus {
          background: rgba(0, 0, 0, 0.6);
          border-color: #40e0ff;
          box-shadow: 0 0 20px rgba(64, 224, 255, 0.4);
          color: #ffffff;
        }
        
        .cyber-input::placeholder {
          color: rgba(255, 255, 255, 0.5);
        }
        
        .cyber-btn {
          background: linear-gradient(45deg, #7c3aed, #40e0ff);
          border: none;
          border-radius: 25px;
          color: white;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 1px;
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
        }
        
        .cyber-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 10px 30px rgba(124, 58, 237, 0.4);
          color: white;
        }
        
        .cyber-btn:disabled {
          background: linear-gradient(45deg, #4a5568, #718096);
          cursor: not-allowed;
          transform: none;
        }
        
        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top: 2px solid #ffffff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-right: 10px;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .image-container {
          background: rgba(0, 0, 0, 0.3);
          border: 2px solid rgba(64, 224, 255, 0.2);
          border-radius: 20px;
          padding: 20px;
          backdrop-filter: blur(10px);
        }
        
        .generated-image {
          border-radius: 15px;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
          transition: transform 0.3s ease;
        }
        
        .generated-image:hover {
          transform: scale(1.02);
        }
        
        .pulse-animation {
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
        
        .tech-label {
          background: linear-gradient(45deg, #7c3aed, #40e0ff);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 1px;
          font-size: 0.9rem;
        }
      `}</style>
      
      <div className="row">
        <div className="col-12">
          <div className="mb-4">
            <label className="form-label tech-label mb-3 d-block">
              Image Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your vision... (e.g., 'A futuristic city at sunset with flying cars')"
              rows="4"
              className="form-control cyber-input p-3"
              style={{ fontSize: "1.1rem", lineHeight: "1.6" }}
            />
          </div>

          <div className="text-center mb-4">
            <button 
              onClick={handleGenerate} 
              disabled={loading} 
              className="btn btn-lg cyber-btn px-5 py-3"
            >
              {loading && <div className="loading-spinner d-inline-block"></div>}
              {loading ? "Generating..." : "Generate Image"}
            </button>
          </div>

          {loading && (
            <div className="text-center mb-4">
              <div className="pulse-animation">
                <h5 className="text-light mb-3">AI is creating your masterpiece...</h5>
                <div className="progress" style={{height: "4px", background: "rgba(255,255,255,0.1)"}}>
                  <div 
                    className="progress-bar" 
                    style={{
                      background: "linear-gradient(45deg, #7c3aed, #40e0ff)",
                      animation: "progress 2s ease-in-out infinite"
                    }}
                  ></div>
                </div>
              </div>
              
              <style jsx>{`
                @keyframes progress {
                  0% { width: 0%; }
                  50% { width: 60%; }
                  100% { width: 100%; }
                }
              `}</style>
            </div>
          )}

          {image && (
            <div className="image-container">
              <div className="text-center mb-3">
                <h4 className="tech-label mb-3">Generated Masterpiece</h4>
              </div>
              <div className="text-center">
                <img 
                  src={image} 
                  alt="Generated" 
                  className="img-fluid generated-image"
                  style={{maxHeight: "500px"}}
                />
              </div>
              <div className="text-center mt-3">
                <small className="text-light opacity-75">
                  Tip: Right-click to save your creation
                </small>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ImageGenerator;