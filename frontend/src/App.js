import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import "./App.css";
import Header from "./components/Header";
import RealTime from "./components/RealTime";

function Placeholder({ label }) {
  return (
    <div style={{ marginTop: 80, fontSize: 24, color: '#888', fontWeight: 500 }}>
      {label} (Coming Soon)
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="app-bg">
        <Header />
        <Routes>
          <Route path="/" element={<RealTime />} />
          <Route path="/image-upload" element={<Placeholder label="Image Upload" />} />
          <Route path="/video-upload" element={<Placeholder label="Video Upload" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
