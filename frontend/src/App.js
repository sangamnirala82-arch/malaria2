import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MalariaDetection from "./components/MalariaDetection";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MalariaDetection />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
