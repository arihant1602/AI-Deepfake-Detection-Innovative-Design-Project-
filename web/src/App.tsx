import { Route, Routes } from "react-router-dom";
import { Header } from "./components/Header";
import { Live } from "./pages/Live";
import { Analyze } from "./pages/Analyze";
import { Device } from "./pages/Device";

export function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Live />} />
        <Route path="/file" element={<Analyze />} />
        <Route path="/device" element={<Device />} />
        <Route path="*" element={<Live />} />
      </Routes>
    </>
  );
}
