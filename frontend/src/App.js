import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';

import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Recommend from './pages/Recommend';

function App() {
  return (
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/sign-up" element={<Signup />} />
            <Route path="/recommend" element={<Recommend />} />
          </Routes>
        </div>
      </Router>
  );
}

export default App;
