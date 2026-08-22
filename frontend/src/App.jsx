import { useEffect, useState, useCallback } from "react";
import "./App.css";
import HazardBanner from "./HazardBanner.jsx";
import StatCards from "./StatCards.jsx";
import ForecastChart from "./ForecastChart.jsx";
import ExplanationPanel from "./ExplanationPanel.jsx";
import AqiGauge from "./AqiGauge.jsx";
import PollutantGrid from "./PollutantGrid.jsx";
import WeatherWidget from "./WeatherWidget.jsx";
import ThreeDayCards from "./ThreeDayCards.jsx";
import { categoryFor } from "./aqiCategories.js";

// Adjust if your FastAPI backend runs elsewhere.
const API_BASE_URL = "http://localhost:8000";

function Blobs() {
  return (
    <>
      <div className="bg-blob bg-blob--one" />
      <div className="bg-blob bg-blob--two" />
    </>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [errorMessage, setErrorMessage] = useState("");
  const [chartView, setChartView] = useState("24h"); // "24h" | "72h"

  const loadForecast = useCallback(async () => {
    setStatus("loading");
    setErrorMessage("");
    try {
      const res = await fetch(`${API_BASE_URL}/predict`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const json = await res.json();
      setData(json);
      setStatus("ready");
    } catch (err) {
      setErrorMessage(err.message || "Something went wrong.");
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    loadForecast();
  }, [loadForecast]);

  if (status === "loading") {
    return (
      <div className="state-screen">
        <Blobs />
        <div className="splash-card glass-card">
          <span className="navbar__logo splash-card__logo" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none">
              <path
                d="M4 13c0-4.4 3.6-8 8-8s8 3.6 8 8"
                stroke="var(--accent)"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <path
                d="M7 17c0-2.8 2.2-5 5-5s5 2.2 5 5"
                stroke="var(--accent-2)"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle cx="12" cy="19" r="1.6" fill="var(--accent)" />
            </svg>
          </span>
          <div className="spinner" />
          <p className="state-screen__text">
            Reading the last 72 hours of Islamabad's air…
          </p>
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="state-screen">
        <Blobs />
        <div className="splash-card glass-card">
          <span className="splash-card__error-icon" aria-hidden="true">
            !
          </span>
          <p className="state-screen__text">Couldn't load the forecast.</p>
          <p className="state-screen__detail mono">{errorMessage}</p>
          <button className="retry-button" onClick={loadForecast}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  const {
    city,
    current_aqi,
    current_conditions,
    forecast,
    peak,
    hazard_alert,
    explanation,
  } = data;

  const currentCategory = categoryFor(current_aqi);
  const chartData = chartView === "24h" ? forecast.slice(0, 24) : forecast;

  return (
    <div className="page">
      <Blobs />
      <div className="page__inner">
        <nav className="navbar">
          <div className="navbar__brand">
            <span className="navbar__logo" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none">
                <path
                  d="M4 13c0-4.4 3.6-8 8-8s8 3.6 8 8"
                  stroke="var(--accent)"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <path
                  d="M7 17c0-2.8 2.2-5 5-5s5 2.2 5 5"
                  stroke="var(--accent-2)"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <circle cx="12" cy="19" r="1.6" fill="var(--accent)" />
              </svg>
            </span>
            <div>
              <p className="navbar__title">Skyra</p>
              <p className="navbar__tagline">
                Predicting Air Quality AQI of {city}
              </p>
            </div>
          </div>
          <span className="navbar__pill">Dashboard</span>
        </nav>
        <header className="header glass-card">
          <div>
            <p className="header__eyebrow">
              <span className="header__eyebrow-dot" />
              Live 72-hour forecast · {city}
            </p>
            <h1 className="header__city">{city}</h1>
          </div>
          <button
            className="refresh-button refresh-button--header"
            onClick={loadForecast}
            title="Refresh data"
          >
            ↻
          </button>
        </header>

        <HazardBanner hazardAlert={hazard_alert} peak={peak} />

        {/* ---- Current Air Quality ---- */}
        <div className="current-grid">
          <div className="glass-card current-card">
            <AqiGauge aqi={current_aqi} />
            <div className="current-card__meta">
              <p
                className="current-card__badge"
                style={{
                  color: currentCategory.color,
                  background: currentCategory.bg,
                }}
              >
                {currentCategory.label}
              </p>
              <PollutantGrid conditions={current_conditions} />
            </div>
          </div>
          <div className="glass-card weather-card">
            <WeatherWidget conditions={current_conditions} />
          </div>
        </div>

        {/* ---- 3-day average cards ---- */}
        <p className="section-title">
          <span className="section-title__icon">✦</span>
          3-day average
        </p>
        <ThreeDayCards forecast={forecast} />

        <StatCards forecast={forecast} peak={peak} hazardAlert={hazard_alert} />

        {/* ---- Trend chart ---- */}
        <div className="section-title-row">
          <p className="section-title">
            <span className="section-title__icon">✦</span>
            {chartView === "24h" ? "24-hour" : "72-hour"} trend
          </p>
          <div className="view-toggle">
            <button
              className={`view-toggle__btn ${chartView === "24h" ? "is-active" : ""}`}
              onClick={() => setChartView("24h")}
            >
              24h
            </button>
            <button
              className={`view-toggle__btn ${chartView === "72h" ? "is-active" : ""}`}
              onClick={() => setChartView("72h")}
            >
              72h
            </button>
          </div>
        </div>
        <div className="glass-card chart-card">
          <ForecastChart forecast={chartData} />
        </div>

        <div className="glass-card">
          <ExplanationPanel explanation={explanation} />
        </div>

        <footer className="footer">
          <span>Ridge regression · Hopsworks feature store</span>
        </footer>
      </div>
    </div>
  );
}
