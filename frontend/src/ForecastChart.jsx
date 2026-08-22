import { useEffect, useRef } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Filler,
} from "chart.js";
import { AQI_CATEGORIES } from "./aqiCategories.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Filler,
);

// Reads a CSS custom property's resolved value, so the chart's
// canvas colors (which Chart.js can't read var() directly for)
// stay in sync with the token system in index.css.
function resolveVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

// Custom Chart.js plugin: paints horizontal AQI-category bands
// behind the line, turning the chart itself into a readout of
// severity zones rather than a plain line on a blank grid.
const bandPlugin = {
  id: "aqiBands",
  beforeDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;

    const yScale = scales.y;
    const bands = [
      { from: 0, to: 50, color: resolveVar("--aqi-good-bg") },
      { from: 50, to: 100, color: resolveVar("--aqi-moderate-bg") },
      { from: 100, to: 150, color: resolveVar("--aqi-usg-bg") },
      { from: 150, to: 200, color: resolveVar("--aqi-unhealthy-bg") },
      { from: 200, to: 300, color: resolveVar("--aqi-very-unhealthy-bg") },
      { from: 300, to: yScale.max, color: resolveVar("--aqi-hazardous-bg") },
    ];

    ctx.save();
    bands.forEach((band) => {
      const yTop = yScale.getPixelForValue(Math.min(band.to, yScale.max));
      const yBottom = yScale.getPixelForValue(Math.max(band.from, yScale.min));
      if (yBottom <= yTop) return;
      ctx.fillStyle = band.color;
      ctx.fillRect(
        chartArea.left,
        yTop,
        chartArea.right - chartArea.left,
        yBottom - yTop,
      );
    });
    ctx.restore();
  },
};

export default function ForecastChart({ forecast }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || forecast.length === 0) return;

    const labels = forecast.map((entry) => {
      const d = new Date(entry.timestamp);
      return d.toLocaleString(undefined, { weekday: "short", hour: "numeric" });
    });
    const values = forecast.map((entry) => entry.predicted_aqi);
    const maxValue = Math.max(...values, 160);

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            data: values,
            borderColor: resolveVar("--ink"),
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: resolveVar("--ink"),
            tension: 0.35,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: resolveVar("--ink"),
            titleFont: { family: "IBM Plex Mono", size: 11 },
            bodyFont: { family: "IBM Plex Mono", size: 12 },
            padding: 10,
            cornerRadius: 6,
            callbacks: {
              label: (ctx) => {
                const entry = forecast[ctx.dataIndex];
                return `${entry.predicted_aqi} AQI · ${entry.category}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: resolveVar("--ink-secondary"),
              font: { family: "IBM Plex Mono", size: 10 },
              maxTicksLimit: 9,
              autoSkip: true,
            },
          },
          y: {
            min: 0,
            max: maxValue,
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              color: resolveVar("--ink-secondary"),
              font: { family: "IBM Plex Mono", size: 10 },
            },
          },
        },
      },
      plugins: [bandPlugin],
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [forecast]);

  return (
    <div style={{ position: "relative", width: "100%", height: 280 }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Line chart of predicted AQI for the next 72 hours, background bands show EPA severity zones`}
      />
    </div>
  );
}
