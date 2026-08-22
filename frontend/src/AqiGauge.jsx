import { categoryFor } from "./aqiCategories.js";

export default function AqiGauge({ aqi }) {
  const category = categoryFor(aqi);
  const size = 168;
  const stroke = 12;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.min(aqi / 300, 1);
  const offset = circumference * (1 - pct);

  return (
    <div className="gauge">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--surface-sunken)"
          strokeWidth={stroke}
        />
        <circle
          className="gauge__progress"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={category.color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge__center">
        <p className="gauge__value mono">{Math.round(aqi)}</p>
        <p className="gauge__label">AQI</p>
      </div>
    </div>
  );
}
