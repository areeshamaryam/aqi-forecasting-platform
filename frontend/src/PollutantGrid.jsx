const POLLUTANTS = [
  { key: "pm2_5", label: "PM2.5", unit: "µg/m³" },
  { key: "pm10", label: "PM10", unit: "µg/m³" },
  { key: "o3", label: "O₃", unit: "µg/m³" },
  { key: "no2", label: "NO₂", unit: "µg/m³" },
  { key: "so2", label: "SO₂", unit: "µg/m³" },
  { key: "co", label: "CO", unit: "µg/m³" },
];

export default function PollutantGrid({ conditions }) {
  if (!conditions) return null;

  return (
    <div className="pollutant-grid">
      {POLLUTANTS.map((p) => (
        <div className="pollutant-tile" key={p.key}>
          <p className="pollutant-tile__label">{p.label}</p>
          <p className="pollutant-tile__value mono">
            {conditions[p.key]?.toFixed(1) ?? "—"}
          </p>
          <p className="pollutant-tile__unit">{p.unit}</p>
        </div>
      ))}
    </div>
  );
}
