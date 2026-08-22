export default function StatCards({ forecast, peak, hazardAlert }) {
  const next24 = forecast.slice(0, 24);
  const avg24 =
    next24.reduce((sum, e) => sum + e.predicted_aqi, 0) / (next24.length || 1);

  return (
    <div className="stat-cards">
      <div className="stat-card">
        <p className="stat-card__label">Next 24h average</p>
        <p className="stat-card__value mono">{avg24.toFixed(0)}</p>
      </div>
      <div className="stat-card">
        <p className="stat-card__label">72h peak</p>
        <p className="stat-card__value mono">{peak.predicted_aqi}</p>
      </div>
      <div className="stat-card">
        <p className="stat-card__label">Hazard hours</p>
        <p className="stat-card__value mono">
          {hazardAlert.hours.length}{" "}
          <span className="stat-card__value-of">/ 72</span>
        </p>
      </div>
    </div>
  );
}
