export default function HazardBanner({ hazardAlert, peak }) {
  if (!hazardAlert || !hazardAlert.has_alert) return null;

  const peakTime = new Date(peak.timestamp).toLocaleString(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <div className="hazard-banner" role="alert">
      <span className="hazard-banner__icon" aria-hidden="true">
        ▲
      </span>
      <div>
        <p className="hazard-banner__title">
          Unhealthy air expected — {hazardAlert.hours.length}{" "}
          {hazardAlert.hours.length === 1 ? "hour" : "hours"} flagged
        </p>
        <p className="hazard-banner__detail">
          Peak of <span className="mono">{peak.predicted_aqi}</span> AQI around{" "}
          {peakTime}
        </p>
      </div>
    </div>
  );
}
