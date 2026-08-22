function IconThermometer() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z" />
    </svg>
  );
}
function IconFeelsLike() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}
function IconDroplet() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2.7s6 6.2 6 10.5a6 6 0 0 1-12 0c0-4.3 6-10.5 6-10.5Z" />
    </svg>
  );
}
function IconGauge() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 20a8 8 0 1 0-8-8" />
      <path d="M12 12 16 8" />
      <path d="M12 20v0" />
    </svg>
  );
}
function IconWind() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 8h9a2.5 2.5 0 1 0-2.5-2.5" />
      <path d="M3 16h12a2.5 2.5 0 1 1-2.5 2.5" />
      <path d="M3 12h15a2 2 0 1 0-2-2" />
    </svg>
  );
}
function IconCloud() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M17.5 19a4.5 4.5 0 0 0 0-9 6 6 0 0 0-11.4 2A4 4 0 0 0 6.5 19h11Z" />
    </svg>
  );
}

const WEATHER_ITEMS = [
  {
    key: "temperature",
    label: "Temperature",
    unit: "°C",
    Icon: IconThermometer,
  },
  { key: "feels_like", label: "Feels like", unit: "°C", Icon: IconFeelsLike },
  { key: "humidity", label: "Humidity", unit: "%", Icon: IconDroplet },
  { key: "pressure", label: "Pressure", unit: "hPa", Icon: IconGauge },
  { key: "wind_speed", label: "Wind", unit: "km/h", Icon: IconWind },
  { key: "clouds", label: "Cloud cover", unit: "%", Icon: IconCloud },
];

export default function WeatherWidget({ conditions }) {
  if (!conditions) return null;

  return (
    <div className="weather-widget">
      <p className="weather-widget__title">Atmospheric conditions</p>
      <div className="weather-widget__grid">
        {WEATHER_ITEMS.map((item) => (
          <div className="weather-item" key={item.key}>
            <span className="weather-item__icon" aria-hidden="true">
              <item.Icon />
            </span>
            <div>
              <p className="weather-item__value mono">
                {conditions[item.key]?.toFixed(1) ?? "—"}
                <span className="weather-item__unit">{item.unit}</span>
              </p>
              <p className="weather-item__label">{item.label}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
