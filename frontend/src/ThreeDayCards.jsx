import { categoryFor } from "./aqiCategories.js";

function bucketByDay(forecast) {
  const buckets = new Map();

  forecast.forEach((entry) => {
    const d = new Date(entry.timestamp);
    const dayKey = d.toDateString();
    if (!buckets.has(dayKey)) buckets.set(dayKey, []);
    buckets.get(dayKey).push(entry.predicted_aqi);
  });

  const days = Array.from(buckets.entries()).slice(0, 3);
  const labels = ["Today", "Tomorrow", "Day after"];

  return days.map(([dayKey, values], idx) => ({
    label: labels[idx] ?? dayKey,
    date: new Date(dayKey).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
    avgAqi: values.reduce((a, b) => a + b, 0) / values.length,
  }));
}

export default function ThreeDayCards({ forecast }) {
  const days = bucketByDay(forecast);

  return (
    <div className="day-cards">
      {days.map((day) => {
        const category = categoryFor(day.avgAqi);
        return (
          <div className="day-card" key={day.label}>
            <div className="day-card__top">
              <div>
                <p className="day-card__label">{day.label}</p>
                <p className="day-card__date">{day.date}</p>
              </div>
              <span
                className="day-card__badge"
                style={{
                  color: category.color,
                  background: category.bg,
                }}
              >
                {category.label}
              </span>
            </div>
            <p className="day-card__value mono">{Math.round(day.avgAqi)}</p>
            <p className="day-card__caption">Average predicted AQI</p>
            <div className="day-card__bar-track">
              <div
                className="day-card__bar-fill"
                style={{
                  width: `${Math.min((day.avgAqi / 300) * 100, 100)}%`,
                  background: category.color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
