export default function ExplanationPanel({ explanation }) {
  if (!explanation || !explanation.top_features?.length) return null;

  const maxAbs = Math.max(
    ...explanation.top_features.map((f) => Math.abs(f.contribution)),
  );

  return (
    <section className="explanation">
      <p className="explanation__title">
        Why this forecast{" "}
        <span className="explanation__subtitle">
          — top drivers of the {explanation.horizon_hours}h prediction
        </span>
      </p>
      <div className="explanation__bars">
        {explanation.top_features.map((f) => {
          const widthPct = (Math.abs(f.contribution) / maxAbs) * 100;
          const isPositive = f.contribution >= 0;
          return (
            <div className="explanation__row" key={f.feature}>
              <span className="explanation__feature">{f.feature}</span>
              <div className="explanation__track">
                <div
                  className={`explanation__fill explanation__fill--${
                    isPositive ? "up" : "down"
                  }`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <span className="explanation__value mono">
                {isPositive ? "+" : ""}
                {f.contribution.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="explanation__legend">
        <span className="explanation__legend-dot explanation__legend-dot--up" />{" "}
        pushes AQI higher &nbsp;&nbsp;
        <span className="explanation__legend-dot explanation__legend-dot--down" />{" "}
        pushes AQI lower
      </p>
    </section>
  );
}
