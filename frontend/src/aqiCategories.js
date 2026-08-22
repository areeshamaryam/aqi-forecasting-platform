// Mirrors backend/model_service.py's categorize_aqi() exactly -
// same cumulative <= thresholds, so the frontend never disagrees
// with what the API already labeled.

export const AQI_CATEGORIES = [
  {
    max: 50,
    label: "Good",
    color: "var(--aqi-good)",
    bg: "var(--aqi-good-bg)",
  },
  {
    max: 100,
    label: "Moderate",
    color: "var(--aqi-moderate)",
    bg: "var(--aqi-moderate-bg)",
  },
  {
    max: 150,
    label: "Unhealthy for Sensitive Groups",
    color: "var(--aqi-usg)",
    bg: "var(--aqi-usg-bg)",
  },
  {
    max: 200,
    label: "Unhealthy",
    color: "var(--aqi-unhealthy)",
    bg: "var(--aqi-unhealthy-bg)",
  },
  {
    max: 300,
    label: "Very Unhealthy",
    color: "var(--aqi-very-unhealthy)",
    bg: "var(--aqi-very-unhealthy-bg)",
  },
  {
    max: Infinity,
    label: "Hazardous",
    color: "var(--aqi-hazardous)",
    bg: "var(--aqi-hazardous-bg)",
  },
];

export function categoryFor(aqiValue) {
  return (
    AQI_CATEGORIES.find((c) => aqiValue <= c.max) ??
    AQI_CATEGORIES[AQI_CATEGORIES.length - 1]
  );
}
