from ml_pipeline.data_collection.openmeteo_client import OpenMeteoClient


client = OpenMeteoClient()

weather, air_quality = client.get_historical_data(
    latitude=33.6844,
    longitude=73.0479,
    start_date="2026-07-25",
    end_date="2026-07-27",
)

print("\n✅ Open-Meteo connection successful!")

print("\nWeather:")
print(weather.keys())

print("\nAir Quality:")
print(air_quality.keys())

print("\nNumber of weather timestamps:")
print(len(weather["hourly"]["time"]))

print("\nNumber of air-quality timestamps:")
print(len(air_quality["hourly"]["time"]))

print("\nFirst weather timestamp:")
print(weather["hourly"]["time"][0])

print("\nFirst AQI value:")
print(air_quality["hourly"]["us_aqi"][0])