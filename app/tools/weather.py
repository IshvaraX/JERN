"""Current weather via OpenWeather (moved here from Saturday). Needs OPENWEATHER_API_KEY in .env."""
import requests

from app.tools.base import Plugin
from constants import WEATHER_URL
from saturday.utils import require_env


class Weather(Plugin):

    name = 'weather'
    description = 'Get the current weather for a city.'
    parameters = {'city': 'city name'}
    required = ('city',)
    expand = True

    def summary(self, arguments):
        return f"check the weather in {arguments.get('city', '')}"

    def run(self, city):
        params = {'q': city, 'appid': require_env('OPENWEATHER_API_KEY'), 'units': 'metric'}
        response = requests.get(WEATHER_URL, params=params, timeout=10)
        if response.status_code == 404:
            return f"error: city not found: {city}"
        response.raise_for_status()
        data = response.json()
        return (f"{data['name']}: {data['main']['temp']:.1f}°C (feels like {data['main']['feels_like']:.1f}°C), "
                f"humidity {data['main']['humidity']}%, wind {data['wind']['speed']} m/s, "
                f"{data['weather'][0]['description']}")
