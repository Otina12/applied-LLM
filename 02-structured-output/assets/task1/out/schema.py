from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str  # Name of the city
    country: str  # Country where the city is located
    population: int  # Population of the city
    landmarks: list[str]  # Famous landmarks in the city
    official_language: str  # Official language of the city
    average_summer_temperature: float  # Average summer temperature in Celsius