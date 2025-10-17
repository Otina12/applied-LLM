import json
import sys
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

def main():
    if len(sys.argv) != 3 or sys.argv[1] != '--json_path':
        print("Usage: python main.py --json_path <path_to_json>")
        sys.exit(1)

    json_path = sys.argv[2]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading JSON file.")
        sys.exit(1)

    cities = {city: coords for city, coords in data.items() if coords.get('lat') is not None and coords.get('lon') is not None}

    if len(cities) < 2:
        print("Not enough cities to compute distances.")
        sys.exit(0)

    closest_pair = None
    min_distance = float('inf')

    city_names = list(cities.keys())
    for i in range(len(city_names)):
        for j in range(i + 1, len(city_names)):
            city1 = city_names[i]
            city2 = city_names[j]
            lat1, lon1 = cities[city1]['lat'], cities[city1]['lon']
            lat2, lon2 = cities[city2]['lat'], cities[city2]['lon']
            distance = haversine(lat1, lon1, lat2, lon2)

            if distance < min_distance:
                min_distance = distance
                closest_pair = (city1, city2)

    if closest_pair:
        print(f"Closest pair: {closest_pair[0]} and {closest_pair[1]}")
        print(f"Distance (km): {min_distance:.3f}")

if __name__ == "__main__":
    main()