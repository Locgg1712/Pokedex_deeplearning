# src/api.py
# PokéAPI integration — fetches Pokémon data from the public REST API.

import requests

POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon/"

# In-memory cache to avoid duplicate API calls within a session
_cache = {}


def fetch_pokemon_info(pokemon_name):
    """
    Fetch Pokémon info from PokéAPI.

    Args:
        pokemon_name (str): The Pokémon name (lowercase, e.g. "pikachu").

    Returns:
        dict: {
            "name":    str,
            "types":   list[str],
            "height":  float (metres),
            "weight":  float (kg),
            "sprite":  str  (URL to official artwork),
        }
        Returns None if the request fails.
    """
    name_lower = pokemon_name.lower().strip()

    # Check cache first
    if name_lower in _cache:
        return _cache[name_lower]

    try:
        response = requests.get(f"{POKEAPI_BASE}{name_lower}", timeout=5)
        response.raise_for_status()
        data = response.json()

        info = {
            "name":   data["name"].capitalize(),
            "types":  [t["type"]["name"].capitalize() for t in data["types"]],
            "height": data["height"] / 10,   # decimetres → metres
            "weight": data["weight"] / 10,   # hectograms → kg
            "sprite": (data.get("sprites", {})
                           .get("other", {})
                           .get("official-artwork", {})
                           .get("front_default", "")),
        }

        _cache[name_lower] = info
        return info

    except requests.RequestException as e:
        print(f"[api] PokéAPI error for '{name_lower}': {e}")
        return None


if __name__ == "__main__":
    import json as _json
    name = input("Pokémon name: ")
    info = fetch_pokemon_info(name)
    if info:
        print(_json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print("Không tìm thấy dữ liệu.")
