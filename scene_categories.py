"""Road-localisation groupings for selected Places365 output classes."""

from typing import Dict, Mapping


# Official Places365 class indices. Unlisted classes deliberately map to
# ``other``: HumanSLAM should not invent fine-grained context for irrelevant
# rooms or ambiguous scenes.
PLACES365_GROUP_BY_ID = {
    # Urban roads and public space.
    4: "urban_road",       # alley
    79: "urban_road",      # canal/urban
    112: "urban_road",     # crosswalk
    125: "urban_road",     # downtown
    270: "urban_road",     # plaza
    273: "urban_road",     # promenade
    301: "urban_road",     # shopfront
    307: "urban_road",     # skyscraper
    308: "urban_road",     # slum
    319: "urban_road",     # street

    # Residential context.
    8: "residential",      # apartment_building/outdoor
    49: "residential",     # beach_house
    74: "residential",     # cabin/outdoor
    87: "residential",     # chalet
    107: "residential",    # cottage
    127: "residential",    # driveway
    183: "residential",    # house
    184: "residential",    # hunting_lodge/outdoor
    193: "residential",    # inn/outdoor
    220: "residential",    # mansion
    221: "residential",    # manufactured_home
    231: "residential",    # motel
    283: "residential",    # residential_neighborhood
    296: "residential",    # schoolhouse

    # Commercial roadside context.
    31: "commercial",      # bakery/shop
    47: "commercial",      # bazaar/outdoor
    50: "commercial",      # beauty_salon
    72: "commercial",      # butchers_shop
    99: "commercial",      # coffee_shop
    114: "commercial",     # delicatessen
    119: "commercial",     # diner/outdoor
    128: "commercial",     # drugstore
    139: "commercial",     # fastfood_restaurant
    147: "commercial",     # florist_shop/indoor
    158: "commercial",     # gas_station
    161: "commercial",     # general_store/outdoor
    162: "commercial",     # gift_shop
    172: "commercial",     # hardware_store
    181: "commercial",     # hotel/outdoor
    198: "commercial",     # jewelry_shop
    223: "commercial",     # market/outdoor
    261: "commercial",     # pet_shop
    262: "commercial",     # pharmacy
    267: "commercial",     # pizzeria
    284: "commercial",     # restaurant
    286: "commercial",     # restaurant_patio
    300: "commercial",     # shoe_shop
    302: "commercial",     # shopping_mall/indoor
    321: "commercial",     # supermarket
    335: "commercial",     # toyshop

    # Parking and vehicle storage.
    156: "parking",        # garage/indoor
    157: "parking",        # garage/outdoor
    255: "parking",        # parking_garage/indoor
    256: "parking",        # parking_garage/outdoor
    257: "parking",        # parking_lot

    # Major transport infrastructure.
    0: "major_transport",  # airfield
    2: "major_transport",  # airport_terminal
    66: "major_transport", # bridge
    67: "urban_road",      # building_facade
    71: "major_transport", # bus_station/indoor
    129: "major_transport",# elevator/door (station-like ambiguity)
    171: "major_transport",# harbor
    174: "major_transport",# heliport
    175: "major_transport",# highway
    207: "major_transport",# landing_deck
    216: "major_transport",# loading_dock
    266: "major_transport",# pier
    278: "major_transport",# railroad_track
    293: "major_transport",# runway
    320: "major_transport",# subway_station/platform
    336: "major_transport",# train_interior
    337: "major_transport",# train_station/platform
    347: "major_transport",# viaduct

    # Industrial/built operational areas.
    18: "industrial",      # army_base
    23: "industrial",      # assembly_line
    28: "industrial",      # auto_factory
    103: "industrial",     # construction_site
    133: "industrial",     # engine_room
    136: "industrial",     # excavation
    144: "industrial",     # fire_station
    169: "industrial",     # hangar/indoor
    170: "industrial",     # hangar/outdoor
    192: "industrial",     # industrial_area
    199: "industrial",     # junkyard
    206: "industrial",     # landfill
    247: "industrial",     # oilrig
    282: "industrial",     # repair_shop
    298: "industrial",     # server_room

    # Rural roads and settled countryside.
    118: "rural_road",     # desert_road
    138: "rural_road",     # farm
    142: "rural_road",     # field_road
    152: "rural_road",     # forest_road
    173: "rural_road",     # hayfield
    249: "rural_road",     # orchard
    258: "rural_road",     # pasture
    287: "rural_road",     # rice_paddy
    338: "rural_road",     # tree_farm
    348: "rural_road",     # village
    349: "rural_road",     # vineyard
    359: "rural_road",     # wheat_field
    360: "rural_road",     # wind_farm

    # Natural/open environments.
    30: "natural", 36: "natural", 48: "natural", 62: "natural",
    73: "natural", 76: "natural", 78: "natural", 81: "natural",
    94: "natural", 97: "natural", 104: "natural", 110: "natural",
    111: "natural", 113: "natural", 116: "natural", 117: "natural",
    140: "natural", 141: "natural", 145: "natural", 150: "natural",
    151: "natural", 163: "natural", 164: "natural", 167: "natural",
    180: "natural", 186: "natural", 187: "natural", 190: "natural",
    194: "natural", 204: "natural", 205: "natural", 209: "natural",
    224: "natural", 232: "natural", 233: "natural", 234: "natural",
    243: "natural", 254: "natural", 265: "natural", 271: "natural",
    279: "natural", 288: "natural", 289: "natural", 304: "natural",
    305: "natural", 306: "natural", 309: "natural", 323: "natural",
    324: "natural", 341: "natural", 342: "natural", 344: "natural",
    350: "natural", 355: "natural", 356: "natural", 357: "natural",

    # Distinctive restricted/special-purpose sites.
    5: "restricted_special", 15: "restricted_special",
    16: "restricted_special", 17: "restricted_special",
    24: "restricted_special", 42: "restricted_special",
    68: "restricted_special", 77: "restricted_special",
    86: "restricted_special", 108: "restricted_special",
    132: "restricted_special", 149: "restricted_special",
    168: "restricted_special", 178: "restricted_special",
    188: "restricted_special", 189: "restricted_special",
    225: "restricted_special", 230: "restricted_special",
    251: "restricted_special", 252: "restricted_special",
    275: "restricted_special", 276: "restricted_special",
    310: "restricted_special", 312: "restricted_special",
    313: "restricted_special", 314: "restricted_special",
    327: "restricted_special", 330: "restricted_special",
    351: "restricted_special", 354: "restricted_special",
}


COMPATIBLE_GROUPS = {
    "urban_road": {"commercial", "residential", "parking"},
    "residential": {"urban_road", "rural_road", "parking"},
    "commercial": {"urban_road", "parking"},
    "parking": {"commercial", "residential", "urban_road"},
    "major_transport": {"urban_road", "industrial"},
    "industrial": {"major_transport", "urban_road"},
    "rural_road": {"natural", "residential"},
    "natural": {"rural_road"},
    "restricted_special": set(),
    "other": set(),
}


def group_places365_probabilities(
    probabilities, top_k: int = 3
) -> Dict[str, float]:
    """Aggregate the top-k Places365 probabilities into broad scene groups."""
    import numpy as np

    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if values.size == 0 or top_k <= 0:
        return {}
    count = min(int(top_k), values.size)
    indices = np.argpartition(values, -count)[-count:]
    grouped: Dict[str, float] = {}
    for class_id in indices:
        group = PLACES365_GROUP_BY_ID.get(int(class_id), "other")
        grouped[group] = grouped.get(group, 0.0) + max(0.0, float(values[class_id]))
    total = sum(grouped.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in grouped.items()}


def category_compatibility(
    query: Mapping[str, float], candidate: Mapping[str, float]
) -> float:
    """Expected compatibility between two grouped top-k distributions."""
    if not query or not candidate:
        return 0.0
    score = 0.0
    for q_group, q_probability in query.items():
        for c_group, c_probability in candidate.items():
            if q_group == c_group and q_group != "other":
                compatibility = 1.0
            elif c_group in COMPATIBLE_GROUPS.get(q_group, set()):
                compatibility = 0.6
            elif "other" in (q_group, c_group):
                compatibility = 0.3
            else:
                compatibility = 0.0
            score += float(q_probability) * float(c_probability) * compatibility
    return max(0.0, min(1.0, score))
