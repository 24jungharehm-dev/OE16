"""
Product catalog tool for retrieving product information.

Provides a mock seasonal product catalog that agents can query
for product-specific content suggestions.
"""

# OpenAI-compatible tool definition for agent tool calling
CATALOG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "product_catalog",
        "description": (
            "Look up product details from the brand's catalog. Returns product "
            "names, descriptions, key features, price ranges, and seasonal relevance. "
            "Use this to create product-specific content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Product category to look up, e.g., 'summer sunglasses', "
                        "'winter jackets', 'fitness gear', 'skincare'."
                    ),
                }
            },
            "required": ["category"],
        },
    },
}

# ---- Mock seasonal product catalog ----
_CATALOG = {
    "sunglasses": {
        "products": [
            {
                "name": "AeroShade Aviators",
                "description": "Lightweight titanium aviators with UV400 polarized lenses",
                "features": ["Polarized UV400", "Titanium frame", "Anti-scratch coating"],
                "price_range": "$89 - $129",
                "season": "Summer",
                "best_seller": True,
            },
            {
                "name": "UrbanEdge Wayfarers",
                "description": "Classic wayfarer silhouette with eco-friendly acetate frames",
                "features": ["Eco-acetate", "Gradient lenses", "Unisex design"],
                "price_range": "$69 - $99",
                "season": "All-year",
                "best_seller": False,
            },
            {
                "name": "SportFlex Wraps",
                "description": "Performance sport sunglasses with grip-lock technology",
                "features": ["Non-slip grip", "Impact resistant", "Ventilated frame"],
                "price_range": "$59 - $89",
                "season": "Summer",
                "best_seller": True,
            },
        ],
        "brand_tagline": "See the world differently.",
        "target_vibe": "Active, stylish, eco-conscious",
    },
    "skincare": {
        "products": [
            {
                "name": "HydraGlow Serum",
                "description": "Hyaluronic acid serum with vitamin C for radiant skin",
                "features": ["Hyaluronic acid", "Vitamin C", "Fragrance-free"],
                "price_range": "$34 - $48",
                "season": "All-year",
                "best_seller": True,
            },
            {
                "name": "SunShield SPF 50",
                "description": "Lightweight daily sunscreen with blue-light protection",
                "features": ["SPF 50", "Blue-light filter", "Non-greasy"],
                "price_range": "$28 - $38",
                "season": "Summer",
                "best_seller": True,
            },
            {
                "name": "NightRepair Cream",
                "description": "Retinol night cream for cell renewal and anti-aging",
                "features": ["Retinol", "Peptides", "Overnight repair"],
                "price_range": "$42 - $58",
                "season": "All-year",
                "best_seller": False,
            },
        ],
        "brand_tagline": "Your skin, perfected.",
        "target_vibe": "Clean beauty, self-care, wellness",
    },
    "fitness": {
        "products": [
            {
                "name": "FlexFit Leggings",
                "description": "High-waist compression leggings with pocket",
                "features": ["4-way stretch", "Squat-proof", "Hidden pocket"],
                "price_range": "$49 - $69",
                "season": "All-year",
                "best_seller": True,
            },
            {
                "name": "PowerGrip Resistance Bands",
                "description": "Set of 5 fabric resistance bands with carry bag",
                "features": ["Non-slip fabric", "5 resistance levels", "Portable"],
                "price_range": "$24 - $34",
                "season": "New Year / Summer",
                "best_seller": True,
            },
            {
                "name": "AquaFlow Water Bottle",
                "description": "32oz insulated water bottle with time markers",
                "features": ["Double-wall insulated", "Time markers", "BPA-free"],
                "price_range": "$19 - $29",
                "season": "All-year",
                "best_seller": False,
            },
        ],
        "brand_tagline": "Move. Sweat. Repeat.",
        "target_vibe": "Energetic, motivational, community-driven",
    },
    "fashion": {
        "products": [
            {
                "name": "StreetWave Hoodie",
                "description": "Oversized organic cotton hoodie with embroidered logo",
                "features": ["Organic cotton", "Oversized fit", "Embroidered"],
                "price_range": "$59 - $79",
                "season": "Fall / Winter",
                "best_seller": True,
            },
            {
                "name": "MinimalTee Collection",
                "description": "Pack of 3 essential tees in neutral tones",
                "features": ["Pima cotton", "Pre-shrunk", "Neutral palette"],
                "price_range": "$39 - $54",
                "season": "All-year",
                "best_seller": False,
            },
        ],
        "brand_tagline": "Wear your vibe.",
        "target_vibe": "Streetwear, minimalist, sustainable",
    },
    "tech": {
        "products": [
            {
                "name": "PulseBuds Pro",
                "description": "Active noise-cancelling wireless earbuds, 36hr battery",
                "features": ["ANC", "36hr battery", "IPX5 waterproof"],
                "price_range": "$79 - $119",
                "season": "All-year",
                "best_seller": True,
            },
            {
                "name": "DeskFlow Stand",
                "description": "Adjustable laptop stand with built-in USB hub",
                "features": ["Aluminum build", "USB-C hub", "Foldable"],
                "price_range": "$49 - $69",
                "season": "Back-to-school",
                "best_seller": False,
            },
        ],
        "brand_tagline": "Tech that fits your flow.",
        "target_vibe": "Productivity, modern lifestyle, premium",
    },
}


def product_catalog_tool(category: str) -> str:
    """
    Look up products in the catalog by category.

    Performs fuzzy matching against catalog keys so the LLM doesn't
    need to provide an exact key.

    Args:
        category: Product category to search for.

    Returns:
        Formatted product catalog information.
    """
    category_lower = category.lower()

    # Fuzzy match: check if any catalog key appears in the query or vice versa
    matched_key = None
    for key in _CATALOG:
        if key in category_lower or category_lower in key:
            matched_key = key
            break

    # Broader match: check individual words
    if not matched_key:
        for key in _CATALOG:
            for word in category_lower.split():
                if word in key or key in word:
                    matched_key = key
                    break
            if matched_key:
                break

    if not matched_key:
        available = ", ".join(_CATALOG.keys())
        return (
            f"No products found for category '{category}'. "
            f"Available categories: {available}. "
            f"Suggest creative content ideas based on general knowledge."
        )

    catalog = _CATALOG[matched_key]
    lines = [
        f"📦 **Product Catalog: {matched_key.title()}**",
        f"Brand Tagline: \"{catalog['brand_tagline']}\"",
        f"Target Vibe: {catalog['target_vibe']}",
        "",
        "**Products:**",
    ]

    for product in catalog["products"]:
        best = " ⭐ BEST SELLER" if product.get("best_seller") else ""
        lines.append(f"\n• **{product['name']}**{best}")
        lines.append(f"  {product['description']}")
        lines.append(f"  Features: {', '.join(product['features'])}")
        lines.append(f"  Price: {product['price_range']}")
        lines.append(f"  Season: {product['season']}")

    return "\n".join(lines)
