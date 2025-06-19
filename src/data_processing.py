import re

def normalise_product_name(name):
    """Aggressively normalises a product name for matching"""
    if not isinstance(name, str):
        return ""
    name = name.lower()

    # Remove units
    name = re.sub(r"\b\d+(\.\d+)?\s?(kg|g|ml|l|m|pack|x)\b", "", name)

    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Remove common supermarket brand names to focus on the core product
    brands_to_remove = ["tesco", "asda", "saintsburys", "morrisons", "aldi"]
    for brand in brands_to_remove:
        name = name.replace(brand, "")

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name

