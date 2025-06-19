import re

def normalise_product_name(name):
    """Aggressively normalises a product name for matching"""
    if not isinstance(name, str)