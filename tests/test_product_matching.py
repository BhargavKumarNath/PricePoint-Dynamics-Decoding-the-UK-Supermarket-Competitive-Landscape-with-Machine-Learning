"""Tests for the product matching module — normalise_product_name."""

from __future__ import annotations


from pricepoint.product_matching import normalise_product_name


class TestNormaliseProductName:
    """Tests for normalise_product_name."""

    # ---- Basic normalisation ----

    def test_lowercase(self):
        assert normalise_product_name("HELLO WORLD") == "hello world"

    def test_strip_units_grams(self):
        assert normalise_product_name("Chicken Breast 500g") == "chicken breast"

    def test_strip_units_kg(self):
        assert normalise_product_name("Potatoes 2.5kg") == "potatoes"

    def test_strip_units_litres(self):
        assert normalise_product_name("Milk 1l") == "milk"

    def test_strip_units_ml(self):
        assert normalise_product_name("Orange Juice 500ml") == "orange juice"

    def test_strip_units_pack(self):
        assert normalise_product_name("Bananas 5pack") == "bananas"

    def test_strip_units_x_multiplier(self):
        assert normalise_product_name("Crisps 6x25g") == "crisps"

    # ---- Brand removal ----

    def test_remove_tesco(self):
        assert normalise_product_name("Tesco Finest Bananas") == "finest bananas"

    def test_remove_asda(self):
        assert normalise_product_name("ASDA Whole Milk") == "whole milk"

    def test_remove_aldi(self):
        assert normalise_product_name("Aldi Nature's Pick Bananas") == "natures pick bananas"

    def test_remove_morrisons(self):
        assert normalise_product_name("Morrisons The Best Bread") == "the best bread"

    def test_remove_sainsburys(self):
        # The function removes "saintsburys" (original spelling)
        assert normalise_product_name("Saintsburys SO Organic Milk") == "so organic milk"

    # ---- Punctuation ----

    def test_remove_punctuation(self):
        result = normalise_product_name("Ben & Jerry's Ice Cream")
        assert "&" not in result
        assert "'" not in result

    # ---- Whitespace ----

    def test_collapse_whitespace(self):
        assert normalise_product_name("  too   many   spaces  ") == "too many spaces"

    # ---- Edge cases ----

    def test_none_input(self):
        assert normalise_product_name(None) == ""

    def test_empty_string(self):
        assert normalise_product_name("") == ""

    def test_numeric_input(self):
        assert normalise_product_name(12345) == ""

    def test_only_units(self):
        # After removing "500g", only whitespace remains
        assert normalise_product_name("500g") == ""

    # ---- Regression tests ----

    def test_real_product_tesco_bananas(self):
        result = normalise_product_name("Tesco Finest Bananas 5pk - 1.2kg")
        assert "tesco" not in result
        assert "5pk" not in result
        assert "bananas" in result

    def test_real_product_walkers_crisps(self):
        result = normalise_product_name("Walkers Meaty Variety Crisps 12x25g")
        assert "12x25g" not in result
        assert "walkers" in result  # Walkers is NOT a retailer brand
