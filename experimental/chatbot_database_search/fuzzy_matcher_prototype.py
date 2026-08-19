import json
import re
import difflib

def clean_product_name(name):
    """DL-06-T11: Cleans and normalises product names for better matching."""
    if not name:
        return ""
    # Lowercase everything
    name = name.lower()
    # Standardise weights (e.g., 'kilos' to 'kg', 'grams' to 'g')
    name = re.sub(r'\bkilos?\b', 'kg', name)
    name = re.sub(r'\bgrams?\b', 'g', name)
    name = re.sub(r'\blitres?\b', 'l', name)
    # Remove special characters but keep spaces and alphanumeric
    name = re.sub(r'[^a-z0-9\s]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fuzzy_match_product(extracted_product, catalog, threshold=0.55):
    """DL-06-T12: Uses difflib to find the closest fuzzy match in the catalog."""
    cleaned_query = clean_product_name(extracted_product)
    
    best_match = None
    highest_score = 0.0

    for item in catalog:
        cleaned_item_name = clean_product_name(item['product_name'])
        # Calculate similarity score (returns a float between 0 and 1)
        score = difflib.SequenceMatcher(None, cleaned_query, cleaned_item_name).ratio()
        
        if score > highest_score:
            highest_score = score
            best_match = item

    # DL-06-T9: Apply confidence threshold logic
    if highest_score >= threshold:
        return {
            "match_found": True,
            "confidence_score": round(highest_score, 2),
            "matched_product": best_match
        }
    else:
        return {
            "match_found": False,
            "confidence_score": round(highest_score, 2),
            "message": "Low confidence: No products met the matching threshold."
        }

if __name__ == "__main__":
    # Load the dummy catalog to run local tests
    try:
        with open("dummy_catalog.json", "r") as f:
            catalog = json.load(f)
    except FileNotFoundError:
        print("Error: dummy_catalog.json not found.")
        catalog = []

    if catalog:
        # Test cases simulating imperfect extractions from the NLP script
        test_entities = [
            "full cream milk",       # Partial match
            "devon butter 250 grams", # Typos and un-normalized weights
            "coles dairy milk",      # Exact brand match
            "organic apples"         # Item not in catalog (should fail threshold)
        ]
        
        print("Running Fuzzy Product Matching Tests...\n")
        for entity in test_entities:
            print(f"User Extracted Entity: '{entity}'")
            result = fuzzy_match_product(entity, catalog)
            
            if result["match_found"]:
                print(f"Matched with: {result['matched_product']['product_name']} (ID: {result['matched_product']['product_id']})")
                print(f"   Score: {result['confidence_score']}")
            else:
                print(f"Match failed. Best score: {result['confidence_score']} - {result['message']}")
            print("-" * 50)