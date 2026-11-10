import os
import re
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# NHTSA vPIC API - VIN Decode
# Docs: https://vpic.nhtsa.dot.gov/api/
# No API key required, completely free and public.
# NOTE: vPIC data represents vehicles intended for sale/import into the US.
# Non-US-market vehicles (very common in Turkey/EU) may return partial data
# or check-digit "errors" even when the VIN itself was read correctly.
# ---------------------------------------------------------------------------
def decode_vin(vin, model_year=None):
    """
    Given a VIN, returns key vehicle details (make, model, year, engine, etc.)
    directly from NHTSA's official database — not from the model's memory.
    NHTSA recommends always passing the model year when known, since it helps
    the decoder disambiguate ambiguous VIN positions.
    """
    if not vin or len(vin.strip()) != 17:
        return {
            "error": "Geçerli bir VIN sağlanmadı (VIN 17 karakter olmalı). Bu aracı çağırmadan önce kullanıcıdan gerçek bir VIN istenmeliydi."}

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json"
    if model_year:
        url += f"&modelyear={model_year}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("Results"):
            return {"error": "Could not decode VIN, no results found."}

        result = data["Results"][0]

        relevant_fields = [
            "Make", "Model", "ModelYear", "Trim", "EngineCylinders",
            "DisplacementL", "FuelTypePrimary", "TransmissionStyle",
            "DriveType", "VehicleType", "BodyClass", "ErrorText"
        ]
        filtered = {field: result.get(field) for field in relevant_fields if result.get(field)}
        return filtered

    except requests.exceptions.RequestException as e:
        return {"error": f"VIN decode API error: {str(e)}"}


# ---------------------------------------------------------------------------
# NHTSA Recalls & Complaints API
# Docs: https://api.nhtsa.gov/
# No API key required.
# ---------------------------------------------------------------------------
def get_recalls(make, model, model_year):
    """
    Returns official recall records for a given make/model/year from NHTSA.
    """
    url = "https://api.nhtsa.gov/recalls/recallsByVehicle"
    params = {"make": make, "model": model, "modelYear": model_year}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return {"recall_count": 0, "recalls": []}

        # NOTE: recall_number comes from NHTSA's official NHTSACampaignNumber
        # field — this MUST be included, otherwise the model has no real
        # recall ID to reference and may fabricate one.
        summarized = [
            {
                "recall_number": r.get("NHTSACampaignNumber"),
                "component": r.get("Component"),
                "summary": r.get("Summary"),
                "consequence": r.get("Consequence"),
                "remedy": r.get("Remedy"),
                "report_date": r.get("ReportReceivedDate"),
            }
            for r in results[:20]
        ]
        return {"recall_count": len(results), "recalls": summarized}

    except requests.exceptions.RequestException as e:
        return {"error": f"Recall API error: {str(e)}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Recall API error: {str(e)}"}


def get_complaints(make, model, model_year):
    """
    Returns owner-reported complaint records for a given make/model/year from NHTSA.
    """
    url = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
    params = {"make": make, "model": model, "modelYear": model_year}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return {"complaint_count": 0, "complaints": []}

        summarized = [
            {
                "component": r.get("components"),
                "summary": r.get("summary"),
                "date_of_incident": r.get("dateOfIncident"),
            }
            for r in results[:3]
        ]
        return {"complaint_count": len(results), "complaints": summarized}

    except requests.exceptions.RequestException as e:
        return {"error": f"Complaints API error: {str(e)}"}


# ---------------------------------------------------------------------------
# Petrol Ofisi - Turkey Fuel Prices (Scraping)
# Source: https://www.petrolofisi.com.tr/akaryakit-fiyatlari
# NO API key required. The page contains prices for 81 cities in a single HTML table,
# so we parse it using requests + BeautifulSoup.
# ---------------------------------------------------------------------------
_TR_CHAR_MAP = str.maketrans({
    "İ": "I", "I": "I", "ı": "i", "Ş": "S", "ş": "s",
    "Ğ": "G", "ğ": "g", "Ü": "U", "ü": "u",
    "Ö": "O", "ö": "o", "Ç": "C", "ç": "c"
})


def _normalize_tr(text):
    """Normalizes Turkish city names for case/diacritic-insensitive matching."""
    return text.translate(_TR_CHAR_MAP).upper().strip()


# (anahtar kelimeler, Türkçe etiket, varsayılan olarak gösterilsin mi)

_TR_FOLD = str.maketrans({
    "İ": "i", "I": "i", "ı": "i",
    "Ş": "s", "ş": "s", "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u", "Ö": "o", "ö": "o",
    "Ç": "c", "ç": "c",
})

FUEL_CATEGORIES = [
    (("kursunsuz", "benzin"), "Benzin", True),
    (("diesel", "motorin", "mazot"), "Motorin", True),
    (("otogaz", "lpg"), "LPG", True),
    (("gazyagi", "kerosen", "kerozin"), "Gazyağı (kerozin)", False),
    (("kalorifer", "isitma"), "Kalorifer yakıtı", False),
    (("fuel",), "Fuel oil", False),
]


def _label_fuel(header):
    """Tablo başlığını Türkçe etikete çevirir. -> (etiket, varsayılan_gösterilsin_mi)"""
    h = header.translate(_TR_FOLD).lower()
    for keys, label, default in FUEL_CATEGORIES:
        if any(k in h for k in keys):
            if label == "Benzin":
                m = re.search(r"\b(9\d)\b", h)
                if m:
                    label = f"Benzin ({m.group(1)} oktan)"
            return label, default
    return header, False  # tanınmayan ürün: sadece include_other_products=True ise

def get_fuel_prices(city="istanbul", include_other_products=False):
    """
    Scrapes current fuel prices for a given Turkish city from Petrol Ofisi's
    public price page. No API key needed.

    By default only vehicle fuels (gasoline, diesel, LPG) are returned.
    Kerosene, heating oil and fuel oil are returned only when
    include_other_products is True, so the model never even sees them
    for a generic "fuel prices" question.
    """
    url = "https://www.petrolofisi.com.tr/akaryakit-fiyatlari"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AutomotiveBot/1.0)"},
            timeout=15
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return {"error": "Price table not found on the page (site structure might have changed)."}

        rows = table.find_all("tr")
        if not rows:
            return {"error": "No rows found in the table."}

        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

        normalized_target = _normalize_tr(city)
        matches = []

        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells:
                continue

            city_cell = cells[0].get_text(strip=True)
            if normalized_target not in _normalize_tr(city_cell):
                continue

            row_data = {"sehir_bolge": city_cell}
            for header, cell in zip(headers[1:], cells[1:]):
                label, is_default = _label_fuel(header)
                if not (is_default or include_other_products):
                    continue
                if label in row_data:  # aynı etikete düşen ikinci sütunu atla
                    continue
                numbers = re.findall(r"\d+[.,]\d+", cell.get_text(" ", strip=True))
                if not numbers:
                    continue
                row_data[label] = f"{numbers[0].replace('.', ',')} TL/litre (KDV dahil)"

            matches.append(row_data)

        if not matches:
            return {"error": f"No prices found for '{city}'. Check the city name."}

        out = {"results": matches}
        if len(matches) > 1:
            out["note"] = (
                "Birden fazla bölge döndü. Her bölgeyi 'sehir_bolge' alanındaki "
                "adıyla (ör. İstanbul (Avrupa), İstanbul (Anadolu)) ayrı başlık "
                "altında yaz. Bölge adını asla atlama, iki bölgeyi birleştirme."
            )
        return out

    except requests.exceptions.RequestException as e:
        return {"error": f"Could not fetch fuel prices page: {str(e)}"}

