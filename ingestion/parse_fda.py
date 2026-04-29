import xml.etree.ElementTree as ET
from pathlib import Path
from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR
import json
import re


# FDA SPL XML uses a namespace prefix on every tag.
# Without this, ET.find() returns nothing — common gotcha.
NS = {"spl": "urn:hl7-org:v3"}

def extract_sections(xml_path: Path) -> list[dict]:
    """
    Parse one FDA SPL XML file.
    Returns a list of dicts, one per section:
    {"drug_name": str, "section": str, "text": str}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # The drug name lives in a specific tag path in SPL format
    drug_name_el = root.find(".//spl:manufacturedProduct/spl:manufacturedProduct/spl:name", NS)
    drug_name = drug_name_el.text.strip() if drug_name_el is not None else xml_path.stem

    sections = []

    # Each major section in an FDA label is a <section> tag
    # with a <code displayName="..."> child that names the section
    for section_el in root.findall(".//spl:section", NS):
        code_el = section_el.find("spl:code", NS)
        if code_el is None:
            continue

        section_name = code_el.get("displayName", "unknown")

        # Only keep sections we care about — skip boilerplate
        KEEP_SECTIONS = {
            "INDICATIONS & USAGE SECTION",
            "CONTRAINDICATIONS SECTION",
            "WARNINGS AND PRECAUTIONS SECTION",
            "DOSAGE & ADMINISTRATION SECTION",
            "ADVERSE REACTIONS SECTION",
            "DRUG INTERACTIONS SECTION",
            "BOXED WARNING SECTION",
            "USE IN SPECIFIC POPULATIONS SECTION",
            "OVERDOSAGE SECTION",
            "CLINICAL PHARMACOLOGY SECTION",
            "MECHANISM OF ACTION SECTION",
            "WARNINGS SECTION",          # some other labels may use this
            "PRECAUTIONS SECTION",       # some older labels may use this
        }
        if section_name not in KEEP_SECTIONS:
            continue

        # itertext() walks all child tags and collects text nodes
        # This handles nested <paragraph>, <list>, etc. cleanly
        raw_text = " ".join(section_el.itertext()).strip()

        # After collecting raw_text, add this line:
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()

        raw_text = raw_text.replace('\u00a0', ' ')
        
        # Skip empty sections
        if len(raw_text) < 50:
            continue

        sections.append({
            "drug_name": drug_name,
            "section": section_name,
            "text": raw_text,
        })

    return sections


def parse_all(limit: int = None) -> list[dict]:
    """
    Parse every XML file in DATA_RAW_DIR.
    limit=20 during development so you don't wait for all files.
    """
    xml_files = list(DATA_RAW_DIR.glob("*.xml"))
    if limit:
        xml_files = xml_files[:limit]

    all_sections = []
    for path in xml_files:
        try:
            all_sections.extend(extract_sections(path))
        except ET.ParseError as e:
            # Some FDA files are malformed — skip them, don't crash
            print(f"Skipping {path.name}: {e}")

    return all_sections


if __name__ == "__main__":
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sections = parse_all(limit=5)
    
    output_path = DATA_PROCESSED_DIR / "sections.json"
    with open(output_path, "w") as f:
        json.dump(sections, f, indent=2)

    print(f"Parsed {len(sections)} sections")
    print(f"\nFirst section preview:")
    if sections:
        print(f"  Drug: {sections[0]['drug_name']}")
        print(f"  Section: {sections[0]['section']}")
        print(f"  Text preview: {sections[0]['text'][:300]}")