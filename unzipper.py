import zipfile
from pathlib import Path

ZIPS_DIR = Path("E:/RAG project/data/raw")      # where your 20 zips are
XML_OUT  = Path("E:/RAG project/data/raw/xml")  # where extracted XMLs go
XML_OUT.mkdir(exist_ok=True)

for zip_path in ZIPS_DIR.glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            # Only extract XML files, skip images and other junk
            if name.endswith(".xml"):
                # Extract and rename to something readable
                data = z.read(name)
                out_path = XML_OUT / f"{zip_path.stem}.xml"
                out_path.write_bytes(data)
                print(f"Extracted: {out_path.name}")