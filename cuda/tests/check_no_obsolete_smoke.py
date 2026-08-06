from pathlib import Path


root = Path(__file__).resolve().parents[1]
obsolete_name = "cuda" + "_hello"
offenders = []

for path in root.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    if any(part.startswith("build") for part in relative.parts):
        continue
    try:
        contents = path.read_text()
    except UnicodeDecodeError:
        continue
    if obsolete_name in contents or obsolete_name in path.name:
        offenders.append(str(relative))

assert not (root / "src" / "main.cu").exists(), "obsolete smoke source remains"
assert not offenders, f"obsolete CUDA smoke references remain: {offenders}"

