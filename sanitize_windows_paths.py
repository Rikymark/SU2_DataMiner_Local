#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

# Caratteri vietati su Windows (Win32):  \ / : * ? " < > |  (in Git / è separatore path)
# Qui trattiamo quelli che possono stare nei nomi in repo Linux ma non in Windows.
ILLEGAL_CHARS = re.compile(r'[:*?"<>|]')

# Nomi riservati su Windows (case-insensitive), anche con estensione (es. CON.txt è vietato)
RESERVED_BASE = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}

def is_reserved_windows(name: str) -> bool:
    base = name.split(".")[0]  # CON.txt -> CON
    return base.upper() in RESERVED_BASE

def sanitize_component(comp: str) -> str:
    # Rimpiazza caratteri illegali
    comp2 = ILLEGAL_CHARS.sub("-", comp)

    # Windows vieta nomi che terminano con spazio o punto
    comp2 = comp2.rstrip(" .")

    # Evita componenti vuote o speciali
    if comp2 in ("", ".", ".."):
        comp2 = "_"

    # Evita nomi riservati
    if is_reserved_windows(comp2):
        comp2 = "_" + comp2

    return comp2

def unique_name(target: Path) -> Path:
    """
    Se target esiste già, crea una variante unica:
    file.ext -> file__2.ext
    dir -> dir__2
    """
    parent = target.parent
    name = target.name
    stem = target.stem
    suffix = target.suffix  # include il punto

    i = 2
    while True:
        if suffix:
            candidate = parent / f"{stem}__{i}{suffix}"
        else:
            candidate = parent / f"{name}__{i}"
        if not candidate.exists():
            return candidate
        i += 1

def compute_new_path(p: Path) -> Path:
    parts = p.parts
    new_parts = []
    for comp in parts:
        new_parts.append(sanitize_component(comp))
    return Path(*new_parts)

def gather_file_paths(root: Path):
    """
    Raccoglie SOLO file (non directory) escluso .git.
    Ordina per profondità decrescente (prima i file più profondi).
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        dp = Path(dirpath)
        for fn in filenames:
            paths.append(dp / fn)

    paths.sort(key=lambda x: len(x.parts), reverse=True)
    return paths


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("--dry-run", "--apply"):
        print("Uso: sanitize_windows_paths.py --dry-run | --apply", file=sys.stderr)
        sys.exit(2)

    mode_apply = (sys.argv[1] == "--apply")
    root = Path(".").resolve()

    paths = gather_file_paths(root)

    # Cleanup directory vuote (non tracciate da Git)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        dp = Path(dirpath)
        if dp.name == ".git":
            continue
        # prova a rimuovere solo se vuota
        try:
            dp.rmdir()
        except OSError:
            pass


    # Mappa case-insensitive per prevenire collisioni tipiche di Windows.
    # (Windows è case-insensitive di default)
    planned_ci = set()

    changes = []
    for abs_path in paths:
        rel_path = abs_path.relative_to(root)
        new_rel = compute_new_path(rel_path)

        if new_rel == rel_path:
            continue

        # Collisione case-insensitive
        key = str(new_rel).lower()
        if key in planned_ci or (root / new_rel).exists():
            # Trova un nome unico
            new_abs = unique_name(root / new_rel)
            new_rel = new_abs.relative_to(root)
            key = str(new_rel).lower()

        planned_ci.add(key)
        changes.append((rel_path, new_rel))

    if not changes:
        print("Nessun path da rinominare: working tree già compatibile Windows.")
        return

    print(f"Trovati {len(changes)} path da rinominare.")
    for old, new in changes[:50]:
        print(f"  {old}  ->  {new}")
    if len(changes) > 50:
        print(f"  ... e altri {len(changes)-50} ...")

    if not mode_apply:
        print("\nDry-run completato. Esegui con --apply per applicare i rename.")
        return

    # Applica rinomine in ordine (già profondità decrescente)
    for old, new in changes:
        old_abs = root / old
        new_abs = root / new

        # Assicura che la dir di destinazione esista
        new_abs.parent.mkdir(parents=True, exist_ok=True)

        # Esegui rename
        old_abs.rename(new_abs)

    print("\nRename applicati con successo.")

if __name__ == "__main__":
    main()
