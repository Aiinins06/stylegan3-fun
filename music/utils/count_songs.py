import os
import re

BASE_DATA_PATH = r"/datafast/105-1/Datasets/INTERNS/anavarror/songs" 
GENRE = ["Electrónico", "Hip Hop Convencional", "Ambiente", "Rock", "Pop", "Jazz Clásico", "Techno Y Trance", "Clásica", 
         "Salsa", "Rnb", "Mundo", "Metal", "Punk", "Pop indie", "Alternativa", "Piano clasico"]   

"""
    Electrónico: 2188 files with 1050 unique songs
    Hip Hop Convencional: 2559 files with 1436 unique songs
    Ambiente: 6083 files with 1004 unique songs
    Rock: 2454 files with 1127 unique songs
    Pop: 2371 files with 1083 unique songs
    Jazz Clásico: 2252 files with 1055 unique songs
    Techno Y Trance: 3047 files with 1031 unique songs
    Clásica: 2373 files with 1143 unique songs
    Salsa: 2182 files with 1055 unique songs
    Rnb: 2360 files with 1017 unique songs
    Mundo: 2392 files with 1016 unique songs
    Metal: 1950 files with 799 unique songs
    Punk: 135 files with 63 unique songs
    Pop indie: 2387 files with 999 unique songs
    Alternativa: 2501 files with 1031 unique songs
    Piano clasico: 2938 files with 925 unique songs
"""


def sanitize_folder_name(name):
    sanitized = re.sub(r'[<>:"/\\|?*&]', ' ', name)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

def normalize_song_name(filename):
    """
    Remove any '_partN' suffix so all parts count as one track.
    Example:
      'Song - Artist_part1.mp3' -> 'Song - Artist'
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Remove suffixes like _part1, _part2, _part10, etc.
    name = re.sub(r"_part\d+$", "", name, flags=re.IGNORECASE)
    return name

def count_unique_songs(base_path, genre):
    safe_genre = sanitize_folder_name(genre)
    genre_folder = os.path.join(base_path, safe_genre)

    if not os.path.exists(genre_folder):
        print(f"❌ Genre folder not found: {genre_folder}")
        return

    mp3_files = [
        f for f in os.listdir(genre_folder)
        if os.path.isfile(os.path.join(genre_folder, f)) and f.lower().endswith(".mp3")
    ]

    unique_tracks = set(normalize_song_name(f) for f in mp3_files)

    print(f"\n🎵 Genre: {genre}")
    # print(f"📁 Folder: {genre_folder}")
    print(f"🎧 Total downloaded MP3 files (including parts): {len(mp3_files)}")
    print(f"✅ Unique songs (counting split tracks only once): {len(unique_tracks)}\n")


if __name__ == "__main__":
    for genre in GENRE:
        count_unique_songs(BASE_DATA_PATH, genre)
