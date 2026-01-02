# src/data/schema.py

"""
Proje genelinde kullanacağımız standart maç tablosu şeması.
Tüm data pipeline ve modeller bu kolon isimlerine güvenecek.
"""

MATCH_COLUMNS = [
    "date",             # Maç tarihi (YYYY-MM-DD string)
    "tourney",          # Turnuva adı
    "surface",          # Zemin (Hard, Clay, Grass, Carpet, Indoor)
    "round",            # Tur (R32, QF, SF, F, vb.)

    "playerA",          # Player A adı
    "playerB",          # Player B adı
    "rankA",            # Player A ATP/WTA rank
    "rankB",            # Player B rank

    "oddsA",            # Player A için pre-match oran
    "oddsB",            # Player B için pre-match oran

    "score",            # "6-4 6-3" gibi skor string'i
    "comment",          # Ek bilgi (Completed, Retired, vb.)
    "source_file",      # Bu satır hangi ham dosyadan geldi
    "gender",           # 'M', 'F' veya 'U' (unknown)

    "winner",           # 'A' veya 'B'

    "playerA_norm",     # normalize edilmiş isim (lowercase, trim)
    "playerB_norm",     # normalize edilmiş isim

    "pA_implied_fair",  # odds'tan gelen adil olasılık
    "pB_implied_fair",  # odds'tan gelen adil olasılık
]
