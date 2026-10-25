"""
utils.py — Funciones utilitarias para el Sistema de Recomendación de Películas
===============================================================================

Módulo central que provee funciones reutilizables para:
- Carga y limpieza de datos
- Parseo de columnas JSON embebidas
- Cálculo de rating ponderado (fórmula IMDB)
- Creación de "sopa de metadatos" para content-based filtering
- Búsqueda fuzzy de títulos
"""

import os
import ast
import numpy as np
import pandas as pd
from difflib import get_close_matches


# =============================================================================
# Rutas de datos
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies_metadata.csv")
KEYWORDS_PATH = os.path.join(DATA_DIR, "keywords.csv")


# =============================================================================
# Funciones de parseo
# =============================================================================

def parse_json_column(value):
    """
    Parsea una cadena que representa una lista de diccionarios JSON embebida en CSV.

    Ejemplo de entrada:
        "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]"

    Retorna:
        ['Animation', 'Comedy']

    Si el valor no es parseable, retorna una lista vacía.
    """
    if isinstance(value, list):
        return [item["name"] for item in value if isinstance(item, dict) and "name" in item]
    if pd.isna(value) or not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]
    except (ValueError, SyntaxError):
        pass
    return []


def safe_int(value):
    """
    Convierte un valor a entero de forma segura.
    Retorna NaN si no es convertible.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# Carga y limpieza de datos
# =============================================================================

def load_and_clean_data(include_keywords=True):
    """
    Carga y limpia los datasets de películas y keywords.

    Pasos de limpieza:
    1. Eliminar filas con IDs no numéricos (datos corruptos en el CSV original)
    2. Convertir tipos: id→int, vote_average→float, vote_count→float
    3. Parsear columnas JSON: genres, production_companies
    4. Rellenar overviews vacíos con cadena vacía
    5. (Opcional) Merge con keywords por id

    Parámetros:
        include_keywords (bool): Si True, hace merge con keywords.csv

    Retorna:
        pd.DataFrame: DataFrame limpio listo para análisis
    """
    print("📂 Cargando movies_metadata.csv...")
    movies = pd.read_csv(MOVIES_PATH, low_memory=False)

    # --- Limpieza de IDs corruptos ---
    # Algunas filas tienen fechas u otros valores en la columna 'id'
    movies["id"] = movies["id"].apply(safe_int)
    movies = movies.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)

    # --- Conversión de tipos numéricos ---
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0.0)
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0.0)
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0.0)
    movies["budget"] = pd.to_numeric(movies["budget"], errors="coerce").fillna(0)
    movies["revenue"] = pd.to_numeric(movies["revenue"], errors="coerce").fillna(0)
    movies["runtime"] = pd.to_numeric(movies["runtime"], errors="coerce")

    # --- Parseo de columnas JSON ---
    print("🔧 Parseando columnas JSON (genres, production_companies)...")
    movies["genres_list"] = movies["genres"].apply(parse_json_column)
    movies["genres_str"] = movies["genres_list"].apply(lambda x: ", ".join(x) if x else "Sin género")

    movies["companies_list"] = movies["production_companies"].apply(parse_json_column)

    # --- Limpieza de texto ---
    movies["overview"] = movies["overview"].fillna("")
    movies["tagline"] = movies["tagline"].fillna("")
    movies["title"] = movies["title"].fillna(movies["original_title"])

    # --- Extraer año de lanzamiento ---
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["year"] = movies["release_date"].dt.year

    # --- Merge con keywords ---
    if include_keywords:
        print("🔗 Mergeando con keywords.csv...")
        keywords = pd.read_csv(KEYWORDS_PATH)
        keywords["id"] = keywords["id"].apply(safe_int)
        keywords = keywords.dropna(subset=["id"])
        keywords["id"] = keywords["id"].astype(int)
        keywords["keywords_list"] = keywords["keywords"].apply(parse_json_column)

        movies = movies.merge(keywords[["id", "keywords_list"]], on="id", how="left")
        movies["keywords_list"] = movies["keywords_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        movies["keywords_list"] = [[] for _ in range(len(movies))]

    # --- Eliminar duplicados por id ---
    movies = movies.drop_duplicates(subset=["id"], keep="first")

    # --- Resetear índice ---
    movies = movies.reset_index(drop=True)

    print(f"✅ Dataset listo: {len(movies)} películas cargadas")
    return movies


# =============================================================================
# Rating ponderado (Fórmula IMDB)
# =============================================================================

def weighted_rating(row, m, C):
    """
    Calcula el Weighted Rating usando la fórmula de IMDB:

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    Donde:
        v = número de votos de la película (vote_count)
        m = mínimo de votos requerido para entrar al ranking
        R = rating promedio de la película (vote_average)
        C = rating promedio global de todas las películas

    Esta fórmula es un promedio bayesiano que penaliza películas con
    pocos votos, acercándolas al promedio global.

    Parámetros:
        row: Fila del DataFrame con 'vote_count' y 'vote_average'
        m (float): Umbral mínimo de votos
        C (float): Rating promedio global

    Retorna:
        float: Rating ponderado
    """
    v = row["vote_count"]
    R = row["vote_average"]
    return (v / (v + m)) * R + (m / (v + m)) * C


# =============================================================================
# Sopa de metadatos (Metadata Soup)
# =============================================================================

def create_soup(row):
    """
    Crea una "sopa de metadatos" concatenando múltiples campos de texto
    (keywords, cast, director, genres, overview) para usar como entrada del
    vectorizador en content-based filtering.

    La sopa combina:
    - Keywords (repetidas para darles más peso)
    - Cast (actores principales, si existen)
    - Director (si existe)
    - Géneros
    - Overview (sinopsis)

    Los espacios dentro de nombres multi-palabra se eliminan para que
    'Science Fiction' → 'sciencefiction' sea tratado como un solo token.

    Parámetros:
        row: Fila del DataFrame

    Retorna:
        str: Cadena concatenada de metadatos en minúsculas
    """
    # Preparar keywords (sin espacios internos, para que sean tokens únicos)
    keywords = " ".join([kw.lower().replace(" ", "") for kw in row.get("keywords_list", [])])

    # Preparar cast y director si están disponibles en los metadatos
    cast = ""
    if "cast" in row and isinstance(row["cast"], list):
        cast = " ".join([str(c).lower().replace(" ", "") for c in row["cast"][:3]])
    elif "cast_list" in row and isinstance(row["cast_list"], list):
        cast = " ".join([str(c).lower().replace(" ", "") for c in row["cast_list"][:3]])

    director = ""
    if "director" in row and pd.notna(row["director"]):
        director = str(row["director"]).lower().replace(" ", "")

    # Preparar géneros (sin espacios internos)
    genres = " ".join([g.lower().replace(" ", "") for g in row.get("genres_list", [])])

    # Overview limpio
    overview = row["overview"].lower() if isinstance(row.get("overview", ""), str) else ""

    # Concatenar todo (keywords repetidos para mayor peso, cast, director, genres, overview)
    parts = [keywords, keywords, cast, director, genres, genres, overview]
    return " ".join([p for p in parts if p]).strip()


# =============================================================================
# Búsqueda fuzzy de títulos
# =============================================================================

def fuzzy_match_title(query, titles_series, n=5, cutoff=0.4):
    """
    Busca el título más cercano al query usando coincidencia difusa.

    Útil cuando el usuario escribe 'Toystory' en vez de 'Toy Story',
    o 'Dark Nite' en vez de 'The Dark Knight'.

    Parámetros:
        query (str): Título buscado por el usuario
        titles_series (pd.Series): Serie con todos los títulos disponibles
        n (int): Número máximo de coincidencias a retornar
        cutoff (float): Umbral mínimo de similitud (0.0 a 1.0)

    Retorna:
        list[str]: Lista de títulos que coinciden (ordenados por similitud)
    """
    titles_list = titles_series.dropna().unique().tolist()
    matches = get_close_matches(query, titles_list, n=n, cutoff=cutoff)
    return matches


# =============================================================================
# Punto de entrada para testing rápido
# =============================================================================

if __name__ == "__main__":
    # Test básico de carga
    df = load_and_clean_data()

    print(f"\n📊 Primeras 5 películas:")
    print(df[["title", "year", "genres_str", "vote_average", "vote_count"]].head().to_string())

    print(f"\n🔍 Test fuzzy matching:")
    matches = fuzzy_match_title("Toystory", df["title"])
    print(f"  'Toystory' → {matches}")

    matches = fuzzy_match_title("Dark Nite", df["title"])
    print(f"  'Dark Nite' → {matches}")
