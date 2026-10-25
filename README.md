# Sistema de Recomendación de Películas

Sistema de recomendación progresivo que cubre fundamentos, filtrado basado en contenido y técnicas de NLP, construido con un dataset de ~45,000 películas.

## Contenido del Proyecto

| Módulo | Archivo | Descripción |
|--------|---------|-------------|
| Utilidades | `utils.py` | Carga de datos, limpieza, funciones compartidas |
| Fundamentos | `01_fundamentos.py` | Teoría de recsys, EDA, recomendador por popularidad (WR/IMDB) |
| Content-Based | `02_content_based.py` | TF-IDF, metadata soup, similitud coseno, filtros avanzados |
| NLP | `03_nlp_tecnicas.py` | Tokenización, stemming, lematización, BoW vs TF-IDF, wordclouds |

## Estructura

```
sistema_recomendaciones/
├── data/
│   ├── movies_metadata.csv   # ~45K películas (título, sinopsis, géneros, ratings...)
│   └── keywords.csv          # ~46K entradas de keywords por película
├── output/                   # Gráficas y visualizaciones generadas
├── utils.py                  # Funciones utilitarias compartidas
├── 01_fundamentos.py         # Módulo 1: Fundamentos
├── 02_content_based.py       # Módulo 2: Content-Based Filtering
├── 03_nlp_tecnicas.py        # Módulo 3: Técnicas de NLP
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd sistema_recomendaciones

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos de NLTK (necesario para el módulo 03)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Ejecución

Cada módulo se ejecuta de forma independiente y genera salida en consola + gráficas en `output/`:

```bash
# Módulo 1: Fundamentos y análisis exploratorio
python 01_fundamentos.py

# Módulo 2: Sistema basado en contenido
python 02_content_based.py

# Módulo 3: Técnicas de NLP
python 03_nlp_tecnicas.py
```

## ¿Qué aprenderás?

### Módulo 1 — Fundamentos
- Tipos de sistemas de recomendación (simple, content-based, colaborativo, híbrido)
- Fórmula de Weighted Rating (promedio bayesiano de IMDB)
- Análisis exploratorio con matplotlib

### Módulo 2 — Content-Based Filtering
- **TF-IDF**: Cómo convertir texto en vectores numéricos ponderados
- **Similitud Coseno**: Medir la similitud entre documentos
- **Metadata Soup**: Combinar múltiples fuentes de texto para mejores recomendaciones
- Comparación TF-IDF vs CountVectorizer

### Módulo 3 — NLP
- Pipeline de preprocesamiento: tokenización → stopwords → stemming → lematización
- Bag of Words vs TF-IDF: diferencias teóricas y prácticas
- Extracción de n-gramas y keywords por género
- Nubes de palabras (wordclouds)
- Similitud coseno vs distancia euclidiana

## Tecnologías

- **Python 3.10+**
- **pandas** — Manipulación de datos
- **scikit-learn** — Vectorización (TF-IDF, CountVectorizer) y métricas de similitud
- **NLTK** — Tokenización, stemming, lematización, stopwords
- **matplotlib** — Visualizaciones
- **wordcloud** — Nubes de palabras

## Licencia

MIT

