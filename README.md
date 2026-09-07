# LoveDA: segmentación semántica de cobertura terrestre

Proyecto universitario de **Visión por Computador**, Tema 3 — LoveDA: Land-Cover Semantic Segmentation.

**Integrantes:** Alvaro Toro Herrera y Jorge Butto Cumsille.

## Estado

Este repositorio contiene la parte de código y análisis de la **Entrega 1**: obtención/localización reproducible, inspección real, lectura correcta de imágenes y máscaras, control de calidad, análisis exploratorio, detección de duplicados exactos, figuras y diseño de experimentos futuros.

Todavía **no se selecciona ni entrena ningún modelo**. No se implementan arquitecturas, pérdidas, optimizadores, augmentations, predicciones ni la innovación multiescala.

## Dataset y fuente oficial

LoveDA es un dataset de segmentación semántica de cobertura terrestre en imágenes RGB aéreas. La fuente prioritaria es [Zenodo, DOI 10.5281/zenodo.5706578](https://doi.org/10.5281/zenodo.5706578), enlazada por el [repositorio oficial Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA). La publicación asociada es *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation* (NeurIPS Datasets and Benchmarks 2021).

Como contexto, la documentación oficial informa 5.987 imágenes de aproximadamente 0,3 m de resolución espacial, procedentes de Nanjing, Changzhou y Wuhan, y organizadas en los dominios Urban y Rural. Destaca como desafíos los objetos multiescala, los fondos complejos y las distribuciones de clases diferentes. Esas afirmaciones describen el dataset publicado; el notebook obtiene por ejecución todo valor que puede comprobarse desde los archivos locales.

El registro oficial distribuye `Train.zip`, `Val.zip`, `Test.zip` y `Datasheet.pdf`, e informa para cada archivo su tamaño y checksum MD5. El código usa esas propiedades para validación. Las cifras publicadas sobre cantidad, resolución y dominios se usan únicamente como sanity checks: todos los resultados de `outputs/` se calculan desde los archivos locales.

Etiquetas originales:

| Valor | Significado |
|---:|---|
| 0 | no-data / ignore |
| 1 | background |
| 2 | building |
| 3 | road |
| 4 | water |
| 5 | barren |
| 6 | forest |
| 7 | agriculture |

El valor 0 permanece separado, no se remapea, no cuenta como clase válida y deberá excluirse de las métricas futuras.

## Estructura del repositorio

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── loveda_eda.py
├── 01_dataset_exploration.ipynb
├── data/                         # local, ignorado por Git
│   ├── archives/                 # ZIP oficiales opcionales
│   └── LoveDA/                   # dataset extraído por defecto
└── outputs/
    ├── figures/
    ├── dataset_index.csv
    ├── dataset_issues.csv
    ├── dataset_summary.csv
    ├── class_pixel_counts.csv
    ├── class_image_presence.csv
    ├── split_counts.csv
    ├── domain_counts.csv
    ├── duplicates.csv
    ├── class_distribution_by_domain.csv
    └── class_distribution_by_split.csv
```

`loveda_eda.py` contiene funciones modulares para que el notebook se mantenga legible. El notebook es la entrada principal y documenta las decisiones metodológicas.

## Instalación

Se recomienda Python 3.10 o posterior. No se necesita PyTorch, TensorFlow ni otro framework de deep learning.

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Descargar o localizar LoveDA

### Opción A: descarga oficial automatizada

La descarga completa ocupa aproximadamente 9,6 GB comprimidos y requiere espacio adicional para la extracción. Para hacer explícita esa transferencia, la celda de descarga solo se activa mediante una variable de entorno.

Windows PowerShell:

```powershell
$env:LOVEDA_DOWNLOAD="1"
jupyter notebook 01_dataset_exploration.ipynb
```

Linux o macOS:

```bash
LOVEDA_DOWNLOAD=1 jupyter notebook 01_dataset_exploration.ipynb
```

La función `download_official_loveda` descarga directamente desde la API de Zenodo, admite reanudación, evita volver a descargar archivos ya válidos, compara tamaño y MD5, y extrae en `data/LoveDA` sin borrar los ZIP.

### Opción B: descarga manual o dataset ya existente

1. Descargar los archivos desde [el DOI oficial](https://doi.org/10.5281/zenodo.5706578).
2. Extraerlos sin cambiar sus nombres ni contenido.
3. Indicar la carpeta contenedora mediante `LOVEDA_ROOT`.

PowerShell:

```powershell
$env:LOVEDA_ROOT="D:\datasets\LoveDA"
jupyter notebook 01_dataset_exploration.ipynb
```

Bash:

```bash
LOVEDA_ROOT="/ruta/a/LoveDA" jupyter notebook 01_dataset_exploration.ipynb
```

Si no se define la variable, se usa `data/LoveDA`. No hay rutas personales hardcodeadas. Si falta el dataset, el notebook termina sin inventar resultados y muestra instrucciones claras.

El dataset y los ZIP están excluidos por `.gitignore`; **no deben subirse a Git**. La licencia declarada por los autores permite uso académico y prohíbe uso comercial, además de exigir respetar los términos de Google Earth.

## Ejecutar el análisis

Abrir y ejecutar todas las celdas en orden:

```powershell
jupyter notebook 01_dataset_exploration.ipynb
```

Para una ejecución limpia y no interactiva:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_dataset_exploration.ipynb
```

El análisis recorre archivo por archivo. No carga el dataset completo en memoria, no guarda imágenes en DataFrames y solo abre las imágenes seleccionadas para las figuras. Los caminos de `dataset_index.csv` son relativos a `DATASET_ROOT`, lo que conserva portabilidad.

## Archivos generados

- `dataset_index.csv`: una fila por imagen, emparejamiento, metadata, conteos de píxeles, hashes y flags.
- `dataset_issues.csv`: hallazgos reales de control de calidad; el notebook también muestra categorías con cero casos.
- `dataset_summary.csv`: métricas generales en formato clave/valor.
- `class_pixel_counts.csv`: distribución de píxeles 1–7 y cobertura cuando la clase aparece.
- `class_image_presence.csv`: presencia de cada clase en imágenes con máscara legible.
- `split_counts.csv` y `domain_counts.csv`: cantidades, máscaras e ignore por agrupación.
- `class_distribution_by_split.csv` y `class_distribution_by_domain.csv`: distribución y presencia de etiquetas por grupo, con denominadores explícitos.
- `duplicates.csv`: grupos de imágenes idénticas byte a byte por SHA-256.
- `figures/*.png`: figuras reproducibles a 200 dpi. Una figura no aplicable no se crea artificialmente y el notebook explica la omisión.

SHA-256 solo comprueba igualdad exacta de archivos; no detecta similitud visual, tiles vecinos, solapamiento ni proximidad geográfica.

La distribución descargada no contiene archivos auxiliares de coordenadas o georreferenciación y los stems observados son numéricos. Por ello no se construyen grupos espaciales: hacerlo a partir de estos archivos exigiría inventar metadata. Esta es una limitación explícita para evaluar posible leakage entre tiles próximos.

## Diseño de las siguientes entregas

El notebook mantiene Test oficial intacto para evaluación externa, usa Train oficial como desarrollo y propone dividir posteriormente Train en 80% de entrenamiento interno y 20% de validación interna. Val oficial queda separado para la evaluación local final. El futuro split usará `RANDOM_SEED` e índices sin mover archivos, preservará aproximadamente Urban/Rural, considerará la distribución multietiqueta y mantendrá unidos los duplicados exactos.

La innovación asignada es **segmentación multiescala**: en el futuro, el mismo modelo se evaluará a varias escalas, los mapas volverán a una resolución común y se combinarán por promedio o votación. En Entrega 1 solo se documenta.

Las métricas futuras principales serán IoU por clase, mIoU y Dice/F1 por clase con promedio macro, siempre ignorando ground truth 0. No se calculan métricas de modelo en esta entrega porque todavía no existen predicciones; Test oficial tampoco permite calcular localmente mIoU o Dice porque no posee ground truth distribuido localmente.

## Resultados verificados de esta ejecución

Ejecución limpia realizada sobre los tres ZIP oficiales, todos con tamaño y MD5 coincidentes con Zenodo. Los CSV bajo `outputs/` son la fuente reproducible de estas cifras:

- 5.987 imágenes RGB PNG de 1024×1024 y tres canales.
- 4.191 máscaras PNG en modo `L`; valores observados: 0–7; todos los tamaños emparejados coinciden.
- Train: 2.522 imágenes/máscaras (Rural 1.366, Urban 1.156).
- Val: 1.669 imágenes/máscaras (Rural 992, Urban 677).
- Test: 1.796 imágenes (Rural 976, Urban 820) y ninguna máscara local.
- Dominios globales: Rural 3.334 imágenes y Urban 2.653.
- 145.409.132 píxeles ignore=0, equivalentes al 3,308827 % de los píxeles etiquetados.
- Ninguna imagen o máscara corrupta, dimensión/canal inesperado, etiqueta fuera de 0–7, máscara huérfana, mismatch de tamaño o máscara all-ignore.
- Test sin ground truth local: 1.796 imágenes (ausencia esperada, no es un problema de integridad).
- `dataset_issues.csv` registra cero problemas reales de integridad.
- Cero grupos de imágenes exactamente duplicadas por SHA-256 y cero duplicados exactos entre splits.
- Las diez figuras previstas se generaron a 200 dpi.

Distribución sobre los **píxeles válidos** (0 excluido):

| Clase | Píxeles | Porcentaje válido | Imágenes donde aparece |
|---|---:|---:|---:|
| background | 1.525.959.012 | 35,911907 % | 4.111 (98,091148 %) |
| building | 404.299.316 | 9,514777 % | 3.110 (74,206633 %) |
| road | 213.710.339 | 5,029457 % | 3.154 (75,256502 %) |
| water | 362.297.559 | 8,526308 % | 3.024 (72,154617 %) |
| barren | 207.131.990 | 4,874643 % | 1.819 (43,402529 %) |
| forest | 534.690.631 | 12,583405 % | 3.052 (72,822715 %) |
| agriculture | 1.001.084.037 | 23,559504 % | 2.473 (59,007397 %) |

En las 4.191 imágenes etiquetadas aparecen en promedio 4,949415 clases válidas distintas (mínimo 1, mediana 5, máximo 7). El análisis por dominio muestra diferencias descriptivas claras: building y road ocupan una proporción válida mayor en Urban (17,960212 % y 8,715302 %) que en Rural (3,088588 % y 2,224873 %), mientras agriculture y forest son mayores en Rural (33,341282 % y 15,298224 %) que en Urban (10,704081 % y 9,015532 %). Estas cifras no implican causalidad ni eliminan el riesgo de dependencia espacial no detectable sin coordenadas.
