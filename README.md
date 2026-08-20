# Q-Mind Backend

Motor de simulación de celdas solares en tándem (perovskita, CdSe, PbS) para el demo de
Q-Mind. Física en MindSpore + algoritmo genético + FastAPI.

## Arquitectura

```
modelarts_worker/
  physics/
    BrusEngine.py         CdSe — ecuación de Brus (radio del punto cuántico)
    PbSEngine.py           PbS — relación de Moreels (diámetro) + Varshni térmico
    PerovskiteEngine.py    Perovskita — ley de Vegard (composición x, fracción de Br)
    SolarPerformanceEvaluator.py   Beer-Lambert, PCE, V_oc, J_sc, CMI
  logic/
    GeneticSolarOptimizer.py       GA sobre un hipercubo unitario [0,1] por capa
    SolarOptimizationManager.py    orquesta engines + GA, calcula bandgaps/PHE
    DataAnalyzer.py                traduce requests <-> SolarOptimizationManager
    MaterialRegistry.py            catálogo estático de los 3 materiales del demo
api/
  simulation.py     POST /api/v1/simulate — endpoint público del demo, sin auth
  optimization.py   POST /optimization/run — endpoint legado autenticado (CSV completo)
  auth.py, material.py, labels.py   CRUD autenticado, no lo usa el demo
db/
  confirmed_materials.csv   constantes físicas validadas de CdSe y PbS
  materials.csv             catálogo legado (7 materiales, ruta autenticada)
  schemas/simulation.py     Pydantic request/response del endpoint del demo
tests/            174 tests — physics_tests, logic_tests, api_tests, db_tests
```

Los tres motores comparten la interfaz `construct(temperature, gene, wavelengths) ->
(absorption, e_g, v_oc)`. El GA nunca ve física: la población vive en `[0,1]` por capa, y
cada motor traduce su propio gen a su variable física (radio, diámetro o composición).

## Setup local (Linux)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si el Python del sistema no es 3.10, instalalo con `pyenv install 3.10` o el paquete
`python310` del AUR — es la versión con la que está probado (174/174 tests verdes).

MindSpore necesita la librería de sistema **`libgomp`** (OpenMP runtime). En Arch suele
venir con `gcc-libs` (casi siempre ya instalado); si falta: `sudo pacman -S gcc-libs`.

Crear `.env` en la raíz del repo:

```
DB_TYPE=sqlite
DB_NAME=qmind.db
SECRET_KEY=<generar con: python -c "import secrets; print(secrets.token_hex(32))">
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

`DB_TYPE` y `DB_NAME` ya tienen default `sqlite`/`qmind.db` en `db/config.py`, así que el
`.env` es opcional para desarrollo local — pero `SECRET_KEY` sí conviene ponerlo si vas a
tocar los endpoints de auth (el demo público no los usa).

## Correr

```bash
uvicorn main:app --reload
```

Docs interactivas en `http://127.0.0.1:8000/docs`.

## Tests

```bash
pytest -q
```

174 tests: física de los 3 motores (incluye los 16 vectores de prueba + invariantes de
`ESPECS.pdf` para perovskita, y los de PbS/Moreels), lógica del GA, endpoints, repos de DB.

## Probar el endpoint del demo directo

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"materials":["CdSe","PbS"],"temperature_c":25,"spectral_min_nm":300,"spectral_max_nm":1400}'
```

Los únicos 3 parámetros que expone el demo son `materials` (exactamente 2, de
`["Perovskita yoduro bromuro", "CdSe", "PbS"]`), `temperature_c` (0-40) y
`spectral_min_nm`/`spectral_max_nm` (300-1400). Todo lo demás del GA (población,
generaciones, kappa, etc.) está fijo en `api/simulation.py` y nunca se expone.

## Despliegue actual

- **Railway**, build con `Dockerfile` propio (no Nixpacks/Railpack — Railway no instala
  `libgomp1` solo, y hubo que forzar el Dockerfile explícito para controlarlo).
- URL pública: `https://q-mind-backend-production.up.railway.app`
- Variables de entorno configuradas en Railway: `DB_TYPE`, `DB_NAME`, `SECRET_KEY`,
  `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` (mismos valores que el `.env` local).
- **Importante**: el campo "Custom Start Command" en Railway Settings → Deploy debe estar
  **vacío** — si tiene algo escrito, pisa el `CMD` del Dockerfile y `$PORT` no se expande
  (rompe con `Error: Invalid value for '--port': '$PORT' is not a valid integer`).

## Limitación conocida: una simulación a la vez

El compilador de grafos de MindSpore no es thread-safe. Si dos requests llegan a la vez,
el segundo corrompe el estado del compilador (`RuntimeError: Illegal AnfNode`). Por eso
`api/simulation.py` serializa la ejecución con `asyncio.Semaphore(1)`
(`DEMO_CONCURRENCY_LIMIT`). Con varias personas escaneando el QR a la vez, se hacen cola
y esperan ~1.5-2s cada una — no se cae, solo se demora.

## Decisiones de física (contexto de `ESPECS.pdf`)

- **`p_sun` corregido**: el espectro AM1.5G de referencia tiene paso no uniforme; se
  integra con `trapz`, no `delta*sum` (el código viejo daba PCE inflado x2 — ~33% en vez
  de los ~15-30% reales).
- **PbS usa Moreels, no Brus**: `Eg(d) = 0.41 + 1/(0.0252*d² + 0.283*d)` para el tamaño,
  más el término térmico de Varshni de `confirmed_materials.csv` (Alpha_evK negativo =
  coeficiente térmico anómalo positivo, igual que la perovskita).
- **Perovskita**: implementación literal de `ESPECS.pdf` (Vegard + bowing, x acotado a
  [0, 0.20] por el efecto Hoke de segregación de haluros).
- **V_oc por material**: CdSe usa el modelo genérico `E_g - 0.4` (la spec no lo cubre),
  PbS usa `0.519*E_g - 0.0221`, perovskita usa `E_g - 0.57`.

## Pendiente / a decidir

- **Eficiencia percibida baja en algunas combinaciones**: CdSe+Perovskita da ~5% PCE
  (ambos son absorbedores de bandgap alto, no se complementan espectralmente — es física
  correcta, no un bug). Si se quiere subir el número sin falsear la física, la palanca
  disponible es aumentar el grosor de capa activa (fijo en 300nm hoy, en
  `SolarPerformanceEvaluator.thickness` y en `api/simulation.py`'s `LayerResult`) — **no
  se aplicó, es una decisión pendiente del equipo**.
- Considerar un subdominio propio para la API (`api.tudominio.com` vía CNAME en Railway)
  en vez de la URL cruda de Railway — es cosmético, no bloquea el demo.
