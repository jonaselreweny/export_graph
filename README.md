# Neo4j Graph Export and Import

Export a Neo4j property graph to Parquet files and import it back into Neo4j.

This project includes:

- `export_graph.py`: Exports nodes, relationships, and schema
- `import_graph.py`: Imports nodes, relationships, and schema
- `read_parquet.py`: Quick local inspection of exported Parquet files

## Requirements

- Python 3
- Neo4j instance
- APOC plugin enabled in Neo4j (used for metadata and dynamic create procedures)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in values:

```bash
cp .env.example .env
```

Environment variables:

- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD` (prompted if empty)
- `NEO4J_DATABASE` (default for export: `neo4j`)
- `OUTPUT_DIR` (default for export: current directory)
- `NEO4J_IMPORT_DATABASE` (required for import if not passed via CLI)
- `INPUT_DIR` (default for import: current directory)

CLI arguments override environment variables.

## Export Graph

Run export:

```bash
python export_graph.py
```

Useful flags:

```bash
python export_graph.py --output-dir output --batch-size 5000 --overwrite
```

Produced files:

- `nodes.parquet`
- `relationships.parquet`
- `schema.cypher`

## Import Graph

Run import:

```bash
python import_graph.py
```

Useful flags:

```bash
python import_graph.py --input-dir output --database neo4j --batch-size 5000
```

Import flow:

- Ensures target database exists (when supported)
- Applies schema statements from `schema.cypher`
- Imports nodes and relationships from Parquet
- Cleans temporary import properties

## Inspect Parquet Files

If `nodes.parquet` and `relationships.parquet` are in the current directory:

```bash
python read_parquet.py
```

## Notes

- Suitable for small to medium graphs. For larger graphs, consider alternative approaches.
- Export stores properties and property types as JSON strings in Parquet columns.
- Import reconstructs properties and applies type casting for Neo4j temporal/spatial types.
- Batch size can be tuned with `--batch-size`.
