"""Import a Neo4j property graph from Parquet files (nodes + relationships) and schema."""

import argparse
import getpass
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.spatial import CartesianPoint, WGS84Point

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_BATCH_SIZE = 5_000
TEMP_IMPORT_LABEL = "__ImportNode"

# Map Neo4j type names (from apoc.meta.cypher.types) to Cypher cast expressions.
# Keys that are not found here are treated as STRING (no cast needed).
NEO4J_TYPE_CAST: dict[str, str] = {
    "INTEGER": "toInteger({val})",
    "FLOAT": "toFloat({val})",
    "BOOLEAN": "toBoolean({val})",
    "DATE": "date({val})",
    "DATE_TIME": "datetime({val})",
    "LOCAL_DATE_TIME": "localdatetime({val})",
    "LOCAL_TIME": "localtime({val})",
    "TIME": "time({val})",
    "DURATION": "duration({val})",
    "POINT": "point({val})",
}


def _cast_properties(props: dict, prop_types: dict) -> dict:
    """Return a new dict with values cast to their Neo4j types where possible.

    For types that survive JSON round-tripping natively (INTEGER, FLOAT, BOOLEAN)
    the Python driver will handle the conversion.

    POINT values are converted to Neo4j spatial point objects so they are stored
    as POINT in the graph instead of plain lists.
    """

    def _to_point(value):
        if isinstance(value, dict):
            srid = value.get("srid")

            if srid in (4326, 4979):
                if "longitude" in value and "latitude" in value:
                    if "height" in value:
                        return WGS84Point((
                            float(value["longitude"]),
                            float(value["latitude"]),
                            float(value["height"]),
                        ))
                    return WGS84Point((
                        float(value["longitude"]),
                        float(value["latitude"]),
                    ))

            if srid in (7203, 9157):
                if "x" in value and "y" in value:
                    if "z" in value:
                        return CartesianPoint((
                            float(value["x"]),
                            float(value["y"]),
                            float(value["z"]),
                        ))
                    return CartesianPoint((
                        float(value["x"]),
                        float(value["y"]),
                    ))

            if "longitude" in value and "latitude" in value:
                if "height" in value:
                    return WGS84Point((
                        float(value["longitude"]),
                        float(value["latitude"]),
                        float(value["height"]),
                    ))
                return WGS84Point((
                    float(value["longitude"]),
                    float(value["latitude"]),
                ))
            if "x" in value and "y" in value:
                if "z" in value:
                    return CartesianPoint((
                        float(value["x"]),
                        float(value["y"]),
                        float(value["z"]),
                    ))
                return CartesianPoint((
                    float(value["x"]),
                    float(value["y"]),
                ))

        if isinstance(value, (list, tuple)) and len(value) in (2, 3):
            return CartesianPoint(tuple(float(v) for v in value))

        return value

    cast = {}
    for key, value in props.items():
        neo4j_type = prop_types.get(key, "STRING")
        if neo4j_type in ("INTEGER",) and isinstance(value, (int, float)):
            cast[key] = int(value)
        elif neo4j_type in ("FLOAT",) and isinstance(value, (int, float)):
            cast[key] = float(value)
        elif neo4j_type in ("BOOLEAN",) and isinstance(value, bool):
            cast[key] = value
        elif neo4j_type == "POINT":
            cast[key] = _to_point(value)
        else:
            # Keep as-is for non-primitive cast types.
            cast[key] = value
    return cast


def _build_set_clause(prop_types: dict) -> str:
    """Build a Cypher SET clause that casts temporal/spatial properties.

    For simple types (STRING, INTEGER, FLOAT, BOOLEAN) we pass them as
    parameters and let the driver handle the casting.
    For temporal/spatial types we wrap the parameter in the appropriate
    Cypher function.
    """
    parts: list[str] = []
    for key, neo4j_type in prop_types.items():
        template = NEO4J_TYPE_CAST.get(neo4j_type)
        if template:
            cast_expr = template.format(val=f"row.props.`{key}`")
            parts.append(f"n.`{key}` = {cast_expr}")
        else:
            parts.append(f"n.`{key}` = row.props.`{key}`")
    return ", ".join(parts)


def _build_rel_set_clause(prop_types: dict) -> str:
    """Build a Cypher SET clause for relationship properties with type casting."""
    parts: list[str] = []
    for key, neo4j_type in prop_types.items():
        template = NEO4J_TYPE_CAST.get(neo4j_type)
        if template:
            cast_expr = template.format(val=f"row.props.`{key}`")
            parts.append(f"r.`{key}` = {cast_expr}")
        else:
            parts.append(f"r.`{key}` = row.props.`{key}`")
    return ", ".join(parts)


def ensure_database(driver, database: str) -> None:
    """Create the database if it does not already exist. Exit if it already exists.

    On Community Edition (no multi-database support) this is a no-op.
    """
    try:
        with driver.session(database="system") as session:
            result = session.run(
                "SHOW DATABASES YIELD name WHERE name = $name RETURN name",
                name=database,
            )
            if not result.single():
                print(f"Database '{database}' does not exist — creating …")
                session.run(f"CREATE DATABASE `{database}` WAIT")
                print(f"Database '{database}' created.")
            else:
                raise SystemExit(f"Database '{database}' already exists. Aborting to avoid overwriting data.")
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"Could not check/create database '{database}' "
            f"(may be Community Edition): {exc}"
        )
        print("Continuing — assuming the database does not exist.")


def apply_schema(session, schema_path: str) -> int:
    """Execute each statement from a .cypher schema file. Returns the count of statements run."""
    if not os.path.exists(schema_path):
        print(f"Schema file '{schema_path}' not found — skipping.")
        return 0

    with open(schema_path) as f:
        content = f.read()

    count = 0
    for line in content.splitlines():
        stmt = line.strip()
        # Skip blank lines and comments.
        if not stmt or stmt.startswith("//"):
            continue
        # Remove trailing semicolon for session.run().
        if stmt.endswith(";"):
            stmt = stmt[:-1].strip()
        if stmt:
            try:
                session.run(stmt)
                count += 1
            except Exception as exc:
                # Constraints / indexes may already exist.
                print(f"  schema: skipped ({exc})")
    return count


def import_nodes(session, df: pd.DataFrame, batch_size: int) -> int:
    """Import nodes from a DataFrame in batches. Returns total rows imported."""
    total = 0
    start = time.perf_counter()
    batch_start = start

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        props_series = batch_df["properties"].map(json.loads)
        types_series = batch_df["property_types"].map(
            lambda v: json.loads(v) if v else {}
        )
        labels_series = batch_df["labels"].map(json.loads)
        rows = [
            {
                "labels": labels,
                "props": {**_cast_properties(props, prop_types), "_import_id": row_id},
            }
            for row_id, labels, props, prop_types in zip(
                batch_df["id"], labels_series, props_series, types_series
            )
        ]
        session.run(
            "UNWIND $rows AS row "
            f"CREATE (node:{TEMP_IMPORT_LABEL}:$(row.labels)) "
            "   SET node = row.props ",
            rows=rows,
        ).consume()

        batch_count = len(batch_df)
        total += batch_count
        batch_elapsed = time.perf_counter() - batch_start
        batch_rate = batch_count / batch_elapsed if batch_elapsed > 0 else 0
        print(f"  nodes: imported batch of {batch_count} (total: {total}, {batch_rate:,.0f} nodes/s)")
        batch_start = time.perf_counter()

    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed > 0 else 0
    print(f"Imported {total} nodes ({rate:,.0f} nodes/s)")
    return total


def import_relationships(session, df: pd.DataFrame, batch_size: int) -> int:
    """Import relationships from a DataFrame in batches. Returns total rows imported."""
    total = 0
    start = time.perf_counter()
    batch_start = start

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        props_series = batch_df["properties"].map(json.loads)
        types_series = batch_df["property_types"].map(
            lambda v: json.loads(v) if v else {}
        )
        rows = [
            {
                "outId": out_id,
                "inId": in_id,
                "type": rel_type,
                "props": _cast_properties(props, prop_types),
            }
            for out_id, in_id, rel_type, props, prop_types in zip(
                batch_df["outId"], batch_df["inId"], batch_df["type"],
                props_series, types_series
            )
        ]
        session.run(
            "UNWIND $rows AS row "
            f"MATCH (a:{TEMP_IMPORT_LABEL} {{_import_id: row.outId}}) "
            f"MATCH (b:{TEMP_IMPORT_LABEL} {{_import_id: row.inId}}) "
            "CREATE (a)-[rel:$(row.type)]->(b) "
            "  SET rel = row.props ",
            rows=rows,
        ).consume()

        batch_count = len(batch_df)
        total += batch_count
        batch_elapsed = time.perf_counter() - batch_start
        batch_rate = batch_count / batch_elapsed if batch_elapsed > 0 else 0
        print(f"  rels: imported batch of {batch_count} (total: {total}, {batch_rate:,.0f} rels/s)")
        batch_start = time.perf_counter()

    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed > 0 else 0
    print(f"Imported {total} relationships ({rate:,.0f} rels/s)")
    return total


def cleanup_import_ids(session, batch_size: int) -> None:
    """Remove temporary import label and _import_id property from imported nodes."""
    print("Cleaning up temporary import label/properties …")
    session.run(
        f"MATCH (n:{TEMP_IMPORT_LABEL}) "
        "CALL (n) { "
        "  REMOVE n._import_id "
        f"  REMOVE n:{TEMP_IMPORT_LABEL} "
        "} IN TRANSACTIONS OF $batchSize ROWS RETURN count(*)",
        batchSize=batch_size
    )
    print("Cleanup complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import a Neo4j property graph from Parquet files and schema."
    )
    parser.add_argument(
        "-a", "--address",
        type=str,
        default=None,
        help=f"Neo4j URI (default: {DEFAULT_URI}). Env: NEO4J_URI",
    )
    parser.add_argument(
        "-u", "--username",
        type=str,
        default=None,
        help=f"Neo4j username (default: {DEFAULT_USERNAME}). Env: NEO4J_USERNAME",
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        default=None,
        help="Neo4j password. Env: NEO4J_PASSWORD. Prompted if not set.",
    )
    parser.add_argument(
        "-d", "--database",
        type=str,
        default=None,
        help="Target Neo4j database (required). Env: NEO4J_IMPORT_DATABASE. Prompted if not set.",
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=None,
        help="Input directory containing exported files (default: current directory). Env: INPUT_DIR",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of nodes/relationships to import per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    """Resolve Neo4j connection config with precedence: CLI > env vars / .env > defaults."""
    load_dotenv()

    uri = args.address or os.getenv("NEO4J_URI", DEFAULT_URI)
    username = args.username or os.getenv("NEO4J_USERNAME", DEFAULT_USERNAME)
    password = args.password or os.getenv("NEO4J_PASSWORD")
    database = args.database or os.getenv("NEO4J_IMPORT_DATABASE")
    input_dir = args.input_dir or os.getenv("INPUT_DIR", ".")

    if not password:
        password = getpass.getpass(prompt="Neo4j password: ")

    if not database:
        database = input("Target database name: ").strip()
    if not database:
        raise SystemExit("Database name is required.")

    return {
        "uri": uri,
        "username": username,
        "password": password,
        "database": database,
        "input_dir": input_dir,
    }


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    batch_size: int = args.batch_size
    input_dir: str = config["input_dir"]

    nodes_path = os.path.join(input_dir, "nodes.parquet")
    rels_path = os.path.join(input_dir, "relationships.parquet")
    schema_path = os.path.join(input_dir, "schema.cypher")

    # Validate input files exist.
    for path in [nodes_path, rels_path]:
        if not os.path.exists(path):
            raise SystemExit(f"Required input file not found: {path}")

    driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
    driver.verify_connectivity()
    print(f"Connected to Neo4j ({config['uri']}).  Batch size: {batch_size}")

    # --- Create database if needed ---
    ensure_database(driver, config["database"])

    try:
        with driver.session(database=config["database"]) as session:
            # --- Schema ---
            schema_count = apply_schema(session, schema_path)
            print(f"Applied {schema_count} schema statement(s) from {schema_path}")

            # --- Nodes ---
            print(f"Reading {nodes_path} …")
            nodes_df = pd.read_parquet(nodes_path)
            import_nodes(session, nodes_df, batch_size)

            # --- Temporary index for _import_id lookups (built after nodes exist) ---
            print("Creating temporary index on _import_id …")
            try:
                session.run(
                    "CREATE INDEX _import_id_index IF NOT EXISTS "
                    f"FOR (n:{TEMP_IMPORT_LABEL}) ON (n._import_id)"
                ).consume()
                # Wait for the index to come online before using it.
                session.run("CALL db.awaitIndexes(300)").consume()
                print("  Index online.")
            except Exception:
                print("  Could not create temporary _import_id index — continuing without it.")

            # --- Relationships ---
            print(f"Reading {rels_path} …")
            rels_df = pd.read_parquet(rels_path)
            import_relationships(session, rels_df, batch_size)

            # --- Cleanup ---
            cleanup_import_ids(session, batch_size)

            # Drop temporary index.
            try:
                session.run("DROP INDEX _import_id_index IF EXISTS").consume()
                print("Dropped temporary _import_id index.")
            except Exception:
                pass

    except Exception as exc:
        print(f"\nImport failed: {exc}")
        print(f"Dropping partially-created database '{config['database']}' to allow a clean re-run …")
        try:
            with driver.session(database="system") as sys_session:
                sys_session.run(
                    f"DROP DATABASE `{config['database']}` IF EXISTS"
                ).consume()
            print("Database dropped.")
        except Exception as drop_exc:
            print(f"  Could not drop database: {drop_exc}")
        driver.close()
        raise SystemExit(1) from exc

    driver.close()
    print("Import complete.")


if __name__ == "__main__":
    main()
