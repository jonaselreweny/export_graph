"""Import a Neo4j property graph from Parquet files (nodes + relationships) and schema."""

import argparse
import getpass
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_BATCH_SIZE = 50_000

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
    the Python driver will handle the conversion.  For temporal and spatial types
    we keep the string representation and let the Cypher query cast them.
    """
    cast = {}
    for key, value in props.items():
        neo4j_type = prop_types.get(key, "STRING")
        if neo4j_type in ("INTEGER",) and isinstance(value, (int, float)):
            cast[key] = int(value)
        elif neo4j_type in ("FLOAT",) and isinstance(value, (int, float)):
            cast[key] = float(value)
        elif neo4j_type in ("BOOLEAN",) and isinstance(value, bool):
            cast[key] = value
        else:
            # Keep as-is; temporal/spatial/string values stay as strings.
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
    """Create the database if it does not already exist.

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
                session.run(f"CREATE DATABASE `{database}` WAIT") # parameterize to avoid injection risk
                print(f"Database '{database}' created.")
            else:
                print(f"Database '{database}' already exists.")
    except Exception as exc:
        print(
            f"Could not check/create database '{database}' "
            f"(may be Community Edition): {exc}"
        )
        print("Continuing — assuming the database exists.") # Improve by catching specific exceptions or checking server version


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
        rows: list[dict] = []

        for _, row in batch_df.iterrows():
            props = json.loads(row["properties"])
            prop_types = json.loads(row["property_types"]) if row.get("property_types") else {}
            cast_props = _cast_properties(props, prop_types)
            labels_list: list[str] = json.loads(row["labels"])
            rows.append({
                "id": row["id"],
                "labels": labels_list,
                "props": cast_props,
                "propTypes": prop_types,
            })

        # Collect all distinct property type maps to build targeted SET clauses.
        # For simplicity and to handle heterogeneous nodes, we use SET n += $props
        # and wrap temporal casts at the Cypher level per unique type signature.
        session.run(
            "UNWIND $rows AS row "
            "CALL { WITH row "
            "  WITH row, row.labels AS lbls "
            "  CALL apoc.create.node(lbls, row.props) YIELD node "
            "  SET node._import_id = row.id "
            "  RETURN node "
            "} IN TRANSACTIONS OF $batchSize ROWS RETURN count(*)",
            rows=rows,
            batchSize=batch_size,
        )

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
        rows: list[dict] = []

        for _, row in batch_df.iterrows():
            props = json.loads(row["properties"])
            prop_types = json.loads(row["property_types"]) if row.get("property_types") else {}
            cast_props = _cast_properties(props, prop_types)
            rows.append({
                "outId": row["outId"],
                "inId": row["inId"],
                "type": row["type"],
                "props": cast_props,
            })

        session.run(
            "UNWIND $rows AS row "
            "CALL { WITH row "
            "  MATCH (a {_import_id: row.outId}) "
            "  MATCH (b {_import_id: row.inId}) "
            "  CALL apoc.create.relationship(a, row.type, row.props, b) YIELD rel "
            "  RETURN rel "
            "} IN TRANSACTIONS OF $batchSize ROWS RETURN count(*)",
            rows=rows,
            batchSize=batch_size,
        )

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


def cleanup_import_ids(session) -> None:
    """Remove the temporary _import_id property from all nodes."""
    print("Cleaning up temporary _import_id properties …")
    session.run(
        "CALL apoc.periodic.iterate("
        "  'MATCH (n) WHERE n._import_id IS NOT NULL RETURN n', "
        "  'REMOVE n._import_id', "
        "  {batchSize: 10000, parallel: false}"
        ")"
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

    with driver.session(database=config["database"]) as session:
        # --- Schema ---
        schema_count = apply_schema(session, schema_path)
        print(f"Applied {schema_count} schema statement(s) from {schema_path}")

        # --- Temporary index for _import_id lookups ---
        print("Creating temporary index on _import_id …")
        try:
            session.run(
                "CREATE INDEX _import_id_index IF NOT EXISTS "
                "FOR (n:__ALL__) ON (n._import_id)"
            )
        except Exception:
            # Fallback: some Neo4j versions don't support __ALL__ label.
            # The import will still work, just slower on relationship matching.
            print("  Could not create universal _import_id index — continuing without it.")

        # --- Nodes ---
        print(f"Reading {nodes_path} …")
        nodes_df = pd.read_parquet(nodes_path)
        import_nodes(session, nodes_df, batch_size)

        # --- Relationships ---
        print(f"Reading {rels_path} …")
        rels_df = pd.read_parquet(rels_path)
        import_relationships(session, rels_df, batch_size)

        # --- Cleanup ---
        cleanup_import_ids(session)

        # Drop temporary index.
        try:
            session.run("DROP INDEX _import_id_index IF EXISTS")
            print("Dropped temporary _import_id index.")
        except Exception:
            pass

    driver.close()
    print("Import complete.")


if __name__ == "__main__":
    main()
