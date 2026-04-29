"""Export a Neo4j property graph to Parquet files (nodes + relationships) and schema."""

import argparse
import getpass
import json
import os
import time
from collections.abc import Generator

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.spatial import Point, WGS84Point

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_DATABASE = "neo4j"
DEFAULT_BATCH_SIZE = 5_000


def _serialize_props(props: dict) -> str:
    """Convert a property dict to a JSON string for the parquet column."""

    def _to_json_value(value):
        if isinstance(value, Point):
            point_map: dict[str, float | int] = {"srid": value.srid}
            if isinstance(value, WGS84Point):
                point_map["longitude"] = float(value.longitude)
                point_map["latitude"] = float(value.latitude)
                if len(value) == 3:
                    point_map["height"] = float(value[2])
            else:
                point_map["x"] = float(value.x)
                point_map["y"] = float(value.y)
                if len(value) == 3:
                    point_map["z"] = float(value[2])
            return point_map

        if isinstance(value, dict):
            return {k: _to_json_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [_to_json_value(v) for v in value]

        return value

    return json.dumps(_to_json_value(props), default=str)


def _serialize_prop_types(props: dict, prop_types: dict) -> str:
    """Serialize property types using the same key order as properties when possible."""
    ordered_prop_types: dict = {}

    # First, follow the key order from the properties map for readability.
    for key in props:
        if key in prop_types:
            ordered_prop_types[key] = prop_types[key]

    # Then append any extra type keys that were not present in properties.
    for key, value in prop_types.items():
        if key not in ordered_prop_types:
            ordered_prop_types[key] = value

    return json.dumps(ordered_prop_types)


NODES_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("labels", pa.string()),
        pa.field("properties", pa.string()),
        pa.field("property_types", pa.string()),
    ]
)

RELATIONSHIPS_SCHEMA = pa.schema(
    [
        pa.field("outId", pa.string()),
        pa.field("inId", pa.string()),
        pa.field("type", pa.string()),
        pa.field("properties", pa.string()),
        pa.field("property_types", pa.string()),
    ]
)


def fetch_nodes(session, batch_size: int = DEFAULT_BATCH_SIZE) -> Generator[pa.Table, None, None]:
    """Yield batches of nodes as PyArrow tables with columns: id, labels, properties, property_types."""
    result = session.run(
        "MATCH (n) "
        "RETURN elementId(n) AS id, labels(n) AS labels, "
        "properties(n) AS props, "
        "apoc.meta.cypher.types(properties(n)) AS propTypes"
    )
    ids: list[str] = []
    labels: list[str] = []
    properties: list[str] = []
    property_types: list[str] = []

    for record in result:
        ids.append(record["id"])
        labels.append(json.dumps(record["labels"]))
        properties.append(_serialize_props(record["props"]))
        property_types.append(_serialize_prop_types(record["props"], record["propTypes"]))

        if len(ids) >= batch_size:
            yield pa.table(
                {
                    "id": pa.array(ids, type=pa.string()),
                    "labels": pa.array(labels, type=pa.string()),
                    "properties": pa.array(properties, type=pa.string()),
                    "property_types": pa.array(property_types, type=pa.string()),
                }
            )
            ids.clear()
            labels.clear()
            properties.clear()
            property_types.clear()

    if ids:
        yield pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "labels": pa.array(labels, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
                "property_types": pa.array(property_types, type=pa.string()),
            }
        )


def fetch_relationships(
    session, batch_size: int = DEFAULT_BATCH_SIZE
) -> Generator[pa.Table, None, None]:
    """Yield batches of relationships as PyArrow tables with columns: outId, inId, type, properties, property_types."""
    result = session.run(
        "MATCH (a)-[r]->(b) "
        "RETURN elementId(a) AS outId, elementId(b) AS inId, "
        "type(r) AS type, properties(r) AS props, "
        "apoc.meta.cypher.types(properties(r)) AS propTypes"
    )
    out_ids: list[str] = []
    in_ids: list[str] = []
    types: list[str] = []
    properties: list[str] = []
    property_types: list[str] = []

    for record in result:
        out_ids.append(record["outId"])
        in_ids.append(record["inId"])
        types.append(record["type"])
        properties.append(_serialize_props(record["props"]))
        property_types.append(_serialize_prop_types(record["props"], record["propTypes"]))

        if len(out_ids) >= batch_size:
            yield pa.table(
                {
                    "outId": pa.array(out_ids, type=pa.string()),
                    "inId": pa.array(in_ids, type=pa.string()),
                    "type": pa.array(types, type=pa.string()),
                    "properties": pa.array(properties, type=pa.string()),
                    "property_types": pa.array(property_types, type=pa.string()),
                }
            )
            out_ids.clear()
            in_ids.clear()
            types.clear()
            properties.clear()
            property_types.clear()

    if out_ids:
        yield pa.table(
            {
                "outId": pa.array(out_ids, type=pa.string()),
                "inId": pa.array(in_ids, type=pa.string()),
                "type": pa.array(types, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
                "property_types": pa.array(property_types, type=pa.string()),
            }
        )


def fetch_constraints(session) -> list[str]:
    """Return CREATE statements for all constraints."""
    result = session.run(
        "SHOW CONSTRAINTS YIELD createStatement RETURN createStatement"
    )
    return [record["createStatement"] for record in result]


def fetch_indexes(session) -> list[str]:
    """Return CREATE statements for indexes not owned by a constraint and not LOOKUP indexes."""
    result = session.run(
        "SHOW INDEXES YIELD type, owningConstraint, createStatement "
        "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' "
        "RETURN createStatement"
    )
    return [record["createStatement"] for record in result]


def _check_output_files(paths: list[str], overwrite: bool) -> None:
    """Exit with an error if any output file already exists and overwrite is False."""
    if overwrite:
        return
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise SystemExit(
            f"Output file(s) already exist: {', '.join(existing)}. "
            "Use --overwrite to replace them."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a Neo4j property graph to Parquet files and schema."
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
        help=f"Neo4j database (default: {DEFAULT_DATABASE}). Env: NEO4J_DATABASE",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for exported files (default: current directory). Env: OUTPUT_DIR",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of nodes/relationships to fetch per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. Without this flag, the script exits if files exist.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    """Resolve Neo4j connection config with precedence: CLI > env vars / .env > defaults."""
    load_dotenv()  # loads .env if present; does not override existing env vars

    uri = args.address or os.getenv("NEO4J_URI", DEFAULT_URI)
    username = args.username or os.getenv("NEO4J_USERNAME", DEFAULT_USERNAME)
    password = args.password or os.getenv("NEO4J_PASSWORD")
    database = args.database or os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE)

    if not password:
        password = getpass.getpass(prompt="Neo4j password: ")

    output_dir = args.output_dir or os.getenv("OUTPUT_DIR", ".")

    return {
        "uri": uri,
        "username": username,
        "password": password,
        "database": database,
        "output_dir": output_dir,
    }


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    batch_size: int = args.batch_size

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    nodes_path = os.path.join(output_dir, "nodes.parquet")
    rels_path = os.path.join(output_dir, "relationships.parquet")
    schema_path = os.path.join(output_dir, "schema.cypher")

    _check_output_files([nodes_path, rels_path, schema_path], args.overwrite)

    driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
    driver.verify_connectivity()
    print(f"Connected to Neo4j ({config['uri']}, db={config['database']}).  Batch size: {batch_size}")

    with driver.session(database=config["database"]) as session:
        # --- Nodes ---
        total_nodes = 0
        nodes_start = time.perf_counter()
        batch_start = nodes_start
        with pq.ParquetWriter(nodes_path, NODES_SCHEMA) as writer:
            for batch in fetch_nodes(session, batch_size):
                writer.write_table(batch)
                total_nodes += batch.num_rows
                batch_elapsed = time.perf_counter() - batch_start
                batch_rate = batch.num_rows / batch_elapsed if batch_elapsed > 0 else 0
                print(
                    f"  nodes: wrote batch of {batch.num_rows} "
                    f"(total: {total_nodes}, {batch_rate:,.0f} nodes/s)"
                )
                batch_start = time.perf_counter()
        nodes_elapsed = time.perf_counter() - nodes_start
        nodes_rate = total_nodes / nodes_elapsed if nodes_elapsed > 0 else 0
        print(f"Wrote {total_nodes} nodes → {nodes_path} ({nodes_rate:,.0f} nodes/s)")

        # --- Relationships ---
        total_rels = 0
        rels_start = time.perf_counter()
        batch_start = rels_start
        with pq.ParquetWriter(rels_path, RELATIONSHIPS_SCHEMA) as writer:
            for batch in fetch_relationships(session, batch_size):
                writer.write_table(batch)
                total_rels += batch.num_rows
                batch_elapsed = time.perf_counter() - batch_start
                batch_rate = batch.num_rows / batch_elapsed if batch_elapsed > 0 else 0
                print(
                    f"  rels: wrote batch of {batch.num_rows} "
                    f"(total: {total_rels}, {batch_rate:,.0f} rels/s)"
                )
                batch_start = time.perf_counter()
        rels_elapsed = time.perf_counter() - rels_start
        rels_rate = total_rels / rels_elapsed if rels_elapsed > 0 else 0
        print(f"Wrote {total_rels} relationships → {rels_path} ({rels_rate:,.0f} rels/s)")

        # --- Schema ---
        constraint_stmts = fetch_constraints(session)
        index_stmts = fetch_indexes(session)

    with open(schema_path, "w") as f:
        if constraint_stmts:
            f.write("// Constraints\n")
            for stmt in constraint_stmts:
                f.write(f"{stmt};\n")
            f.write("\n")
        if index_stmts:
            f.write("// Indexes\n")
            for stmt in index_stmts:
                f.write(f"{stmt};\n")
    print(
        f"Wrote {len(constraint_stmts)} constraint(s) and "
        f"{len(index_stmts)} index(es) → {schema_path}"
    )

    driver.close()


if __name__ == "__main__":
    main()
