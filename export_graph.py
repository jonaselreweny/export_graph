"""Export a Neo4j property graph to Parquet files (nodes + relationships) and schema."""

import argparse
import asyncio
import getpass
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_DATABASE = "neo4j"
DEFAULT_BATCH_SIZE = 50_000
QUEUE_MAXSIZE = 2  # double-buffer: at most 2 batches in flight


def _serialize_props(props: dict) -> str:
    """Convert a property dict to a JSON string for the parquet column."""
    return json.dumps(props, default=str)


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


async def produce_nodes(
    session, queue: asyncio.Queue, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Fetch nodes from Neo4j and put pa.Table batches onto the queue."""
    result = await session.run(
        "MATCH (n) "
        "RETURN elementId(n) AS id, labels(n) AS labels, "
        "properties(n) AS props, "
        "apoc.meta.cypher.types(properties(n)) AS propTypes"
    )
    ids: list[str] = []
    labels: list[str] = []
    properties: list[str] = []
    property_types: list[str] = []

    async for record in result:
        ids.append(record["id"])
        labels.append(json.dumps(record["labels"]))
        properties.append(_serialize_props(record["props"]))
        property_types.append(json.dumps(record["propTypes"]))

        if len(ids) >= batch_size:
            table = pa.table(
                {
                    "id": pa.array(ids, type=pa.string()),
                    "labels": pa.array(labels, type=pa.string()),
                    "properties": pa.array(properties, type=pa.string()),
                    "property_types": pa.array(property_types, type=pa.string()),
                }
            )
            await queue.put(table)
            ids.clear()
            labels.clear()
            properties.clear()
            property_types.clear()

    if ids:
        table = pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "labels": pa.array(labels, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
                "property_types": pa.array(property_types, type=pa.string()),
            }
        )
        await queue.put(table)

    await queue.put(None)  # sentinel to signal completion


async def produce_relationships(
    session, queue: asyncio.Queue, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Fetch relationships from Neo4j and put pa.Table batches onto the queue."""
    result = await session.run(
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

    async for record in result:
        out_ids.append(record["outId"])
        in_ids.append(record["inId"])
        types.append(record["type"])
        properties.append(_serialize_props(record["props"]))
        property_types.append(json.dumps(record["propTypes"]))

        if len(out_ids) >= batch_size:
            table = pa.table(
                {
                    "outId": pa.array(out_ids, type=pa.string()),
                    "inId": pa.array(in_ids, type=pa.string()),
                    "type": pa.array(types, type=pa.string()),
                    "properties": pa.array(properties, type=pa.string()),
                    "property_types": pa.array(property_types, type=pa.string()),
                }
            )
            await queue.put(table)
            out_ids.clear()
            in_ids.clear()
            types.clear()
            properties.clear()
            property_types.clear()

    if out_ids:
        table = pa.table(
            {
                "outId": pa.array(out_ids, type=pa.string()),
                "inId": pa.array(in_ids, type=pa.string()),
                "type": pa.array(types, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
                "property_types": pa.array(property_types, type=pa.string()),
            }
        )
        await queue.put(table)

    await queue.put(None)  # sentinel to signal completion


async def fetch_constraints(session) -> list[str]:
    """Return CREATE statements for all constraints."""
    result = await session.run(
        "SHOW CONSTRAINTS YIELD createStatement RETURN createStatement"
    )
    return [record["createStatement"] async for record in result]


async def fetch_indexes(session) -> list[str]:
    """Return CREATE statements for indexes not owned by a constraint and not LOOKUP indexes."""
    result = await session.run(
        "SHOW INDEXES YIELD type, owningConstraint, createStatement "
        "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' "
        "RETURN createStatement"
    )
    return [record["createStatement"] async for record in result]


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


async def consume_to_parquet(
    queue: asyncio.Queue,
    path: str,
    schema: pa.Schema,
    label: str,
) -> tuple[int, float]:
    """Read pa.Table batches from *queue* and write them to a Parquet file.

    The blocking ``write_table`` call is offloaded to a thread so the event
    loop stays responsive.  Returns ``(total_rows, elapsed_seconds)``.
    """
    loop = asyncio.get_running_loop()
    writer = pq.ParquetWriter(path, schema)
    total = 0
    start = time.perf_counter()
    batch_start = start

    try:
        while True:
            table = await queue.get()
            if table is None:
                break
            await loop.run_in_executor(None, writer.write_table, table)
            total += table.num_rows
            batch_elapsed = time.perf_counter() - batch_start
            batch_rate = table.num_rows / batch_elapsed if batch_elapsed > 0 else 0
            print(
                f"  {label}: wrote batch of {table.num_rows} "
                f"(total: {total}, {batch_rate:,.0f} {label}/s)"
            )
            batch_start = time.perf_counter()
    finally:
        await loop.run_in_executor(None, writer.close)

    elapsed = time.perf_counter() - start
    return total, elapsed


async def export_pipeline(
    driver,
    database: str,
    producer_fn,
    path: str,
    schema: pa.Schema,
    label: str,
    batch_size: int,
) -> tuple[int, float]:
    """Run an async producer/consumer pipeline for one entity type.

    Returns ``(total_rows, elapsed_seconds)``.
    """
    queue: asyncio.Queue[pa.Table | None] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    async with driver.session(database=database) as session:
        _, (total, elapsed) = await asyncio.gather(
            producer_fn(session, queue, batch_size),
            consume_to_parquet(queue, path, schema, label),
        )

    return total, elapsed


async def async_main() -> None:
    args = parse_args()
    config = resolve_config(args)
    batch_size: int = args.batch_size

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    nodes_path = os.path.join(output_dir, "nodes.parquet")
    rels_path = os.path.join(output_dir, "relationships.parquet")
    schema_path = os.path.join(output_dir, "schema.cypher")

    _check_output_files([nodes_path, rels_path, schema_path], args.overwrite)

    job_start = time.perf_counter()

    async with AsyncGraphDatabase.driver(
        config["uri"], auth=(config["username"], config["password"])
    ) as driver:
        await driver.verify_connectivity()
        print(
            f"Connected to Neo4j ({config['uri']}, db={config['database']}).  "
            f"Batch size: {batch_size}"
        )

        # --- Nodes + Relationships (concurrent pipelines) ---
        (total_nodes, nodes_elapsed), (total_rels, rels_elapsed) = await asyncio.gather(
            export_pipeline(
                driver, config["database"],
                produce_nodes, nodes_path, NODES_SCHEMA, "nodes", batch_size,
            ),
            export_pipeline(
                driver, config["database"],
                produce_relationships, rels_path, RELATIONSHIPS_SCHEMA, "rels", batch_size,
            ),
        )

        # --- Per-entity summaries ---
        nodes_rate = total_nodes / nodes_elapsed if nodes_elapsed > 0 else 0
        rels_rate = total_rels / rels_elapsed if rels_elapsed > 0 else 0
        print(f"Wrote {total_nodes:,} nodes → {nodes_path} "
              f"({nodes_rate:,.0f} nodes/s, {nodes_elapsed:.1f}s)")
        print(f"Wrote {total_rels:,} relationships → {rels_path} "
              f"({rels_rate:,.0f} rels/s, {rels_elapsed:.1f}s)")

        # --- Schema (small, sequential) ---
        async with driver.session(database=config["database"]) as session:
            constraint_stmts, index_stmts = await asyncio.gather(
                fetch_constraints(session),
                fetch_indexes(session),
            )

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

    # --- Overall job summary ---
    job_elapsed = time.perf_counter() - job_start
    print(
        f"\nExport complete: {total_nodes:,} nodes + {total_rels:,} relationships "
        f"in {job_elapsed:.1f}s"
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
