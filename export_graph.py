"""Export a Neo4j property graph to Parquet files (nodes + relationships) and schema."""

import argparse
import json
from collections.abc import Generator

import pyarrow as pa
import pyarrow.parquet as pq
from neo4j import GraphDatabase

URI = "bolt+ssc://localhost:7687"
AUTH = ("neo4j", "test1234")
DATABASE = "movies"  # set to None to use the default database


def _serialize_props(props: dict) -> str:
    """Convert a property dict to a JSON string for the parquet column."""
    return json.dumps(props, default=str)


NODES_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("labels", pa.string()),
        pa.field("properties", pa.string()),
    ]
)

RELATIONSHIPS_SCHEMA = pa.schema(
    [
        pa.field("outId", pa.string()),
        pa.field("inId", pa.string()),
        pa.field("type", pa.string()),
        pa.field("properties", pa.string()),
    ]
)


def fetch_nodes(session, batch_size: int = 1000) -> Generator[pa.Table, None, None]:
    """Yield batches of nodes as PyArrow tables with columns: id, labels, properties."""
    skip = 0
    while True:
        result = session.run(
            "MATCH (n) "
            "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
            "ORDER BY elementId(n) "
            "SKIP $skip LIMIT $limit",
            skip=skip,
            limit=batch_size,
        )
        ids: list[str] = []
        labels: list[str] = []
        properties: list[str] = []

        for record in result:
            ids.append(record["id"])
            labels.append(json.dumps(record["labels"]))
            properties.append(_serialize_props(record["props"]))

        if not ids:
            break

        yield pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "labels": pa.array(labels, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
            }
        )
        skip += batch_size


def fetch_relationships(
    session, batch_size: int = 1000
) -> Generator[pa.Table, None, None]:
    """Yield batches of relationships as PyArrow tables with columns: outId, inId, type, properties."""
    skip = 0
    while True:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(a) AS outId, elementId(b) AS inId, "
            "type(r) AS type, properties(r) AS props "
            "ORDER BY elementId(r) "
            "SKIP $skip LIMIT $limit",
            skip=skip,
            limit=batch_size,
        )
        out_ids: list[str] = []
        in_ids: list[str] = []
        types: list[str] = []
        properties: list[str] = []

        for record in result:
            out_ids.append(record["outId"])
            in_ids.append(record["inId"])
            types.append(record["type"])
            properties.append(_serialize_props(record["props"]))

        if not out_ids:
            break

        yield pa.table(
            {
                "outId": pa.array(out_ids, type=pa.string()),
                "inId": pa.array(in_ids, type=pa.string()),
                "type": pa.array(types, type=pa.string()),
                "properties": pa.array(properties, type=pa.string()),
            }
        )
        skip += batch_size


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a Neo4j property graph to Parquet files and schema."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of nodes/relationships to fetch per batch (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size: int = args.batch_size

    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print(f"Connected to Neo4j.  Batch size: {batch_size}")

    with driver.session(database=DATABASE) as session:
        # --- Nodes ---
        total_nodes = 0
        nodes_path = "nodes.parquet"
        with pq.ParquetWriter(nodes_path, NODES_SCHEMA) as writer:
            for batch in fetch_nodes(session, batch_size):
                writer.write_table(batch)
                total_nodes += batch.num_rows
                print(f"  nodes: wrote batch of {batch.num_rows} (total: {total_nodes})")
        print(f"Wrote {total_nodes} nodes → {nodes_path}")

        # --- Relationships ---
        total_rels = 0
        rels_path = "relationships.parquet"
        with pq.ParquetWriter(rels_path, RELATIONSHIPS_SCHEMA) as writer:
            for batch in fetch_relationships(session, batch_size):
                writer.write_table(batch)
                total_rels += batch.num_rows
                print(f"  rels: wrote batch of {batch.num_rows} (total: {total_rels})")
        print(f"Wrote {total_rels} relationships → {rels_path}")

        # --- Schema ---
        constraint_stmts = fetch_constraints(session)
        index_stmts = fetch_indexes(session)

    schema_path = "schema.cypher"
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
