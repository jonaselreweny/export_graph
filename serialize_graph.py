"""Serialize a Neo4j property graph to Parquet files (nodes + relationships)."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from neo4j import GraphDatabase

URI = "bolt+ssc://localhost:7687"
AUTH = ("neo4j", "test1234")
DATABASE = "movies"  # set to None to use the default database


def _serialize_props(props: dict) -> str:
    """Convert a property dict to a JSON string for the parquet column."""
    return json.dumps(props, default=str)


def fetch_nodes(session) -> pa.Table:
    """Return all nodes as a PyArrow table with columns: id, labels, properties."""
    result = session.run(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
    )
    ids: list[str] = []
    labels: list[str] = []
    properties: list[str] = []

    for record in result:
        ids.append(record["id"])
        labels.append(json.dumps(record["labels"]))
        properties.append(_serialize_props(record["props"]))

    return pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "labels": pa.array(labels, type=pa.string()),
            "properties": pa.array(properties, type=pa.string()),
        }
    )


def fetch_relationships(session) -> pa.Table:
    """Return all relationships as a PyArrow table with columns: outId, inId, type, properties."""
    result = session.run(
        "MATCH (a)-[r]->(b) "
        "RETURN elementId(a) AS outId, elementId(b) AS inId, "
        "type(r) AS type, properties(r) AS props"
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

    return pa.table(
        {
            "outId": pa.array(out_ids, type=pa.string()),
            "inId": pa.array(in_ids, type=pa.string()),
            "type": pa.array(types, type=pa.string()),
            "properties": pa.array(properties, type=pa.string()),
        }
    )


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Connected to Neo4j.")

    with driver.session(database=DATABASE) as session:
        nodes_table = fetch_nodes(session)
        rels_table = fetch_relationships(session)

    pq.write_table(nodes_table, "nodes.parquet")
    print(f"Wrote {nodes_table.num_rows} nodes  → nodes.parquet")

    pq.write_table(rels_table, "relationships.parquet")
    print(f"Wrote {rels_table.num_rows} relationships → relationships.parquet")

    driver.close()


if __name__ == "__main__":
    main()
