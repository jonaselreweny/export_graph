"""Export a Neo4j property graph to Parquet files (nodes + relationships) and schema."""

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


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Connected to Neo4j.")

    with driver.session(database=DATABASE) as session:
        nodes_table = fetch_nodes(session)
        rels_table = fetch_relationships(session)
        constraint_stmts = fetch_constraints(session)
        index_stmts = fetch_indexes(session)

    pq.write_table(nodes_table, "nodes.parquet")
    print(f"Wrote {nodes_table.num_rows} nodes  → nodes.parquet")

    pq.write_table(rels_table, "relationships.parquet")
    print(f"Wrote {rels_table.num_rows} relationships → relationships.parquet")

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
