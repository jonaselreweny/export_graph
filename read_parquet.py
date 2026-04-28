import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Read exported Parquet files and print a preview."
	)
	parser.add_argument(
		"--nodes-path",
		type=str,
		default="output/nodes.parquet",
		help="Path to nodes parquet file (default: output/nodes.parquet).",
	)
	parser.add_argument(
		"--relationships-path",
		type=str,
		default="output/relationships.parquet",
		help="Path to relationships parquet file (default: output/relationships.parquet).",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=5,
		help="Number of rows to display for each DataFrame preview (default: 5).",
	)
	args = parser.parse_args()
	if args.rows < 1:
		raise SystemExit("--rows must be at least 1.")
	return args


def main() -> None:
	args = parse_args()

	# Read the parquet files
	nodes_df = pd.read_parquet(args.nodes_path)
	relationships_df = pd.read_parquet(args.relationships_path)

	# Show full-width DataFrame output without horizontal truncation.
	pd.set_option("display.max_columns", None)
	pd.set_option("display.width", None)
	pd.set_option("display.max_colwidth", None)
	pd.set_option("display.expand_frame_repr", False)

	# Display basic info about the dataframes
	print("Nodes DataFrame:")
	print(f"Shape: {nodes_df.shape}")
	print(f"Columns: {list(nodes_df.columns)}")
	print(f"\nFirst {args.rows} rows:")
	print(nodes_df.head(args.rows).to_string())

	print("\n" + "="*50 + "\n")

	print("Relationships DataFrame:")
	print(f"Shape: {relationships_df.shape}")
	print(f"Columns: {list(relationships_df.columns)}")
	print(f"\nFirst {args.rows} rows:")
	print(relationships_df.head(args.rows).to_string())


if __name__ == "__main__":
	main()