# Copilot Instructions

## Project Overview

This project exports graph data from a Neo4j database to Parquet files and provides utilities to read them back using Pandas.

## Tech Stack

- **Python 3** with a virtual environment (`venv/`)
- **Neo4j** for graph database connectivity
- **PyArrow** for Parquet file writing
- **Pandas** for reading Parquet files
- **Cypher** for database schema and queries

## Git Workflow Rules

1. **Always create a feature branch** before making changes:
   - Format: `feature/<short-description>` or `fix/<short-description>`
2. **Never commit directly to `main`.**
3. **Write meaningful commit messages** using conventional commits:
   - `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
4. **Make small, atomic commits** — one logical change per commit.
5. **Pull latest changes** from `main` before starting work.
6. **Rebase or merge `main`** into the feature branch before opening a PR.

## Coding Standards

- Use Python type hints where possible.
- Follow PEP 8 style guidelines.
- Keep dependencies listed in `requirements.txt`.
- Use the `venv/` virtual environment for all Python work.

## File Conventions

- **Cypher files** (`.cypher`) are used for Neo4j schema definitions (constraints, indexes).
- **Parquet files** are used as the export format for graph data.
- Do not commit Parquet output files or `venv/` to the repository.