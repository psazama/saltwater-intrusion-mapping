from pathlib import Path

from swmaps.infra.db import get_connection, run_migration, seed_task_types

INFRA_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    run_migration(str(INFRA_DIR / "schema.sql"))

    # Run numbered migrations in order
    for migration in sorted((INFRA_DIR / "migrations").glob("*.sql")):
        print(f"Running {migration.name}...")
        run_migration(str(migration))

    with get_connection() as conn:
        seed_task_types(conn)

    print("Migration complete.")
