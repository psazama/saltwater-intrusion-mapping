import glob
import os

from swmaps.infra.db import get_connection, run_migration, seed_task_types

if __name__ == "__main__":
    run_migration("swmaps/infra/schema.sql")

    # Run numbered migrations in order
    migrations = sorted(glob.glob("swmaps/infra/migrations/*.sql"))
    for migration in migrations:
        print(f"Running {os.path.basename(migration)}...")
        run_migration(migration)

    with get_connection() as conn:
        seed_task_types(conn)

    print("Migration complete.")
