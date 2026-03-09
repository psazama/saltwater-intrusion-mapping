from swmaps.infra.db import get_connection, run_migration, seed_task_types

if __name__ == "__main__":
    run_migration("swmaps/infra/schema.sql")
    with get_connection() as conn:
        seed_task_types(conn)
    print("Migration complete.")
