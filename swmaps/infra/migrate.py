from swmaps.infra.db import run_migration

if __name__ == "__main__":
    run_migration("swmaps/infra/schema.sql")
    print("Migration complete.")
