from pathlib import Path

from swmaps.infra.db import get_connection


def reconcile_filesystem():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, scene_id, file_locations FROM imagery")
            rows = cursor.fetchall()

        missing = []
        for row in rows:
            # Check if any of the registered file paths still exist
            files_exist = any(Path(f).exists() for f in row["file_locations"])
            if not files_exist:
                missing.append((row["id"], row["scene_id"]))

        if not missing:
            print("All registered scenes have files on disk.")
            return

        print(f"Found {len(missing)} scenes with missing files:")
        for id_, scene_id in missing:
            print(f"    {scene_id}")

        confirm = input("Mark these entries as missing? (y/n): ")
        if confirm.lower() == "y":
            with conn.cursor() as cursor:
                for id_, scene_id in missing:
                    cursor.execute(
                        "UPDATE imagery SET status = 'missing' WHERE id = %s", (id_,)
                    )
            conn.commit()
            print(f"Marked {len(missing)} missing entries.")


if __name__ == "__main__":
    reconcile_filesystem()
