"""Filesystem reconciliation utility for the imagery catalog.

Checks every registered scene in the database against the local filesystem
and marks any records whose files no longer exist as ``"missing"``.

Intended to be run periodically or after bulk file deletions to keep the
catalog in sync with what is actually on disk.

Usage::

    python -m swmaps.infra.reconcile
"""

from pathlib import Path

from swmaps.infra.db import get_connection


def reconcile_filesystem() -> None:
    """Check all registered scenes and mark any with missing files.

    Queries all imagery records from the database, checks whether at least
    one of each record's ``file_locations`` paths exists on disk, and
    prompts the user to mark missing records as ``"missing"`` in the catalog.

    Args:
        None

    Returns:
        None: Updates are applied directly to the database after user
        confirmation.
    """
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
