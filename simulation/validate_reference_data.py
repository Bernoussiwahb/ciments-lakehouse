from pathlib import Path

import pandas as pd


EQUIPMENT_PATH = Path("data/reference/equipment.csv")


def main() -> None:
    if not EQUIPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {EQUIPMENT_PATH}"
        )

    equipment = pd.read_csv(EQUIPMENT_PATH)

    required_columns = {
        "equipment_id",
        "equipment_name",
        "equipment_type",
        "site_id",
        "workshop_id",
        "nominal_capacity_tph",
        "installation_date",
        "status",
    }

    missing_columns = required_columns - set(equipment.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    if equipment["equipment_id"].duplicated().any():
        raise ValueError(
            "Certains identifiants d'équipements sont dupliqués."
        )

    if (equipment["nominal_capacity_tph"] <= 0).any():
        raise ValueError(
            "La capacité nominale doit être strictement positive."
        )

    print("Fichier equipment.csv valide.")
    print(f"Nombre d'équipements : {len(equipment)}")
    print()
    print(
        equipment[
            [
                "equipment_id",
                "equipment_type",
                "nominal_capacity_tph",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()