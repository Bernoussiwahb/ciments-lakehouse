from pathlib import Path

import yaml


CONFIG_PATH = Path("config/simulation.yml")


def load_config(config_path: Path) -> dict:
    """Charge et vérifie le fichier de configuration YAML."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Le fichier de configuration est introuvable : {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Le fichier de configuration est vide.")

    required_sections = [
        "project",
        "simulation",
        "equipment",
        "data_quality",
        "anomalies",
        "output",
    ]

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    if missing_sections:
        raise ValueError(
            f"Sections manquantes : {', '.join(missing_sections)}"
        )

    return config


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)

        simulation = config["simulation"]
        equipment = config["equipment"]

        print("Configuration valide.")
        print(f"Projet : {config['project']['name']}")
        print(f"Début : {simulation['start_date']}")
        print(f"Fin : {simulation['end_date']}")
        print(f"Fréquence : {simulation['frequency']}")
        print(f"Nombre d'équipements : {len(equipment)}")

        print("\nÉquipements configurés :")

        for equipment_id, parameters in equipment.items():
            print(
                f"- {equipment_id} "
                f"({parameters['equipment_type']})"
            )

    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        print(f"Erreur de configuration : {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()