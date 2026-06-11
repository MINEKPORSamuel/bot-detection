"""Raccourci pour lancer la phase 1.5 depuis la racine du projet."""

import argparse

from src.models.train_models import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5 - entrainement des modeles ML")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Relance les phases 1.3 et 1.4 avant l'entrainement.",
    )
    args = parser.parse_args()
    main(rebuild_features=args.rebuild_features)
