#!/usr/bin/env python
import sys
import warnings

from epolin.crew import Epolin

# CrewAI utilise pysbd pour segmenter du texte. On masque ce warning connu afin
# de garder la sortie console propre pour un utilisateur final.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Paramètres d'entrée par défaut injectés dans les tâches via les placeholders
# {linkedin_url} et {website_url}. Modifie cette structure selon ton scénario,
# ou remplace-la par des valeurs lues depuis `.env` / la CLI.
DEFAULT_INPUTS = {
    "info_prospect": "Arnaud Defaut LAS/SRA BL Procurement Director chez Thales ",
    "website_url": "https://www.thalesgroup.com/fr",
}


def run():
    """Point d'entrée principal utilisé par ``crewai run``."""
    # On copie le dictionnaire pour éviter qu'une tâche ne modifie l'état global
    # (CrewAI peut ajouter des champs comme le timestamp d'exécution).
    Epolin().crew().kickoff(inputs=DEFAULT_INPUTS.copy())


def train():
    """Boucle d'entraînement (replay automatisé) fournie par CrewAI."""
    Epolin().crew().train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs=DEFAULT_INPUTS.copy(),
    )


def replay():
    """Relance la crew depuis un identifiant de tâche enregistré."""
    Epolin().crew().replay(task_id=sys.argv[1])


def test():
    """Évalue la crew en boucle en utilisant un LLM d'auto-évaluation."""
    Epolin().crew().test(
        n_iterations=int(sys.argv[1]),
        eval_llm=sys.argv[2],
        inputs=DEFAULT_INPUTS.copy(),
    )
