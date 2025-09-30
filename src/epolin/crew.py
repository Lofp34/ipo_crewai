import os
from crewai import LLM
from typing import Any, Dict, List
from crewai_tools import (ScrapeWebsiteTool, SerperDevTool, FileReadTool)
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from mcp import StdioServerParameters
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

def _resolve_config(config: Dict[str, Any], key: str, section: str) -> Dict[str, Any]:
    """Return the configuration block associated with ``key``."""
    if key in config and isinstance(config[key], dict):
        return config[key]
    nested = config.get(section)
    if isinstance(nested, dict) and key in nested and isinstance(nested[key], dict):
        return nested[key]
    raise KeyError(f"Configuration '{key}' non trouvée dans '{section}'.")

@CrewBase
class Epolin:
    """Définition de la crew Epolin alignée sur les fichiers YAML."""

    agents: List[BaseAgent]
    tasks: List[Task]

        # Ajoutez votre knowledge source ici
    knowledge_source_1 = TextFileKnowledgeSource(file_paths=["info_IPO.md"])

    # Paramètres de connexion au serveur MCP Horizon Datawave
    mcp_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@horizondatawave/mcp"],
        env={
            "HDW_ACCESS_TOKEN": os.getenv("HDW_ACCESS_TOKEN"),
            "HDW_ACCOUNT_ID": os.getenv("HDW_ACCOUNT_ID"),
        },
    )

    def get_mcp_tools(self):
        return [MCPServerAdapter(self.mcp_server_params)]

    def _agent_config(self, key: str) -> Dict[str, Any]:
        """Lire la configuration de l'agent correspondant dans le YAML."""
        return _resolve_config(self.agents_config, key, "agents")

    def _task_config(self, key: str) -> Dict[str, Any]:
        """Lire la configuration de la tâche correspondante dans le YAML."""
        return _resolve_config(self.tasks_config, key, "tasks")

    def _agent_instance(self, key: str) -> Agent:
        """Instancier dynamiquement l'agent nommé dans le YAML."""
        agent_builder = getattr(self, key, None)
        if callable(agent_builder):
            return agent_builder()
        raise KeyError(f"Agent '{key}' introuvable dans la crew.")

    def _build_task(self, key: str) -> Task:
        """Construire un objet Task complet à partir du YAML."""
        config = dict(self._task_config(key))
        agent_key = config.pop("agent", None)
        agent_instance = self._agent_instance(agent_key) if agent_key else None

        # Extraire les paramètres du config pour les passer directement
        description = config.get("description", "")
        expected_output = config.get("expected_output", "")
        output_file = config.get("output_file", None)
        
        # Ne pas traiter le context ici - il sera résolu automatiquement par CrewAI
        context_names = config.get("context", [])

        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent_instance,
            output_file=output_file,
            # Ne pas passer context ici - CrewAI le résoudra automatiquement
        )

    @agent
    def profil_social_agent(self) -> Agent:
        """Collecte toutes les données publiques du profil LinkedIn via les outils MCP."""
        return Agent(
            config=self._agent_config("profil_social_agent"),
            verbose=True,
            # Ne plus inclure SerperDevTool ; on se contente des outils MCP
            tools=self.get_mcp_tools() if hasattr(self, 'get_mcp_tools') else [],
            llm=LLM(model="gpt-4o-mini"),
        )

    @agent
    def web_company_agent(self) -> Agent:
        """Scraping du site de l'entreprise et consolidation légale."""
        return Agent(
            config=self._agent_config("web_company_agent"),
            verbose=True,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            llm=LLM(model="gpt-4o-mini"),
        )

    @agent
    def sector_watch_agent(self) -> Agent:
        """Analyse des tendances et actualités du secteur cible."""
        return Agent(
            config=self._agent_config("sector_watch_agent"),
            verbose=True,
            tools=[SerperDevTool()],
            llm=LLM(model="gpt-4o-mini"),
        )

    @agent
    def outreach_strategy_agent(self) -> Agent:
        """Génération de la stratégie de prise de contact personnalisée."""
        return Agent(
            config=self._agent_config("outreach_strategy_agent"),
            verbose=True,
            tools=[SerperDevTool(), FileReadTool("/knowledge/Fiche_IPO_Technologie.md")],
            knowledge_sources_1=[self.knowledge_source_1],  # Ajout de la knowledge source
            llm=LLM(model="gpt-5-mini"),
        )

    @task
    def t1_profil_linkedin(self) -> Task:
        """Étape 1 : fiche profil LinkedIn structurée."""
        return self._build_task("t1_profil_linkedin")

    @task
    def t2_cinq_derniers_posts(self) -> Task:
        """Étape 2 : extraction des 5 derniers posts."""
        return self._build_task("t2_cinq_derniers_posts")

    @task
    def t3_commentaires_du_prospect_sur_posts(self) -> Task:
        """Étape 3 : collecte des commentaires sur nos contenus."""
        return self._build_task("t3_commentaires_du_prospect_sur_posts")

    @task
    def t4_fiche_entreprise_site(self) -> Task:
        """Étape 4 : fiche entreprise basée sur le site officiel."""
        return self._build_task("t4_fiche_entreprise_site")

    @task
    def t5_fiche_officielle_mcp(self) -> Task:
        """Étape 5 : enrichissement avec la base MCP officielle."""
        return self._build_task("t5_fiche_officielle_mcp")

    @task
    def t6_veille_sectorielle(self) -> Task:
        """Étape 6 : synthèse de veille sectorielle."""
        return self._build_task("t6_veille_sectorielle")

    @task
    def t7_strategie_prise_de_contact(self) -> Task:
        """Étape 7 : plan d'approche multicanal personnalisé."""
        task = self._build_task("t7_strategie_prise_de_contact")
        # Définir manuellement le contexte avec les tâches précédentes
        task.context = [
            self.t1_profil_linkedin(),
            self.t2_cinq_derniers_posts(),
            self.t3_commentaires_du_prospect_sur_posts(),
            self.t4_fiche_entreprise_site(),
            self.t5_fiche_officielle_mcp(),
            self.t6_veille_sectorielle()
        ]
        return task

    @crew
    def crew(self) -> Crew:
        """Assembler agents et tâches puis renvoyer la crew prête à lancer."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )