# Agents IA pour la génération de code en PNT

Ce repo regroupe mes essais d'agents IA pour la génération de code. Les agents IA de code apparaissent très performants sur les tâches de génération et de traduction de code, notamment sur les languages (python, C/C++, Fortran) et frameworks populaires (Pytorch, Jax).


3 initiatives sont mises à l'épreuve :
- Albert-Large (Mistral Small), proposé par la DSI Météo-France et la DINUM -> [Interface OpenWebUI](https://openwebui.s1.kube-sidev.meteo.fr/),
- Codestral (Mistral) sur Ollama : installation sur une instance EWC (Nvidia A100 + 40GB RAM GPU) + SSH Tunneling + client cline sur vscode,
- Claude Sonnet 4 (Anthropic) servi via [cline](https://app.cline.bot/dashboard), ~20 € pour 200K tokens,

Synthèse :
- Mistral-Small exposé par la DINUM est un petit modèle généraliste, il répond efficacement à des questions de code génériques (ex : numpy ou matplotlib),
- Codestral est performant sur les tâches de traduction de code : Fortran vers python/numpy, Fortran vers C++, insertion de directives OpenACC,
- Claude Sonnet est la référence des agents IA pour le développement logiciel, sa puissance est de prendre en compte le contexte du projet pour proposer un code adéquat. Performant pour les tâches de traduction (Fortran -> Jax notamment).

Mon retour est que les agents, notamment Claude Sonnet, gèrent très bien le contexte du projet, et proposent des solutions pertinentes (qui s'exécutent/compilent à minima) dans la plupart des cas. 

L'exemple de génération de schémas d'advections est bluffant : l'agent propose une revue des principaux schémas de la littérature, leur implémentation en python et une documentation (avec références scientifiques).

Claude Sonnet a été mis en défaut sur des tâches spécifiques et peu documentées : génération de code GT4Py, génération d'une interface Fortran -> Python. 

En synthèse, les agents IA font gagner un temps précieux sur les tâches à répétitives de génération de code. Ils ne dispensent pas d'une revue de code et de tests adéquats, mais permettent à l'utilisateur de se concentrer sur les points clés d'architecture logicielle et les cas "pathologiques". Codestral et Claude requièrent un temps d'apprentissage pour les prompts et le contexte à renseigner : plus la tâche demandée est précise et concise, plus l'agent est pertinent (en particulier pour Codestral). Le vrai atout d'un agent de AI comme Claude Sonnet 4 est sa prise en compte du contexte : sur le projet courant, il cherche et prend en contexte le code du projet, peut imiter l'architecture mise en place, et proposer des modifications sur des routines existantes.


## Albert Large - Mistral Small

Prompt : "Je veux faire  la figure d’une section verticale d’un traceur avec epygram et une légende colorbound variable. Ecrit la fonction python."

Réponse :

```python
import epygram
import matplotlib.pyplot as plt
import numpy as np

def plot_vertical_section(field_name, level_dim, file_path, output_path):
    # Charger les données
    f = epygram.formats.resource.open(file_path)

    # Extraire les données du champ spécifié
    field = f.read_field(field_name)
    levels = f.read_field(level_dim)

    # Extraire les valeurs du champ
    data = field.values
    lons = field.grid.lon.values
    lats = field.grid.lat.values
    z_levels = levels.values

    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracer la section verticale
    contour = ax.contourf(lons, z_levels, data, cmap='viridis')


    # Ajouter une barre de couleur
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(field_name)

    # Ajouter des labels et un titre
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Niveau vertical')
    ax.set_title(f'Section verticale de {field_name}')

    # Sauvegarder la figure
    plt.savefig(output_path)
    plt.close(fig)

# Exemple d'utilisation
file_path = 'path/to/your/datafile.arn'  # Remplacez par le chemin de votre fichier de données
output_path = 'vertical_section.png'     # Chemin de sortie pour l'image
field_name = 'T'                          # Nom du champ à tracer (par exemple, température)
level_dim = 'lev'                         # Nom de la dimension des niveaux verticaux

plot_vertical_section(field_name, level_dim, file_path, output_path)

```

## Codestral - European Weather Cloud

[codestral:22b](https://mistral.ai/news/codestral) a été installé sur une instance de l'EWC. 

Ici, codestral est exposé via le serveur d'inférence [ollama](ollama.ai), et donne accès à une CLI et un endpoint API. 

```bash
    ollama run codestral:22b "your-prompt" 
```

Intégration VSCode via le plugin [Continue.dev](https://docs.continue.dev/). Le plugin propose un chat et l'autocomplétion (flexible, avec plusieurs modèles), avec la possibilité d'ajouter un contexte prédéfini aux prompts. 

3 tâches ont été exécutées sur la subroutine **convect_closure_shal.F90**:
- conversion de la subroutine en numpy,
- insertion de directives OpenACC,
- conversion de la routine en DaCe.

Les 3 exemples donnent des résultats qui compilent ou s'exécutent sans erreurs.

[Résumé détaillé](/convection_codestral/genai.md).

## Claude-Sonnet 4 (via Cline)

### 1. Génération d'un schéma d'advection

- [Advection](ia_advection) : "Give me the code of an advection scheme for an atmospheric model."

Claude propose l'implémentation de 6 schémas d'advections usuels de la littérature accompagnés de leurs références scientifiques et d'un test 1D.

### 2. Génération d'une interface Fortran / Python

- [FortranCppPlugin](fortran_cpp_plugin) : "Generate the interface to pass a fortran array to c restricted pointer."

Une interface de Fortran vers C est générée. L'interface présente tous les cas possibles [1d, 2d, 3d] x [float32, float64], compile et est directement utilisable dans un code Fortran.

### 3. Traduction d'ice3 vers jax et gt4py

- [jax-IceAdjust](jax_ice_adjust) : "Translate ice_adjust.F90 to jax." 

Claude propose une traduction de ice_adjust vers Jax, avec implémentation des particularités de jax (@jit). La reproductibilité n'a pas été vérifiée mais le code est directement utilisable.

