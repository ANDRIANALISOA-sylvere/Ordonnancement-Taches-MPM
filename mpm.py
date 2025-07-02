import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import math

# Set page configuration
st.set_page_config(page_title="Ordonnancement MPM - Gestion des Tâches", layout="wide")
st.title("Ordonnancement MPM - Gestion des Tâches")

# Initialize session state for storing tasks
if 'tasks' not in st.session_state:
    st.session_state.tasks = {}

# Form for adding tasks
task_name = st.text_input("Nom de la tâche")
task_duration = st.number_input("Durée (jours)", min_value=1, step=1)

# Show available tasks for selection as antecedents
available_tasks = list(st.session_state.tasks.keys())
task_antecedents = st.multiselect("Tâches antérieures", options=available_tasks)

# Add task button
if st.button("Ajouter Tâche"):
    if task_name and task_name not in st.session_state.tasks:
        st.session_state.tasks[task_name] = {"duree": task_duration, "antecedents": task_antecedents}
        st.success(f"Tâche '{task_name}' ajoutée avec succès!")
        # Force a rerun to update the available tasks list
        st.rerun()
    elif task_name in st.session_state.tasks:
        st.error(f"Une tâche avec le nom '{task_name}' existe déjà.")
    else:
        st.error("Veuillez entrer un nom de tâche valide.")

# Display tasks
if st.session_state.tasks:
    st.subheader("Liste des Tâches")
    
    # Convert tasks to DataFrame for better display
    tasks_data = []
    for name, data in st.session_state.tasks.items():
        tasks_data.append({
            "Tâche": name,
            "Durée (jours)": data["duree"],
            "Antérieurs": ", ".join(data["antecedents"]) if data["antecedents"] else "Aucun"
        })
    
    df = pd.DataFrame(tasks_data)
    st.dataframe(df, use_container_width=True)
    
    # Add option to remove a task
    task_to_remove = st.selectbox("Sélectionner une tâche à supprimer", 
                                 options=[""] + list(st.session_state.tasks.keys()))
    
    if st.button("Supprimer la tâche") and task_to_remove:
        # Remove the task
        del st.session_state.tasks[task_to_remove]
        # Update antecedents in other tasks
        for task, data in st.session_state.tasks.items():
            if task_to_remove in data["antecedents"]:
                data["antecedents"].remove(task_to_remove)
        st.success(f"Tâche '{task_to_remove}' supprimée!")
        st.rerun()
else:
    st.info("Aucune tâche n'a été ajoutée. Utilisez le formulaire ci-dessus pour ajouter des tâches.")

# Generate graph button
if st.session_state.tasks and st.button("Générer le Graphe MPM"):
    try:
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for nom, data in st.session_state.tasks.items():
            G.add_node(nom, duree=data["duree"])
            for antecedent in data["antecedents"]:
                G.add_edge(antecedent, nom)
        
        # Verify graph is a DAG (Directed Acyclic Graph)
        if not nx.is_directed_acyclic_graph(G):
            st.error("Le graphe contient des cycles. Veuillez vérifier les relations entre les tâches.")
        else:
            # MPM calculations
            
            # Calcul des dates au plus tôt (Forward Pass)
            earliest_start = {}
            earliest_end = {}
            
            for node in nx.topological_sort(G):
                if not list(G.predecessors(node)):  # Si pas de prédécesseur, commencer à 0
                    earliest_start[node] = 0
                else:
                    earliest_start[node] = max(earliest_end[predecessor] for predecessor in G.predecessors(node))
                
                earliest_end[node] = earliest_start[node] + G.nodes[node]['duree']
            
            # Calcul des dates au plus tard (Backward Pass)
            latest_end = {}
            latest_start = {}
            
            # Initialize latest_end with project end time
            project_end = max(earliest_end.values())
            
            for node in reversed(list(nx.topological_sort(G))):
                if not list(G.successors(node)):  # Dernière tâche
                    latest_end[node] = project_end
                else:
                    latest_end[node] = min(latest_start[successor] for successor in G.successors(node))
                
                latest_start[node] = latest_end[node] - G.nodes[node]['duree']
            
            # Calcul des marges et du chemin critique
            margins = {node: latest_start[node] - earliest_start[node] for node in G.nodes()}
            critical_path = [node for node in G.nodes() if margins[node] == 0]
            
            # Display MPM results in a table
            st.subheader("Résultats MPM")
            
            results_data = []
            for node in G.nodes():
                results_data.append({
                    "Tâche": node,
                    "Durée": G.nodes[node]['duree'],
                    "Date tôt début": earliest_start[node],
                    "Date tôt fin": earliest_end[node],
                    "Date tard début": latest_start[node],
                    "Date tard fin": latest_end[node],
                    "Marge": margins[node],
                    "Chemin critique": "Oui" if node in critical_path else "Non"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Display graph
            st.subheader("Graphe MPM")
            
            # Paramètres fixés
            node_radius = 0.5  # Taille fixe pour tous les nœuds
            max_nodes_per_row = 10  # Nombre max de tâches par ligne
            
            # Créer le rang des nœuds pour un meilleur placement
            ranks = {}
            nodes_by_rank = {}
            
            for node in nx.topological_sort(G):
                # Calculez le rang (niveau) de chaque nœud basé sur le chemin le plus long depuis une source
                if not list(G.predecessors(node)):  # Si pas de prédécesseur
                    ranks[node] = 0
                else:
                    ranks[node] = 1 + max(ranks[pred] for pred in G.predecessors(node))
                
                # Regroupez les nœuds par rang
                if ranks[node] not in nodes_by_rank:
                    nodes_by_rank[ranks[node]] = []
                nodes_by_rank[ranks[node]].append(node)

            # Déterminer la hauteur maximale nécessaire basée sur le nombre de nœuds à chaque rang
            max_nodes_in_any_rank = max(len(nodes) for nodes in nodes_by_rank.values())
            
            # Calculer la hauteur totale de la figure
            max_rank = max(ranks.values())
            num_rows_needed = math.ceil((max_rank + 1) / max_nodes_per_row)
            
            # Ajuster la taille de la figure en fonction du nombre de lignes et de la taille des nœuds
            fig_height = max(8, 6 * num_rows_needed)
            fig_width = 14
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Assignez des positions basées sur le rang, avec "wrapping" pour les rangs élevés
            pos = {}
            horizontal_spacing = node_radius * 4  # Plus d'espace entre les nœuds horizontalement
            vertical_spacing = node_radius * 6  # Plus d'espace entre les lignes
            
            # Pour chaque rang, déterminer sur quelle ligne il sera et positionner les nœuds
            for rank, nodes in nodes_by_rank.items():
                # Déterminer sur quelle ligne ce rang doit être dessiné
                row = rank // max_nodes_per_row
                col = rank % max_nodes_per_row
                
                # Espacez les nœuds de même rang uniformément
                for i, node in enumerate(nodes):
                    # Position horizontale basée sur le rang, avec "wrapping"
                    x_pos = col * horizontal_spacing
                    
                    # Position verticale - séparation par lignes
                    # Les nœuds sont placés plus loin les uns des autres verticalement
                    y_base = -row * vertical_spacing
                    y_pos = y_base - (i - (len(nodes) - 1) / 2) * (node_radius * 3)
                    
                    pos[node] = (x_pos, y_pos)
            
            # Créer un dictionnaire pour stocker les connexions entre les lignes
            connections_between_rows = {}
            
            # Trouver les nœuds à la fin de chaque ligne et ceux au début de la ligne suivante
            for row in range(num_rows_needed):
                if row < num_rows_needed - 1:  # S'il y a une ligne suivante
                    # Trouver les rangs dans cette ligne et la ligne suivante
                    ranks_in_this_row = [r for r in range(row * max_nodes_per_row, min((row + 1) * max_nodes_per_row, max_rank + 1))]
                    ranks_in_next_row = [r for r in range((row + 1) * max_nodes_per_row, min((row + 2) * max_nodes_per_row, max_rank + 1))]
                    
                    if not ranks_in_next_row:
                        continue
                    
                    # Trouver les nœuds à la fin de cette ligne
                    if ranks_in_this_row:
                        last_rank_in_row = max(ranks_in_this_row)
                        nodes_at_end = nodes_by_rank.get(last_rank_in_row, [])
                        
                        # Trouver les nœuds au début de la ligne suivante
                        first_rank_in_next_row = min(ranks_in_next_row)
                        nodes_at_start_of_next = nodes_by_rank.get(first_rank_in_next_row, [])
                        
                        # Pour chaque nœud à la fin, s'il a une connexion avec un nœud au début de la ligne suivante
                        for end_node in nodes_at_end:
                            for start_node in nodes_at_start_of_next:
                                if (end_node, start_node) in G.edges():
                                    # Stocker cette connexion pour référence
                                    connections_between_rows[(end_node, start_node)] = True
            
            # Draw nodes as perfect circles with three sections
            for node in G.nodes():
                # Position du nœud
                node_x, node_y = pos[node]
                is_critical = node in critical_path
                circle_color = "lightcoral" if is_critical else "lightblue"
                
                # Dessiner le cercle
                circle = Circle((node_x, node_y), node_radius, fill=True, 
                              edgecolor='black', facecolor=circle_color, zorder=2)
                ax.add_patch(circle)
                
                # Ajouter ligne verticale QUI NE DESCEND PAS en-dessous de la ligne horizontale
                ax.plot([node_x, node_x], [node_y, node_y + node_radius], 
                      color='black', zorder=3, linewidth=1)
                
                # Ajouter ligne horizontale pour séparer la partie inférieure
                ax.plot([node_x - node_radius, node_x + node_radius], [node_y, node_y], 
                      color='black', zorder=3, linewidth=1)
                
                # Ajouter les étiquettes de texte (plus petites)
                # Gauche: Date au plus tôt
                ax.text(node_x - node_radius*0.6, node_y + node_radius*0.4, 
                      f"{earliest_start[node]}", ha='center', va='center', 
                      fontsize=8, fontweight='bold', zorder=4)
                
                # Droite: Date au plus tard
                ax.text(node_x + node_radius*0.6, node_y + node_radius*0.4, 
                      f"{latest_start[node]}", ha='center', va='center', 
                      fontsize=8, fontweight='bold', zorder=4)
                
                # Bas: Nom de la tâche
                ax.text(node_x, node_y - node_radius*0.5, 
                      f"{node}", ha='center', va='center', 
                      fontsize=8, fontweight='bold', zorder=4)

            # Draw edges
            for u, v in G.edges():
                # Ne pas tracer la flèche si cette connexion est entre des lignes
                if (u, v) in connections_between_rows:
                    continue
                
                edge_color = "red" if u in critical_path and v in critical_path else "black"
                
                u_x, u_y = pos[u]
                v_x, v_y = pos[v]
                
                # Calculer les points d'extrémité qui sont exactement sur les bords des cercles
                angle = np.arctan2(v_y - u_y, v_x - u_x)
                
                # Point de départ sur le bord du cercle de départ
                start_x = u_x + node_radius * np.cos(angle)
                start_y = u_y + node_radius * np.sin(angle)
                
                # Point d'arrivée sur le bord du cercle d'arrivée
                end_x = v_x - node_radius * np.cos(angle)
                end_y = v_y - node_radius * np.sin(angle)
                
                # Dessiner la flèche
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                                       arrowstyle='->', 
                                       connectionstyle='arc3,rad=0.1',
                                       color=edge_color, 
                                       linewidth=1.5, 
                                       mutation_scale=15,
                                       zorder=1)
                ax.add_patch(arrow)
                
                # Ajouter la durée de la tâche au-dessus de la flèche
                # Calculer le point médian de la flèche
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # Offset pour placer le texte au-dessus de la flèche
                offset_angle = angle + np.pi/2
                offset_distance = 0.2
                
                label_x = mid_x + offset_distance * np.cos(offset_angle)
                label_y = mid_y + offset_distance * np.sin(offset_angle)
                
                # Ajouter la durée comme label
                ax.text(label_x, label_y, f"{G.nodes[u]['duree']}", 
                      ha='center', va='center', 
                      fontsize=8, fontweight='bold', 
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), 
                      zorder=5)
            
            # Dessiner les connexions virtuelles entre les lignes
            # Pour chaque noeud à la fin d'une ligne, créer un "clone" au début de la ligne suivante
            for row in range(num_rows_needed - 1):
                # Trouver les rangs dans cette ligne et la ligne suivante
                ranks_in_this_row = [r for r in range(row * max_nodes_per_row, min((row + 1) * max_nodes_per_row, max_rank + 1))]
                ranks_in_next_row = [r for r in range((row + 1) * max_nodes_per_row, min((row + 2) * max_nodes_per_row, max_rank + 1))]
                
                if not ranks_in_next_row:
                    continue
                
                # Identifier les nœuds qui devraient être "clonés"
                if ranks_in_this_row:
                    last_rank_in_row = max(ranks_in_this_row)
                    nodes_to_clone = nodes_by_rank.get(last_rank_in_row, [])
                    
                    # Positionner les clones au début de la ligne suivante
                    clone_pos_x = 0  # Position x au début de la ligne suivante
                    for i, node in enumerate(nodes_to_clone):
                        # Position y basée sur l'organisation des nœuds dans le rang précédent
                        clone_pos_y = -(row + 1) * vertical_spacing - (i - (len(nodes_to_clone) - 1) / 2) * (node_radius * 3)
                        
                        is_critical = node in critical_path
                        circle_color = "lightcoral" if is_critical else "lightblue"
                        
                        # Dessiner le cercle cloné
                        clone_circle = Circle((clone_pos_x, clone_pos_y), node_radius, fill=True, 
                                           edgecolor='black', facecolor=circle_color, zorder=2)
                        ax.add_patch(clone_circle)
                        
                        # Ajouter ligne verticale
                        ax.plot([clone_pos_x, clone_pos_x], [clone_pos_y, clone_pos_y + node_radius], 
                              color='black', zorder=3, linewidth=1)
                        
                        # Ajouter ligne horizontale
                        ax.plot([clone_pos_x - node_radius, clone_pos_x + node_radius], [clone_pos_y, clone_pos_y], 
                              color='black', zorder=3, linewidth=1)
                        
                        # Ajouter les étiquettes de texte
                        ax.text(clone_pos_x - node_radius*0.6, clone_pos_y + node_radius*0.4, 
                              f"{earliest_start[node]}", ha='center', va='center', 
                              fontsize=8, fontweight='bold', zorder=4)
                        
                        ax.text(clone_pos_x + node_radius*0.6, clone_pos_y + node_radius*0.4, 
                              f"{latest_start[node]}", ha='center', va='center', 
                              fontsize=8, fontweight='bold', zorder=4)
                        
                        ax.text(clone_pos_x, clone_pos_y - node_radius*0.5, 
                              f"{node}", ha='center', va='center', 
                              fontsize=8, fontweight='bold', zorder=4)
                        
                        # Dessiner des flèches partant de ces clones vers leurs successeurs
                        for succ in G.successors(node):
                            # Vérifier si le successeur est sur la même ligne
                            if ranks[succ] in ranks_in_next_row:
                                succ_x, succ_y = pos[succ]
                                
                                # Calculer les points d'extrémité
                                angle = np.arctan2(succ_y - clone_pos_y, succ_x - clone_pos_x)
                                
                                # Point de départ sur le bord du cercle cloné
                                start_x = clone_pos_x + node_radius * np.cos(angle)
                                start_y = clone_pos_y + node_radius * np.sin(angle)
                                
                                # Point d'arrivée sur le bord du cercle d'arrivée
                                end_x = succ_x - node_radius * np.cos(angle)
                                end_y = succ_y - node_radius * np.sin(angle)
                                
                                edge_color = "red" if node in critical_path and succ in critical_path else "black"
                                
                                # Dessiner la flèche
                                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                                                      arrowstyle='->', 
                                                      connectionstyle='arc3,rad=0.1',
                                                      color=edge_color, 
                                                      linewidth=1.5, 
                                                      mutation_scale=15,
                                                      zorder=1)
                                ax.add_patch(arrow)
                                
                                # Ajouter la durée comme label
                                mid_x = (start_x + end_x) / 2
                                mid_y = (start_y + end_y) / 2
                                offset_angle = angle + np.pi/2
                                offset_distance = 0.2
                                
                                label_x = mid_x + offset_distance * np.cos(offset_angle)
                                label_y = mid_y + offset_distance * np.sin(offset_angle)
                                
                                ax.text(label_x, label_y, f"{G.nodes[node]['duree']}", 
                                      ha='center', va='center', 
                                      fontsize=8, fontweight='bold', 
                                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), 
                                      zorder=5)
            
            # Définir l'aspect du graphe comme égal pour maintenir des cercles parfaits
            ax.set_aspect('equal')
            
            # Remove axis
            ax.axis('off')
            
            # Adjust plot limits with margin
            margin = node_radius * 2
            x_values = [pos[node][0] for node in G.nodes()]
            y_values = [pos[node][1] for node in G.nodes()]
            
            # Ajouter les positions des nœuds clonés pour s'assurer qu'ils sont dans la vue
            for row in range(num_rows_needed - 1):
                ranks_in_this_row = [r for r in range(row * max_nodes_per_row, min((row + 1) * max_nodes_per_row, max_rank + 1))]
                if ranks_in_this_row:
                    last_rank_in_row = max(ranks_in_this_row)
                    nodes_to_clone = nodes_by_rank.get(last_rank_in_row, [])
                    for i in range(len(nodes_to_clone)):
                        y_values.append(-(row + 1) * vertical_spacing - (i - (len(nodes_to_clone) - 1) / 2) * (node_radius * 3))
                    x_values.append(0)  # Position x des clones
            
            if x_values and y_values:  # S'assurer qu'il y a des valeurs
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                plt.xlim(x_min - margin, x_max + margin)
                plt.ylim(y_min - margin, y_max + margin)
            
            # Add a legend
            critical_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='lightcoral', markersize=15, label='Chemin critique')
            normal_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='lightblue', markersize=15, label='Tâche normale')
            plt.legend(handles=[critical_patch, normal_patch], loc='upper left')
            
            # Show the plot
            st.pyplot(fig)
            
            # Show project information
            st.subheader("Informations du Projet")
            st.write(f"Durée totale du projet: {project_end} jours")
            st.write(f"Chemin critique: {' → '.join(critical_path)}")
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la génération du graphe: {str(e)}")
        st.error(f"Détails: {type(e).__name__}")