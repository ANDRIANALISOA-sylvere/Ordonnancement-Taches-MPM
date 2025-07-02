import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import math
import json
import os

# Set page configuration
st.set_page_config(page_title="Ordonnancement MPM - Gestion des Tâches", layout="wide")
st.title("Ordonnancement MPM - Gestion des Tâches")

# File to store tasks data
DATA_FILE = "tasks_data.json"

# Functions to save and load data
def save_tasks_to_file(tasks):
    """Save tasks to a JSON file"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")

def load_tasks_from_file():
    """Load tasks from JSON file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return {}

# Initialize session state for storing tasks
if 'tasks' not in st.session_state:
    # Try to load from file first
    st.session_state.tasks = load_tasks_from_file()

# Sidebar for data management
with st.sidebar:
    st.header("Gestion des Données")
    
    # Clear data button
    if st.button("🗑️ Effacer toutes les données", type="secondary"):
        if st.session_state.tasks:
            st.session_state.tasks = {}
            # Also delete the file
            try:
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                st.success("Toutes les données ont été effacées!")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la suppression du fichier: {e}")
        else:
            st.info("Aucune donnée à effacer.")
    
    # Show number of tasks
    st.info(f"Nombre de tâches: {len(st.session_state.tasks)}")
    
    # Export/Import functionality
    if st.session_state.tasks:
        st.download_button(
            label="📥 Télécharger les données",
            data=json.dumps(st.session_state.tasks, ensure_ascii=False, indent=2),
            file_name="mpm_tasks.json",
            mime="application/json"
        )
    
    uploaded_file = st.file_uploader("📤 Importer des données", type=['json'])
    if uploaded_file is not None:
        try:
            imported_data = json.load(uploaded_file)
            st.session_state.tasks = imported_data
            save_tasks_to_file(st.session_state.tasks)
            st.success("Données importées avec succès!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {e}")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Ajouter une Tâche")
    
    # Form for adding tasks
    task_name = st.text_input("Nom de la tâche")
    task_duration = st.number_input("Durée (jours)", min_value=1, step=1)
    
    # Show available tasks for selection as antecedents
    available_tasks = list(st.session_state.tasks.keys())
    task_antecedents = st.multiselect("Tâches antérieures", options=available_tasks)
    
    # Add task button
    if st.button("➕ Ajouter Tâche", type="primary"):
        if task_name and task_name not in st.session_state.tasks:
            st.session_state.tasks[task_name] = {"duree": task_duration, "antecedents": task_antecedents}
            save_tasks_to_file(st.session_state.tasks)  # Save to file
            st.success(f"Tâche '{task_name}' ajoutée avec succès!")
            # Force a rerun to update the available tasks list
            st.rerun()
        elif task_name in st.session_state.tasks:
            st.error(f"Une tâche avec le nom '{task_name}' existe déjà.")
        else:
            st.error("Veuillez entrer un nom de tâche valide.")

with col2:
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
        
        if st.button("🗑️ Supprimer la tâche") and task_to_remove:
            # Remove the task
            del st.session_state.tasks[task_to_remove]
            # Update antecedents in other tasks
            for task, data in st.session_state.tasks.items():
                if task_to_remove in data["antecedents"]:
                    data["antecedents"].remove(task_to_remove)
            save_tasks_to_file(st.session_state.tasks)  # Save to file
            st.success(f"Tâche '{task_to_remove}' supprimée!")
            st.rerun()
    else:
        st.info("Aucune tâche n'a été ajoutée. Utilisez le formulaire ci-contre pour ajouter des tâches.")

# Generate graph button
if st.session_state.tasks:
    st.divider()
    if st.button("📊 Générer le Graphe MPM", type="primary"):
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
                st.subheader("📋 Résultats MPM")
                
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
                        "Chemin critique": "✅ Oui" if node in critical_path else "❌ Non"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Display graph
                st.subheader("🎯 Graphe MPM")
                
                # Improved graph layout parameters
                node_radius = 0.8
                horizontal_spacing = 4.0
                vertical_spacing = 3.0
                
                # Calculate positions using hierarchical layout
                # Group nodes by their topological level
                levels = {}
                for node in nx.topological_sort(G):
                    if not list(G.predecessors(node)):
                        levels[node] = 0
                    else:
                        levels[node] = max(levels[pred] for pred in G.predecessors(node)) + 1
                
                # Group nodes by level
                nodes_by_level = {}
                for node, level in levels.items():
                    if level not in nodes_by_level:
                        nodes_by_level[level] = []
                    nodes_by_level[level].append(node)
                
                # Calculate positions
                pos = {}
                max_level = max(levels.values()) if levels else 0
                
                for level, nodes in nodes_by_level.items():
                    num_nodes = len(nodes)
                    # Center the nodes in each level
                    start_y = -(num_nodes - 1) * vertical_spacing / 2
                    
                    for i, node in enumerate(nodes):
                        x = level * horizontal_spacing
                        y = start_y + i * vertical_spacing
                        pos[node] = (x, y)
                
                # Create figure with better sizing
                fig_width = max(12, (max_level + 1) * 3)
                fig_height = max(8, len(G.nodes()) * 1.5)
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # Draw nodes
                for node in G.nodes():
                    node_x, node_y = pos[node]
                    is_critical = node in critical_path
                    circle_color = "lightcoral" if is_critical else "lightblue"
                    edge_color = "darkred" if is_critical else "black"
                    
                    # Draw the circle
                    circle = Circle((node_x, node_y), node_radius, fill=True, 
                                  edgecolor=edge_color, facecolor=circle_color, 
                                  linewidth=2, zorder=2)
                    ax.add_patch(circle)
                    
                    # Add vertical line (doesn't go below horizontal line)
                    ax.plot([node_x, node_x], [node_y, node_y + node_radius], 
                          color=edge_color, zorder=3, linewidth=2)
                    
                    # Add horizontal line
                    ax.plot([node_x - node_radius, node_x + node_radius], [node_y, node_y], 
                          color=edge_color, zorder=3, linewidth=2)
                    
                    # Add text labels with better positioning and sizing
                    # Left: Earliest start
                    ax.text(node_x - node_radius*0.5, node_y + node_radius*0.4, 
                          f"{earliest_start[node]}", ha='center', va='center', 
                          fontsize=10, fontweight='bold', zorder=4, color='darkblue')
                    
                    # Right: Latest start
                    ax.text(node_x + node_radius*0.5, node_y + node_radius*0.4, 
                          f"{latest_start[node]}", ha='center', va='center', 
                          fontsize=10, fontweight='bold', zorder=4, color='darkgreen')
                    
                    # Bottom: Task name
                    ax.text(node_x, node_y - node_radius*0.3, 
                          f"{node}", ha='center', va='center', 
                          fontsize=9, fontweight='bold', zorder=4)
                
                # Draw edges
                for u, v in G.edges():
                    edge_color = "red" if u in critical_path and v in critical_path else "black"
                    line_width = 3 if u in critical_path and v in critical_path else 2
                    
                    u_x, u_y = pos[u]
                    v_x, v_y = pos[v]
                    
                    # Calculate edge endpoints on circle boundaries
                    angle = np.arctan2(v_y - u_y, v_x - u_x)
                    
                    start_x = u_x + node_radius * np.cos(angle)
                    start_y = u_y + node_radius * np.sin(angle)
                    
                    end_x = v_x - node_radius * np.cos(angle)
                    end_y = v_y - node_radius * np.sin(angle)
                    
                    # Draw arrow
                    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                                           arrowstyle='->', 
                                           connectionstyle='arc3,rad=0.1',
                                           color=edge_color, 
                                           linewidth=line_width, 
                                           mutation_scale=20,
                                           zorder=1)
                    ax.add_patch(arrow)
                    
                    # Add duration label on edge
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # Offset for label positioning
                    offset_angle = angle + np.pi/2
                    offset_distance = 0.3
                    
                    label_x = mid_x + offset_distance * np.cos(offset_angle)
                    label_y = mid_y + offset_distance * np.sin(offset_angle)
                    
                    ax.text(label_x, label_y, f"{G.nodes[u]['duree']}", 
                          ha='center', va='center', 
                          fontsize=9, fontweight='bold', 
                          bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black', pad=2), 
                          zorder=5)
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Adjust plot limits
                if pos:
                    x_values = [pos[node][0] for node in G.nodes()]
                    y_values = [pos[node][1] for node in G.nodes()]
                    
                    margin = node_radius * 2
                    plt.xlim(min(x_values) - margin, max(x_values) + margin)
                    plt.ylim(min(y_values) - margin, max(y_values) + margin)
                
                # Add legend
                critical_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='lightcoral', markeredgecolor='darkred',
                                         markersize=15, markeredgewidth=2, label='Chemin critique')
                normal_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='lightblue', markeredgecolor='black',
                                        markersize=15, markeredgewidth=2, label='Tâche normale')
                
                plt.legend(handles=[critical_patch, normal_patch], 
                         loc='upper left', fontsize=12, framealpha=0.9)
                
                # Add title with project info
                plt.suptitle(f"Diagramme MPM - Durée totale: {project_end} jours", 
                           fontsize=16, fontweight='bold', y=0.95)
                
                # Show the plot
                st.pyplot(fig)
                
                # Show project information
                st.subheader("📊 Informations du Projet")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Durée totale", f"{project_end} jours")
                with col2:
                    st.metric("Nombre de tâches", len(G.nodes()))
                with col3:
                    st.metric("Tâches critiques", len(critical_path))
                
                # Show critical path
                st.info(f"🔥 **Chemin critique:** {' → '.join(critical_path)}")
                
                # Legend explanation
                with st.expander("📖 Légende du graphique"):
                    st.markdown("""
                    **Structure des nœuds:**
                    - **Coin supérieur gauche:** Date au plus tôt de début
                    - **Coin supérieur droit:** Date au plus tard de début  
                    - **Partie inférieure:** Nom de la tâche
                    
                    **Couleurs:**
                    - **Rouge/Rose:** Tâches sur le chemin critique (marge = 0)
                    - **Bleu clair:** Tâches normales (avec marge)
                    
                    **Flèches:**
                    - **Rouge épais:** Liaisons du chemin critique
                    - **Noir normal:** Liaisons normales
                    - **Étiquettes jaunes:** Durée de la tâche source
                    """)
        
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la génération du graphe: {str(e)}")
            st.error(f"Détails: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown("*Outil de gestion de projet MPM (Méthode des Potentiels Métra)*")