import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Define a dictionary of common terms with their respective weights
# to indicate frequency or importance in the word cloud.
common_terms = {
    "python": 10000, "r": 9500, "sql": 9000, "markdown": 8500, "bash": 7500,
    "numpy": 7000, "pandas": 6500, "matplotlib": 6000, "seaborn": 5500, "ggplot2": 5000,
    "tensorflow": 4500, "keras": 4000, "scikit_learn": 3500, "jupyter_notebook": 3000, "docker": 2500,
    "kubernetes": 2000, "git": 1800, "github": 1600, "rest_api": 1400, "json": 1200, "xml": 1100,
    "regular_expressions": 1000, "svm": 900, "natural_language_processing": 800, "convolutional_neural_network": 700,
    "long_short_term_memory": 600, "roc_curve": 500, "auc_score": 400, "k_nearest_neighbors": 300,
    "big_data": 200, "apache_hadoop": 1000, "apache_spark": 900, "apache_airflow": 800, "apache_kafka": 700,
    "machine_learning": 600, "artificial_intelligence": 500, "neural_network": 400, "clustering": 300,
    "classification": 200, "regression": 100, "anova": 900, "t_test": 800, "bayesian_statistics": 700,
    "markov_chain_monte_carlo": 600, "monte_carlo_simulation": 500, "markov_chain": 400, "decision_trees": 300,
    "random_forest": 200, "support_vector_machines": 100, "logistic_regression": 1000,
    "linear_regression": 900, "polynomial_regression": 800, "ridge_regression": 700,
    "lasso_regression": 600, "elastic_net_regression": 500, "cross_validation": 400,
    "regularization": 300, "hyperparameter_tuning": 200, "ensemble_methods": 100,
    "gradient_boosting": 1000, "bagging": 900, "unsupervised_learning": 800,
    "reinforcement_learning": 700, "supervised_learning": 600, "flask": 500,
    "django": 400, "r_shiny": 300, "biopython": 200, "sympy": 100, "scipy": 1000, "plotly": 900,
    "bokeh": 800, "altair": 700, "plotly_dash": 600, "virtualenv": 500, "pip": 400, "conda": 300,
    "vagrant": 200, "ansible": 100, "terraform": 1000, "devops": 900, "agile_methodology": 800,
    "scrum": 700, "kanban": 600, "join": 500, "select": 400, "where_clause": 300, "order_by": 200,
    "group_by": 100, "having_clause": 1000, "union": 900, "intersect": 800, "except": 700,
    "limit": 600, "inner_join": 500, "left_join": 400, "right_join": 300, "full_join": 200,
    "index": 100, "view": 1000, "trigger": 900, "stored_procedure": 800, "cursor": 600,
    "rollback": 500, "commit": 400, "savepoint": 300, "transaction": 200, "vim": 100,
    "emacs": 1000, "nano": 900, "sed": 800, "awk": 700, "grep": 600, "find": 500, "sort": 400,
    "tar": 300, "gzip": 200, "bzip2": 100, "xz": 1000, "cron": 900, "systemd": 800, "init": 700,
    "daemon": 600, "shell_scripting": 500, "loop": 400, "if_statement": 300, "else_clause": 200, "elif_clause": 100,
    "case": 1000, "switch": 900, "function": 800, "array": 700, "variable": 600,
    "error_handling": 500, "debugging": 400, "unit_testing": 300, "integration_testing": 200,
    "functional_testing": 100, "performance_testing": 1000, "security_testing": 900,
    "load_testing": 800, "stress_testing": 700, "penetration_testing": 600,
    "optimization": 500, "profiling": 400, "benchmarking": 300, "logging": 200,
    "monitoring": 100, "alerting": 1000, "visualization": 900, "reporting": 800,
    "analysis": 700, "prediction": 600, "forecasting": 500, "modeling": 400, "sampling": 300,
    "estimation": 200, "inference": 100, "dimensionality_reduction": 1000, "feature_selection": 900,
    "feature_engineering": 800, "data_cleaning": 700, "data_preprocessing": 600,
    "data_wrangling": 500, "data_munging": 400, "data_transformation": 300,
    "data_integration": 200, "data_migration": 100, "data_warehousing": 1000,
    "data_lake": 900, "data_mart": 800, "olap": 700, "oltp": 600, "etl": 500, "elt": 400,
    "data_pipeline": 300, "data_flow": 200, "data_governance": 100, "data_security": 1000,
    "data_privacy": 900, "data_ethics": 800, "data_compliance": 700, "data_audit": 600,
    "data_quality": 500, "data_accuracy": 400, "data_reliability": 300, "data_validity": 200,
    "data_completeness": 100, "data_consistency": 1000, "data_timeliness": 900,
    "data_lineage": 800, "data_provenance": 700, "data_catalog": 600,
    "metadata_management": 500, "metadata_extraction": 400, "metadata_indexing": 300,
    "metadata_storage": 200, "metadata_search": 100, "metadata_analysis": 1000,
    "metadata_visualization": 900, "metadata_reporting": 800, "metadata_security": 700,
    "metadata_privacy": 600
}

# Randomize weights within their logical ranges to introduce variability
randomized_terms = {
    term: random.randint(max(100, weight - 500), min(10000, weight + 500))
    for term, weight in common_terms.items()
}

# Define available color palettes with a dictionary mapping names to colormap strings
color_palettes = {
    "Vibrant": "turbo",
    "Monochrome": "gray",
    "Ocean": "ocean",
    "Hot": "hot",
    "Rainbow": "rainbow",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Magma": "magma",
}

# Present the available color palettes to the user and ask for their choice
print("Available color palettes:")
for i, palette in enumerate(color_palettes.keys(), 1):
    print(f"{i}. {palette}")
choice = int(input("Choose a color palette (enter the number): ")) - 1
selected_palette = list(color_palettes.values())[choice]

# Create a word cloud with the corrected frequencies using the user-selected colormap
wordcloud = WordCloud(
    width=1584, height=396, background_color='black', colormap=selected_palette
).generate_from_frequencies(randomized_terms)

# Display the word cloud using matplotlib
plt.figure(figsize=(15.84, 3.96))  # Size in inches to match the pixel size
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Save the word cloud as a PNG image to the specified file
wordcloud.to_file("linkedin_banner.png")

# Save the list of common words and their weights to a text file
output_file = "common_words.txt"
with open(output_file, 'w') as f:
    for term, weight in sorted(randomized_terms.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{term}\t{weight}\n")
print(f"Common words saved to '{output_file}'")
