import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Define a dictionary of common terms with their respective weights
# to indicate frequency or importance in the word cloud.
common_terms = {
    "Financial_Advisory": 2000, "Data_Analysis": 10000, "Auditing": 4000,
    "Financial_Modeling": 6000, "Database_Management": 7000, "Power_BI": 7000,
    "Tableau": 7000, "SQL": 9500, "R": 9000, "Python": 10000, "Machine_Learning": 9500,
    "Business_Analytics": 9000, "Data_Analytics": 10000, "Financial_Analysis": 6000,
    "Audit_Assurance": 3000, "Financial_Services": 5000, "CICPA": 1000, "FRM": 1000,
    "CFA_Level_III": 1000, "Transaction_Services": 4000, "Data-driven_financial_analysis": 8000,
    "Real_estate": 2000, "Healthcare": 2000, "Education": 2000, "E-commerce": 2000,
    "Manufacturing": 2000, "Advanced_data_analytics": 8500, "Transactional_operations": 4000,
    "Investment_value": 5000, "Power_Query": 6000, "Earnings_quality": 4000,
    "Deal_negotiation": 4000, "Financial_policies": 4000, "Income_data": 5000,
    "Decision-making_processes": 7000, "Audit_planning": 4000, "Project_scope": 3000,
    "Milestone_planning": 3000, "Compliance": 5000, "Risk_exposure": 5000,
    "Quantitative_analyses": 8000, "Qualitative_analyses": 8000, "Investment_profits": 4000,
    "Strategic_fund_management": 5000, "Financial_guidance": 5000, "Financial_forecast": 7000,
    "Budget_expenditures": 5000, "Audit_procedures": 4000, "IFRS": 3000, "HKFRS": 3000,
    "CAS_standards": 3000, "AI-enhanced_Gradio_interface": 6000, "Financial_data_queries": 7000,
    "Automated_processing": 7000, "Large_Language_Models": 8500, "Time_series_models": 9000,
    "Sales_forecasting": 8500, "Visual_trend_analysis": 7000, "Scalable_Database_Management_System": 7000,
    "Relational_schemas": 6000, "ERDs": 6000, "Data_dictionary": 6000, "SQL_scripts": 7000,
    "Operational_efficiency": 7000, "Data_integrity": 7000, "CFA_ESG": 1000, "SAC": 1000,
    "English": 2000, "Mandarin": 2000
}

# Randomize weights within their logical ranges to introduce variability
randomized_terms = {
    term: random.randint(max(100, weight - 500), min(10000, weight + 500))
    for term, weight in common_terms.items()
}

# Ask the user to choose between a light and dark background
print("Choose a background color:")
background_color_options = {
    "1": "white",
    "2": "black"
}
while True:
    print("1. Light (White) - Ideal for a clean and professional look.")
    print("2. Dark (Black) - Great for highlighting colors and a modern aesthetic.")
    background_choice = input("Enter your choice (1 or 2): ")
    if background_choice in background_color_options:
        selected_background = background_color_options[background_choice]
        break
    else:
        print("Please input only 1 or 2 for the background color choice.")

# Define available color palettes with a dictionary mapping names to colormap strings and descriptions
color_palettes = {
    "Vibrant": ("turbo", "A dynamic range of colors for a lively and energetic appearance."),
    "Monochrome": ("gray", "A grayscale palette for a sophisticated and timeless look."),
    "Ocean": ("ocean", "Blues and greens evoke the calm and depth of the sea."),
    "Hot": ("hot", "Warm colors like reds and oranges for a bold, intense effect."),
    "Rainbow": ("rainbow", "All the colors of the rainbow for a playful and inclusive feel."),
    "Viridis": ("viridis", "A smooth gradient from yellow-green to dark blue, modern and clear."),
    "Plasma": ("plasma", "Bright, luminous colors transitioning from purple to yellow."),
    "Inferno": ("inferno", "Deep reds to bright yellows, capturing the essence of heat and energy."),
    "Magma": ("magma", "Rich purples and reds for a mysterious and powerful vibe.")
}

# Present the available color palettes to the user and ask for their choice
print("Available color palettes:")
for i, (palette, description) in enumerate(color_palettes.items(), 1):
    print(f"{i}. {palette} - {description}")
while True:
    palette_choice = input("Choose a color palette (enter the number): ")
    if palette_choice.isdigit() and int(palette_choice) in range(1, len(color_palettes) + 1):
        selected_palette = list(color_palettes.values())[int(palette_choice) - 1][0]
        break
    else:
        print("Please input only the numbers corresponding to the color palettes.")

# Create a word cloud with the corrected frequencies using the user-selected colormap and background color
wordcloud = WordCloud(
    width=1584, height=396, background_color=selected_background, colormap=selected_palette
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
