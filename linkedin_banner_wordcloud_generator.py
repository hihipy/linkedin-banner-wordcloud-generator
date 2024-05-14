import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from tqdm import tqdm
import os

# Example dictionary of common terms with their respective weights
common_terms = {
    "Financial_Advisory": 2000, "Data_Analytics": 10000, "Auditing": 4000,
    "Financial_Modeling": 6000, "Database_Management": 7000, "Power_BI": 7000,
    "Tableau": 7000, "SQL": 9500, "R": 9000, "Python": 10000, "Machine_Learning": 9500,
    "Business_Analytics": 9000, "Financial_Analysis": 6000, "Financial_Services": 5000,
    "Advanced_Data_Analytics": 8500, "Real_Estate": 2000, "Healthcare": 2000,
    "Education": 2000, "E-commerce": 2000, "Manufacturing": 2000, "Transaction_Services": 4000,
    "Power_Query": 6000, "Financial_Forecast": 7000, "Decision_Making": 7000,
    "Project_Scope": 3000, "Compliance": 5000, "Risk_Exposure": 5000,
    "Quantitative_Analysis": 8000, "Qualitative_Analysis": 8000, "Investment_Profits": 4000,
    "Strategic_Fund_Management": 5000, "Financial_Guidance": 5000, "Budget_Expenditures": 5000,
    "AI-Enhanced_Gradio_Interface": 6000, "Financial_Data_Queries": 7000,
    "Automated_Processing": 7000, "Large_Language_Models": 8500, "Time_Series_Models": 9000,
    "Sales_Forecasting": 8500, "Visual_Trend_Analysis": 7000,
    "Scalable_Database_Management_System": 7000, "Relational_Schemas": 6000, "ERDs": 6000,
    "Data_Dictionary": 6000, "SQL_Scripts": 7000, "Operational_Efficiency": 7000,
    "Data_Integrity": 7000, "English": 2000, "Mandarin": 2000
}

# Adjust weights to ensure they fall within the range 100 to 10,000
adjusted_terms = {
    term: min(10000, max(100, weight))
    for term, weight in common_terms.items()
}

# Randomize weights within their logical ranges to introduce variability
randomized_terms = {
    term: random.randint(max(100, weight - 500), min(10000, weight + 500))
    for term, weight in adjusted_terms.items()
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
for i, (palette, (cmap, description)) in enumerate(color_palettes.items(), 1):
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
    width=1584,
    height=396,
    background_color=selected_background,
    colormap=selected_palette
).generate_from_frequencies(randomized_terms)

# Display the word cloud using matplotlib
plt.figure(figsize=(15.84, 3.96))  # Size in inches to match the pixel size
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Save the word cloud as a PNG image with higher quality
image_path = "linkedin_banner.png"
plt.savefig(image_path, format='png', dpi=300)

# Adjust image size to ensure it is under 3MB
quality_setting = 95

# Adjust quality to get desired file size with progress bar
with tqdm(total=100, desc="Adjusting image size", unit="percent") as pbar:
    while True:
        # Redefine image within the loop
        image = Image.open(image_path)
        image.save(image_path, format='png', optimize=True, quality=quality_setting)
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # Get file size in MB

        if file_size <= 3:
            pbar.update(100 - pbar.n)
            break
        else:
            quality_setting = max(10, quality_setting - 5)
            pbar.update(5)

# Show the word cloud
plt.show()

# Save the list of common words and their weights to a text file
output_file = "common_words.txt"
with open(output_file, 'w') as f:
    for term, weight in sorted(randomized_terms.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{term}\t{weight}\n")

# Print output file details including file size
print(f"Common words saved to '{output_file}'")
print(f"Word cloud image saved to '{image_path}' with file size {file_size:.2f} MB")
