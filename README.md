# linkedin-banner-wordcloud-generator
This Python script generates a customizable word cloud from a predefined list of terms, allowing the user to select a color palette, randomize term weights, and save the resulting visualization and ranked term list.

## Features

- **Customizable Color Palette**: Allows the user to select from a list of predefined color palettes to style the word cloud.

- **Interactive User Input**: Prompts the user to select a color palette from the console.

- **Automated Randomization**: Adjusts the frequency of terms within a specified range to introduce variability.

- **Visualization**: Displays the word cloud using matplotlib and saves it as a PNG image.

- **Output Saving**: Saves the ranked list of terms and their weights to a text file.

## Requirements

To run this script, ensure you have the following Python libraries installed:

- `matplotlib`: For generating and displaying the word cloud image.

- `wordcloud`: For creating the word cloud from frequencies.

- `random`: For randomizing term weights.

You can install the necessary libraries using pip:

```bash
pip install matplotlib wordcloud
```

## Usage

1. **Run the Script**: Execute the script in a Python environment. The script will display available color palettes.

2. **Select a Color Palette**: Enter the number corresponding to your chosen color palette when prompted.

3. **View and Save Outputs**: The script will display the word cloud in the chosen color scheme and save it as `linkedin_banner.png`. Additionally, it will save the sorted terms and their weights to `common_words.txt`.

## Code Overview

### Defining Terms and Weights

A dictionary `common_terms` holds key terms and their initial weights. These weights represent the frequency or importance of each term in the word cloud.

### Randomizing Term Weights

The script uses the `random` library to adjust each term's weight within a logical range, adding variability to the visualization.

### Color Palette Selection

A dictionary `color_palettes` maps user-friendly names to matplotlib colormap strings. The script prompts the user to select one of these palettes.

### Generating and Displaying the Word Cloud

The `WordCloud` class from the `wordcloud` library is used to generate the word cloud with the specified dimensions, background color, and colormap. The word cloud is then displayed using `matplotlib`.

### Saving Outputs

The word cloud image is saved as `linkedin_banner.png`, and the terms with their adjusted weights are saved in `common_words.txt`.

## Customization

To customize the script further, you can modify the `common_terms` dictionary to include different terms and weights or add more color palettes to the `color_palettes` dictionary.

linkedin-banner-wordcloud-generator © 2024 by Philip Bachas-Daunert is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
