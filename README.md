# linkedin-banner-wordcloud-generator
This Python script generates a customizable word cloud from a predefined list of terms, allowing the user to select a background color and a color palette, randomize term weights, and save the resulting visualization and ranked term list. The output is tailored to fit the LinkedIn banner size.

## Features

- **Background Color Selection**: Users can choose between a light or dark background for the word cloud, enhancing visibility based on preference.

- **Customizable Color Palette with Descriptions**: The script offers a selection of predefined color palettes, each described to help users decide the best fit for their visualization.

- **Interactive User Input**: Prompts the user to select a background color and color palette from the console, handling any input errors gracefully.

- **Automated Randomization**: Adjusts the frequency of terms within a specified range to introduce variability and reflect the relative importance of each term.

- **Visualization**: Displays the word cloud using matplotlib and saves it as a PNG image formatted specifically for LinkedIn banners.

- **Output Saving**: Saves the ranked list of terms and their weights to a text file, `common_words.txt`.

## LinkedIn Banner Size
As of the current date, the LinkedIn banner size used in this script is 1584x396 pixels, ensuring that the generated word cloud fits perfectly as a LinkedIn profile banner.

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

1. **Run the Script**: Execute the script in a Python environment. The script first asks you to choose a background color and then displays available color palettes with descriptions.

2. **Select a Background Color and Color Palette**: Enter the number corresponding to your chosen background color and color palette when prompted.

3. **View and Save Outputs**: The script will display the word cloud in the chosen color scheme and save it as `linkedin_banner.png`. Additionally, it will save the sorted terms and their weights to `common_words.txt`.

## Code Overview

### Defining Terms and Weights

A dictionary `common_terms` holds key terms and their initial weights. These weights represent the frequency or importance of each term in the word cloud.

### Randomizing Term Weights

The script uses the `random` library to adjust each term's weight within a logical range, adding variability to the visualization.

### Background and Color Palette Selection

Users first choose between a light or dark background. A dictionary `color_palettes` maps user-friendly names to matplotlib colormap strings and includes short descriptions. The script prompts the user to select one of these palettes.

### Generating and Displaying the Word Cloud

The `WordCloud` class from the `wordcloud` library generates the word cloud with the specified dimensions (1584x396 pixels), background color, and colormap. The word cloud is then displayed using `matplotlib`.

### Saving Outputs

The word cloud image is saved as `linkedin_banner.png`, fitting the LinkedIn banner size. The terms with their adjusted weights are saved in `common_words.txt`.

## Customization

To customize the script further, you can modify the `common_terms` dictionary to include different terms and weights or add more color palettes to the `color_palettes` dictionary.

linkedin-banner-wordcloud-generator © 2024 by Philip Bachas-Daunert is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)