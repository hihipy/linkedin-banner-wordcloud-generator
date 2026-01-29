# LinkedIn Banner Wordcloud Generator

A Python script that generates a word cloud from a list of terms, sized for LinkedIn banners (1584x396 pixels). Supports customizable colors, randomized term weights, and automatic file size optimization.

## Features

- **Background Color Selection:** Choose between light or dark backgrounds.
- **Color Palettes:** Multiple predefined color palettes with descriptions.
- **Randomized Weights:** Adjusts term frequency within a range to add variability.
- **File Size Management:** Ensures output is under 3MB by adjusting quality if needed.
- **Output Files:** Saves the word cloud as PNG and exports ranked terms to a text file.

## Requirements

Python 3.6+ and the following libraries:

```bash
pip install matplotlib wordcloud Pillow tqdm
```

## Usage

1. **Run the Script:** Execute in a Python environment.
2. **Select Background Color:** Choose light or dark when prompted.
3. **Select Color Palette:** Pick from the available options.
4. **View Output:** The word cloud displays and saves as `linkedin_banner.png`. Terms and weights save to `common_words.txt`.

## Customization

Modify the `common_terms` dictionary to change terms and weights, or add color palettes to the `color_palettes` dictionary.

## Technical Details

- **Dimensions:** 1584x396 pixels (LinkedIn banner size)
- **Word Cloud:** Generated using the `wordcloud` library
- **Visualization:** Displayed with `matplotlib`
- **Image Processing:** Uses Pillow for quality adjustment

## License

LinkedIn Banner Wordcloud Generator © 2024 Philip Bachas-Daunert

Distributed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
