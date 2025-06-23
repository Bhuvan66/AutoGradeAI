# Data Generation for AutoGradeAI

This directory contains scripts for generating training data for the AutoGradeAI model.

## Running the Data Generation Script

To generate additional entries for the training dataset:

1. Make sure you have the required dependencies:
   ```
   pip install transformers torch pandas
   ```

2. Run the generation script:
   ```
   python addentries.py
   ```

3. The script will generate approximately 1000 new entries and add them to `AutoGradeAI_generated_dataset.csv`.

## How the Data Generation Works

The script uses the following process:

1. Loads the Gemma-3-4B model from Hugging Face
2. Randomly selects topics from a predefined list
3. Generates a detailed "preferred answer" for each topic
4. Creates a corresponding "student answer" with varying quality levels:
   - Excellent (10% of answers)
   - Good (20% of answers)
   - Average (40% of answers)
   - Poor (20% of answers)
   - Very poor (10% of answers)
5. Assigns an appropriate AI score based on the quality level
6. Saves the generated data to the CSV file

## Notes

- The generation process is designed to create diverse and realistic student answers
- Quality levels are distributed to match real-world grading curves
- The script checks for duplicates to avoid redundant entries
- Generation might take several hours depending on your hardware
