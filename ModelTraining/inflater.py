import pandas as pd
import os

def inflate_ai_scores(input_file_path, output_file_path, inflation_rate=0.25):
    """
    Inflate AI scores by specified percentage while ensuring they don't exceed 100%
    
    Args:
        input_file_path (str): Path to the original CSV file
        output_file_path (str): Path for the new inflated CSV file
        inflation_rate (float): Inflation rate (default 0.10 for 10%)
    """
    try:
        # Read the original CSV file
        df = pd.read_csv(input_file_path)
        
        # Find AI score column (assuming it contains 'ai' in the name)
        ai_score_column = None
        for col in df.columns:
            if 'ai' in col.lower() or 'score' in col.lower():
                ai_score_column = col
                break
        
        if ai_score_column is None:
            print("Warning: Could not find AI score column. Using first numeric column.")
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                ai_score_column = numeric_columns[0]
            else:
                raise ValueError("No numeric columns found in the dataset")
        
        print(f"Inflating column: {ai_score_column}")
        
        # Create a copy of the dataframe
        inflated_df = df.copy()
        
        # Apply inflation and cap at 100 (or 1.0 if scores are in 0-1 range)
        inflated_scores = inflated_df[ai_score_column] * (1 + inflation_rate)
        
        # Check if scores are in 0-1 range or 0-100 range
        max_original_score = df[ai_score_column].max()
        if max_original_score <= 1.0:
            # Scores are in 0-1 range, cap at 1.0
            inflated_df[ai_score_column] = inflated_scores.clip(upper=1.0)
        else:
            # Scores are in 0-100 range, cap at 100
            inflated_df[ai_score_column] = inflated_scores.clip(upper=100)

        
        # Save to new CSV file
        inflated_df.to_csv(output_file_path, index=False)
        
        print(f"Successfully created inflated dataset: {output_file_path}")
        print(f"Original scores range: {df[ai_score_column].min():.2f} - {df[ai_score_column].max():.2f}")
        print(f"Inflated scores range: {inflated_df[ai_score_column].min():.2f} - {inflated_df[ai_score_column].max():.2f}")
        
        return inflated_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    # Define file paths
    input_file = r"C:\Users\bhuva\Desktop\Projects\Working\AutoGradeAI\ModelTraining\AutoGradeAI_enhanced_dataset2.csv"
    output_file = r"C:\Users\bhuva\Desktop\Projects\Working\AutoGradeAI\ModelTraining\AutoGradeAI_enhanced_dataset2_inflated.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
    else:
        # Inflate the scores
        result = inflate_ai_scores(input_file, output_file)
        
        if result is not None:
            print("Inflation process completed successfully!")
        else:
            print("Inflation process failed!")
