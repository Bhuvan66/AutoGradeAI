import pandas as pd
import requests
import random
import time
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher

# Function to call Ollama API with Gemma3:4b model
def query_ollama(prompt, model="gemma3:4b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

# Function to calculate similarity score for AI scoring
def calculate_similarity(student_answer, preferred_answer):
    if not student_answer or not preferred_answer:
        return 0.0
    
    if student_answer.lower() == "it is used for something related to tech. i don't remember exactly.":
        return 0.0
        
    matcher = SequenceMatcher(None, student_answer, preferred_answer)
    similarity = matcher.ratio()
    
    # Round to 2 decimal places with some randomness to simulate AI scoring
    similarity = round(min(similarity + random.uniform(-0.1, 0.1), 1.0), 2)
    
    return similarity

# Load existing dataset to understand structure
dataset_path = r"ModelTraining\AutoGradeAI_enhanced_dataset3.csv"
existing_df = pd.read_csv(dataset_path)

# Extract unique preferred answers to use as reference
unique_preferred_answers = existing_df['Preferred Answer'].unique()

# Topics for new generation
tech_topics = [
    "Machine Learning", "Database Management", "Cloud Computing", "Network Security",
    "Data Structures", "Web Development", "Artificial Intelligence", "Operating Systems",
    "Computer Architecture", "Software Engineering", "Cybersecurity", "Big Data",
    "Internet of Things", "Blockchain", "DevOps", "Mobile Development",
    "Quantum Computing", "Version Control", "API Design", "Microservices"
]

# Modify the prompts to explicitly instruct the model to provide clean answers
def get_clean_prompt(base_prompt):
    return f"{base_prompt} Provide ONLY the direct answer without any prefixes like 'Here is', 'I would define', etc. Don't include phrases like 'Hope this helps' or any other commentary. Just provide the technical definition directly."

# Generate exactly 100 exact-match entries
print("Generating 100 exact-match entries using Gemma3:4b via Ollama...")
new_entries = []  # Initialize new_entries for exact matches
for _ in tqdm(range(100)):
    # Generate a new preferred answer
    topic = random.choice(tech_topics)
    prompt = get_clean_prompt(f"Generate a concise, accurate technical definition of {topic} in one sentence. Make it factually correct and clear.")
    preferred_answer = query_ollama(prompt)
    
    # Use the exact same answer for student answer
    student_answer = preferred_answer
    
    # Set AI score to perfect 1.0 since answers match exactly
    ai_score = 1.0
    
    # Add to new entries
    if student_answer and preferred_answer:
        new_entries.append({
            "Student Answer": student_answer,
            "Preferred Answer": preferred_answer,
            "AI Score": ai_score
        })
    
    # Add a small delay to prevent rate limiting
    time.sleep(0.1)

# Create DataFrame with new entries
new_df = pd.DataFrame(new_entries)

# Combine with existing dataset
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save the updated dataset
output_path = r"ModelTraining\AutoGradeAI_enhanced_dataset4.csv"
combined_df.to_csv(output_path, index=False)

print(f"Successfully added {len(new_entries)} new entries to the dataset.")
print(f"Total dataset size: {len(combined_df)} entries")
print(f"Enhanced dataset saved to: {output_path}")

# Generate 100 more exact-match entries
print("Generating 100 more exact-match entries using Gemma3:4b via Ollama...")
more_entries = []  # Initialize for more exact matches
for _ in tqdm(range(100)):
    # Generate a new preferred answer
    topic = random.choice(tech_topics)
    prompt = get_clean_prompt(f"Generate a concise, accurate technical definition of {topic} in one sentence. Make it factually correct and clear.")
    preferred_answer = query_ollama(prompt)
    
    # Use the exact same answer for student answer
    student_answer = preferred_answer
    
    # Set AI score to perfect 1.0 since answers match exactly
    ai_score = 1.0
    
    # Add to new entries
    if student_answer and preferred_answer:
        more_entries.append({
            "Student Answer": student_answer,
            "Preferred Answer": preferred_answer,
            "AI Score": ai_score
        })
    
    # Add a small delay to prevent rate limiting
    time.sleep(0.1)

# Create DataFrame with additional exact-match entries
more_df = pd.DataFrame(more_entries)

# Combine with existing dataset
final_df = pd.concat([combined_df, more_df], ignore_index=True)

# Save the updated dataset
output_path = r"ModelTraining\AutoGradeAI_enhanced_dataset4.csv"
final_df.to_csv(output_path, index=False)

print(f"Successfully added {len(more_entries)} more exact-match entries to the dataset.")
print(f"Total dataset size: {len(final_df)} entries")
print(f"Enhanced dataset saved to: {output_path}")

# Commenting out variations code
"""
# Generate 100 more entries with variations
print("Generating 100 entries with variations using Gemma3:4b via Ollama...")
new_entries_variations = []  # Initialize for varied entries
for _ in tqdm(range(100)):
    # Generate a new preferred answer
    topic = random.choice(tech_topics)
    prompt = get_clean_prompt(f"Generate a concise, accurate technical definition of {topic} in one sentence. Make it factually correct and clear.")
    preferred_answer = query_ollama(prompt)
    
    # Then generate a student answer based on it with clean prompt
    student_type = random.choice(["good", "partial", "poor"])
    
    if student_type == "good":
        prompt = get_clean_prompt(f"Generate a very close but slightly different version of this definition: {preferred_answer}")
    elif student_type == "partial":
        prompt = get_clean_prompt(f"Generate a partial or incomplete version of this definition: {preferred_answer}")
    else:
        prompt = get_clean_prompt(f"Generate a somewhat related but significantly incomplete version of this definition: {preferred_answer}")
        
    student_answer = query_ollama(prompt)

    # Calculate AI score based on similarity
    ai_score = calculate_similarity(student_answer, preferred_answer)
    
    # Add to new entries
    if student_answer and preferred_answer:
        new_entries_variations.append({
            "Student Answer": student_answer,
            "Preferred Answer": preferred_answer,
            "AI Score": ai_score
        })
    
    # Add a small delay to prevent rate limiting
    time.sleep(0.1)
"""

# Removing duplicate code at the end
