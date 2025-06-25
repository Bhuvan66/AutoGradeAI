import pandas as pd
import random
import time
from tqdm import tqdm
import numpy as np
import ollama  # Using ollama Python module instead of HTTP requests

# Function to call Ollama API with Gemma3:4b model
def query_ollama(prompt, model="gemma3:4b"):
    try:
        response = ollama.generate(model=model, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

# Modify the prompts to explicitly instruct the model to provide clean answers
def get_clean_prompt(base_prompt):
    return f"{base_prompt} Provide ONLY the direct answer without any prefixes like 'Here is', 'I would define', etc. Don't include phrases like 'Hope this helps' or any other commentary. Just provide the technical definition directly."

# Topics for generation
tech_topics = [
    "Machine Learning", "Database Management", "Cloud Computing", "Network Security",
    "Data Structures", "Web Development", "Artificial Intelligence", "Operating Systems",
    "Computer Architecture", "Software Engineering", "Cybersecurity", "Big Data",
    "Internet of Things", "Blockchain", "DevOps", "Mobile Development",
    "Quantum Computing", "Version Control", "API Design", "Microservices",
    "Front-end Development", "Back-end Development", "Python Programming", "JavaScript",
    "Data Mining", "Natural Language Processing", "Computer Vision", "Robotics",
    "User Experience Design", "Agile Methodology", "Docker", "Kubernetes",
    "Serverless Computing", "Edge Computing", "5G Technology", "Virtual Reality",
    "Augmented Reality", "Cryptography", "Neural Networks", "TensorFlow"
]

# Function to generate student answer with a specific target score
def generate_answer_with_target_score(preferred_answer, target_score):
    """Generate a student answer targeting a specific score (0-100)"""
    
    if target_score >= 95:
        # For perfect or near-perfect scores
        return preferred_answer
    
    elif target_score >= 80:
        # For excellent answers (80-94)
        prompt = get_clean_prompt(f"I have an ideal answer: '{preferred_answer}'. Generate a student answer that would deserve a score of {target_score}/100. It should be very good but with minor imperfections or omissions.")
        
    elif target_score >= 60:
        # For good answers (60-79)
        prompt = get_clean_prompt(f"I have an ideal answer: '{preferred_answer}'. Generate a student answer that would deserve a score of {target_score}/100. It should include most key points but miss some details or have some minor misconceptions.")
        
    elif target_score >= 40:
        # For average answers (40-59)
        prompt = get_clean_prompt(f"I have an ideal answer: '{preferred_answer}'. Generate a student answer that would deserve a score of {target_score}/100. It should be partially correct but miss important concepts.")
        
    elif target_score >= 20:
        # For below average answers (20-39)
        prompt = get_clean_prompt(f"I have an ideal answer: '{preferred_answer}'. Generate a student answer that would deserve a score of {target_score}/100. It should contain only a few correct points and miss most key concepts.")
        
    else:
        # For poor answers (0-19)
        if target_score <= 5:
            return "I don't know the answer to this question."
        else:
            prompt = get_clean_prompt(f"I have an ideal answer: '{preferred_answer}'. Generate a student answer that would deserve a score of {target_score}/100. It should be mostly incorrect with only minimal understanding shown.")
    
    return query_ollama(prompt)

# Create the controlled dataset
print("Generating controlled dataset with 400 entries (4 entries per score from 0-100)...")
controlled_entries = []

for score in tqdm(range(101)):  # 0 to 100 inclusive
    target_score = score  # The actual score we want (0-100)
    
    for entry_num in range(4):  # 4 entries per score
        # Generate a preferred answer
        topic = random.choice(tech_topics)
        prompt = get_clean_prompt(f"Generate a concise, accurate technical definition of {topic} in one sentence. Make it factually correct and clear.")
        preferred_answer = query_ollama(prompt)
        
        if not preferred_answer:
            continue
            
        # Generate student answer targeting specific score
        student_answer = generate_answer_with_target_score(preferred_answer, target_score)
        
        if not student_answer:
            continue
            
        # Add to entries with the exact target score
        controlled_entries.append({
            "Student Answer": student_answer,
            "Preferred Answer": preferred_answer,
            "AI Score": target_score / 100.0  # Store as decimal (0-1)
        })
        
        # Add a small delay to prevent rate limiting
        time.sleep(0.1)

# Create DataFrame
controlled_df = pd.DataFrame(controlled_entries)

# Save the dataset
output_path = r"ModelTraining\AutoGradeAI_controlled_dataset.csv"
controlled_df.to_csv(output_path, index=False)

print(f"Successfully created controlled dataset with {len(controlled_entries)} entries.")
print(f"Dataset saved to: {output_path}")

# Optionally merge with existing dataset
merge_with_existing = False
if merge_with_existing:
    existing_path = r"ModelTraining\AutoGradeAI_enhanced_dataset2.csv"
    existing_df = pd.read_csv(existing_path)
    merged_df = pd.concat([existing_df, controlled_df], ignore_index=True)
    merged_path = r"ModelTraining\AutoGradeAI_combined_dataset.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"Combined dataset with {len(merged_df)} entries saved to: {merged_path}")

