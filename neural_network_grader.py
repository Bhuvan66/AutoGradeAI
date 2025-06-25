
import torch
import json
import os
import numpy as np

class GradingNN(torch.nn.Module):
    def __init__(self, input_size=7):
        super(GradingNN, self).__init__()
        # Simpler network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class NeuralNetworkGrader:
    def __init__(self, model_path, config_path=None):
        # Try to load config if available
        input_size = 7  # Default to simplified model size
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                input_size = config.get('input_size', 7)
                
        self.model = GradingNN(input_size=input_size)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def predict_score(self, bert_similarity, sbert_similarity, keyword_similarity):
        """Predict score using the neural network model with derived features"""
        # Calculate derived features
        product_similarity = bert_similarity * sbert_similarity * keyword_similarity
        bert_similarity_squared = bert_similarity ** 2
        sbert_similarity_squared = sbert_similarity ** 2
        
        # Calculate harmonic mean
        if bert_similarity > 0 and sbert_similarity > 0 and keyword_similarity > 0:
            harmonic_mean = 3 / (1/bert_similarity + 1/sbert_similarity + 1/keyword_similarity)
        else:
            harmonic_mean = 0
            
        # Create feature vector
        features = torch.FloatTensor([{
            bert_similarity, 
            sbert_similarity, 
            keyword_similarity,
            product_similarity,
            bert_similarity_squared,
            sbert_similarity_squared,
            harmonic_mean
        }])
        
        # Get prediction
        with torch.no_grad():
            score = self.model(features).item()
        return max(0.0, min(1.0, score))

# Example usage
if __name__ == "__main__":
    # Initialize the grader
    model_path = "trained_nn_grading_modelcontrolled.pt"
    config_path = "model_config.json"
    grader = NeuralNetworkGrader(model_path, config_path)
    
    # Use the grader to predict scores
    score = grader.predict_score(
        bert_similarity=0.85,   # Semantic similarity from BERT
        sbert_similarity=0.78,  # Sentence similarity from SBERT
        keyword_similarity=0.92 # Keyword matching score
    )
    
    print(f"Predicted score: {score:.2f}")
