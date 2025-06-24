
from neural_network_grader import NeuralNetworkGrader

# Initialize the grader
grader = NeuralNetworkGrader("trained_nn_grading_model.pt")

# Use the grader to predict scores
score = grader.predict_score(
    bert_similarity=0.85,   # Semantic similarity from BERT
    sbert_similarity=0.78,  # Sentence similarity from SBERT
    keyword_similarity=0.92 # Keyword matching score
)

print(f"Predicted score: {score:.2f}")
