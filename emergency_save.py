import json
from pathlib import Path

# Quick check for training metadata
metadata_path = Path("models/trained_standalone/training_metadata.json")

if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("🏆 MODEL PERFORMANCE METRICS:")
    performance = metadata.get('model_performance', {})
    
    for metric, value in performance.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")
else:
    print("❌ training_metadata.json not found")
