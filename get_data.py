from datasets import load_dataset
import os
from tqdm import tqdm

# Specific classes to keep
FOLDERS_TO_KEEP = [
    "beef_tartare",
    "chicken_quesadilla",
    "risotto",
    "spaghetti_carbonara",
    "pancakes"
]

OUTPUT_DIR = "./food_subset"

def main():
    print("Loading dataset from Hugging Face...")
    # Load the dataset
    dataset = load_dataset("ethz/food101", trust_remote_code=True)
    
    # Get class names from features
    class_names = dataset["train"].features["label"].names
    # Map our target names to their label IDs
    target_labels = {name: class_names.index(name) for name in FOLDERS_TO_KEEP}
    
    for split_name in ["train", "validation"]:
        print(f"Processing {split_name} split...")
        split_data = dataset[split_name]
        
        # Filter for our specific classes
        filtered_data = split_data.filter(lambda x: x["label"] in target_labels.values())
        
        for item in tqdm(filtered_data, desc=f"Saving {split_name} images"):
            label_id = item["label"]
            label_name = class_names[label_id]
            
            # Create class directory
            target_dir = os.path.join(OUTPUT_DIR, split_name, label_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Generate a filename (using count or hash)
            # Find current number of files to avoid collision
            count = len(os.listdir(target_dir))
            filename = f"{count:04d}.jpg"
            filepath = os.path.join(target_dir, filename)
            
            # Save the image
            item["image"].convert("RGB").save(filepath)

    print(f"Dataset subset created at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
