from datasets import load_dataset
import pandas as pd

def inspect():
    try:
        dataset = load_dataset("gonzaloa/fake_news")
        train_data = dataset['train']
        
        print("Checking first 10 examples...")
        for i in range(10):
            item = train_data[i]
            print(f"Label: {item['label']} | Title: {item['title'][:100]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
