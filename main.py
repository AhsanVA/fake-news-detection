
import argparse
import sys
import os
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, default='logistic_regression', help='Model to train')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', type=str, default='logistic_regression', help='Model to evaluate')
    
    # EDA command
    subparsers.add_parser('eda', help='Run EDA script')
    
    # App command
    subparsers.add_parser('app', help='Run Streamlit App')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.train import train
        train(args.model)
        
    elif args.command == 'evaluate':
        from src.evaluate import evaluate
        evaluate(args.model)
        
    elif args.command == 'eda':
        subprocess.run(["python", "notebooks/eda_script.py"])
        
    elif args.command == 'app':
        print("Starting Streamlit App...")
        subprocess.run(["streamlit", "run", "app/app.py"])
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
