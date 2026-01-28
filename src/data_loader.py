
import os
import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Class to handle loading of datasets for Fake News Detection.
    Supports loading from local CSV files or Hugging Face Datasets API.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_social_media_dataset(self):
        """
        Loads the 'gonzaloa/fake_news' dataset which contains 
        diverse news articles often shared on social media.
        """
        logging.info("Loading Social Media Fake News dataset (gonzaloa/fake_news)...")
        try:
            dataset = load_dataset("gonzaloa/fake_news")
            # This dataset has 'train', 'validation', 'test'
            # Columns: 'id', 'title', 'text', 'label'
            # Label: 0 (Fake), 1 (Real) -- need to verify this standard
            
            train_df = pd.DataFrame(dataset['train'])
            val_df = pd.DataFrame(dataset['validation'])
            test_df = pd.DataFrame(dataset['test'])
            
            # Combine title and text for better context
            train_df.fillna("", inplace=True)
            val_df.fillna("", inplace=True)
            test_df.fillna("", inplace=True)
            
            train_df['statement'] = train_df['title'] + " " + train_df['text']
            val_df['statement'] = val_df['title'] + " " + val_df['text']
            test_df['statement'] = test_df['title'] + " " + test_df['text']
            
            # --- Domain Adaptation / Augmentation ---
            # Inject generic knowledge concepts (Tech, Science, Health) that might be missing
            # or biased in the original political dataset.
            logging.info("Augmenting training data with generic domain knowledge...")
            aug_df = self._get_augmented_data()
            
            # OVERSAMPLING: Replicate these high-quality samples to ensure they survive
            # the statistical noise of the massive 70k merged dataset.
            # 30 samples * 200 = 6000 samples (~8% of data).
            aug_df = pd.concat([aug_df] * 200, ignore_index=True)
            
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            
            # --- Merge with Local ISOT Dataset (User Provided) ---
            isot_train, isot_val, isot_test = self._load_custom_isot_data()
            if isot_train is not None:
                logging.info(f"Merging ISOT Dataset ({len(isot_train)} samples) with Social Media Dataset...")
                train_df = pd.concat([train_df, isot_train], ignore_index=True)
                val_df = pd.concat([val_df, isot_val], ignore_index=True)
                test_df = pd.concat([test_df, isot_test], ignore_index=True)
            
            logging.info(f"Final Training Set Size: {len(train_df)} samples.")
            return train_df, val_df, test_df
        except Exception as e:
            logging.error(f"Failed to load social media dataset: {e}")
            return None, None, None

    def _load_custom_isot_data(self):
        """
        Loads the Fake.csv and True.csv files from data/raw if they exist.
        Returns train/val/test splits.
        """
        fake_path = os.path.join(self.raw_dir, "Fake.csv")
        true_path = os.path.join(self.raw_dir, "True.csv")
        
        if not (os.path.exists(fake_path) and os.path.exists(true_path)):
            logging.info("Custom ISOT files (Fake.csv/True.csv) not found. Skipping.")
            return None, None, None
            
        logging.info("Loading custom ISOT dataset (Fake.csv + True.csv)...")
        try:
            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)
            
            fake_df['label'] = 0
            true_df['label'] = 1
            
            # Combine
            df = pd.concat([fake_df, true_df], ignore_index=True)
            df.fillna("", inplace=True)
            df['statement'] = df['title'] + " " + df['text']
            
            # Keep only necessary columns
            df = df[['statement', 'label']]
            
            # Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split (80/10/10)
            n = len(df)
            train_end = int(n * 0.8)
            val_end = int(n * 0.9)
            
            train = df.iloc[:train_end]
            val = df.iloc[train_end:val_end]
            test = df.iloc[val_end:]
            
            return train, val, test
        except Exception as e:
            logging.error(f"Error loading ISOT files: {e}")
            return None, None, None

    def _get_augmented_data(self):
        """
        Returns a DataFrame of diverse real/fake examples to improve generalization
        beyond US politics.
        """
        data = {
            'label': [],
            'statement': []
        }
        
        # REAL NEWS SAMPLES (Science, Tech, Health, World)
        real_news = [
            "Apple unveiled its latest iPhone model featuring improved battery efficiency and AI.",
            "Scientists at ISRO successfully tested a reusable launch vehicle.",
            "The World Health Organization reported a decline in global COVID-19 deaths.",
            "NASA confirms the discovery of water molecules on the sunlit surface of the Moon.",
            "Tesla releases new software update improving self-driving capabilities.",
            "Global stock markets rallied today reaching new all-time highs.",
            "Study shows that regular exercise reduces the risk of heart disease.",
            "Microsoft announces partnership to advance quantum computing research.",
            "Olympics committee introduces new sports for the upcoming summer games.",
            "Astronomers capture the first image of a black hole in the Milky Way.",
            "New battery technology promises to double electric vehicle range.",
            "Researchers develop a malaria vaccine with high efficacy.",
            "Google updates search algorithm to prioritize high-quality content.",
            "Amazon launches drone delivery service in select cities.",
            "SpaceX successfully lands Starship rocket after orbital flight.",
            "The World Health Organization explicitly stated that COVID-19 vaccines are safe.",
            "ISRO successfully launches Chandrayaan-3 into orbit."
        ]
        
        # FAKE NEWS SAMPLES (Conspiracy, Pseudoscience, Clickbait)
        fake_news = [
            "Apple secretly records every conversation you have and sells it to aliens.",
            "ISRO faked the moon landing in a studio in Bollywood.",
            "World Health Organization admits vitamins are poisonous and banned them.",
            "NASA confirms that the earth is actually a flat disc resting on turtles.",
            "Tesla cars have gained sentience and are plotting world domination.",
            "Stock market crash predicted to wipe out all money next Tuesday guaranteed.",
            "Eating dirt is the secret cure to all known diseases according to blog.",
            "Microsoft puts microchips in vaccines to control your thoughts.",
            "Olympics cancelled forever due to ghost sightings in the stadium.",
            "Astronomers admit the sky is a hologram projected by the government.",
            "New battery made of infinite energy crystals discovered in Atlantis.",
            "Malaria is actually caused by 5G towers not mosquitoes.",
            "Google will pay you $5000 just for opening this email.",
            "Amazon drones are actually spy cameras for the lizard people.",
            "SpaceX rocket hits the firmament proves space isn't empty.",
            "Apple releases new iPhone with holographic display.",
            "Aliens found in Mars by NASA rover.",
            "Government announces free gold for everyone."
        ]
        
        for txt in real_news:
            data['label'].append(1) # REAL
            data['statement'].append(txt)
            
        for txt in fake_news:
            data['label'].append(0) # FAKE
            data['statement'].append(txt)
            
        return pd.DataFrame(data)

    def load_liar_dataset(self):
        """
        Loads the LIAR dataset. 
        Attempts Hugging Face first, falls back to direct download/manual loading.
        Returns:
            train_df, val_df, test_df (pandas.DataFrame)
        """
        logging.info("Attempting to load LIAR dataset from Hugging Face...")
        try:
            # Try loading via datasets library (might fail due to script policy)
            # We explicitly catch the error now.
            dataset = load_dataset("liar", trust_remote_code=True)
            train_df = pd.DataFrame(dataset['train'])
            val_df = pd.DataFrame(dataset['validation'])
            test_df = pd.DataFrame(dataset['test'])
            logging.info("LIAR dataset loaded via Hugging Face.")
            return train_df, val_df, test_df
        except Exception as e:
            logging.warning(f"Hugging Face load failed ({e}). Falling back to manual download...")
            return self._download_and_load_manual()

    def _download_and_load_manual(self):
        """
        Manually downloads the LIAR dataset or loads from local if present.
        """
        import requests
        import zipfile
        import io

        # Check if files already exist locally
        train_path = os.path.join(self.raw_dir, 'train.tsv')
        val_path = os.path.join(self.raw_dir, 'valid.tsv')
        test_path = os.path.join(self.raw_dir, 'test.tsv')

        if os.path.exists(train_path):
            logging.info(f"Found local dataset files in {self.raw_dir}. Loading...")
            try:
                col_names = [
                    'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                    'state_info', 'party_affiliation', 'barely_true_counts', 
                    'false_counts', 'half_true_counts', 'mostly_true_counts', 
                    'pants_on_fire_counts', 'context'
                ]
                train_df = pd.read_csv(train_path, sep='\t', header=None, names=col_names, on_bad_lines='skip')
                val_df = pd.read_csv(val_path, sep='\t', header=None, names=col_names, on_bad_lines='skip')
                test_df = pd.read_csv(test_path, sep='\t', header=None, names=col_names, on_bad_lines='skip')
                logging.info(f"Loaded {len(train_df)} rows from local train.tsv")
                return train_df, val_df, test_df
            except Exception as e:
                logging.error(f"Error loading local files: {e}")
                # Fallthrough to mock if local load fails
        
        logging.warning("Local files not found or failed to load.")
        logging.warning("Defaulting to MOCK DATASET for demonstration.")
        return self._create_mock_data()


    def _create_mock_data(self):
        """
        Creates a small mock dataset to verify the pipeline execution.
        Using standard LIAR labels to ensure correct mapping in train.py.
        """
        # labels: 0=fake, 1=real
        # Mapping in train.py: 
        #   Fake: ['false', 'barely-true', 'pants-fire']
        #   Real: ['true', 'mostly-true', 'half-true']
        
        mock_data = {
            'label': [
                'false', 'true', 
                'pants-fire', 'mostly-true', 
                'barely-true', 'half-true'
            ] * 20, # 120 samples
            'statement': [
                'The earth is flat and the moon is made of cheese.', 
                'The sun rises in the east and sets in the west.',
                'Aliens have taken over the government and are controlling our minds.',
                'The economy grew by 2% last quarter according to the report.',
                'Vaccines cause magnetism in humans and allow them to stick to fridges.',
                'Studies show that eating vegetables is generally good for your health.'
            ] * 20
        }
        df = pd.DataFrame(mock_data)
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train/val/test
        train = df.iloc[:80]
        val = df.iloc[80:100]
        test = df.iloc[100:]
        
        logging.info("Mock dataset created (Balanced).")
        return train, val, test

    def load_local_csv(self, filename, split='train'):
        """
        Loads a local CSV file from data/raw directory.
        """
        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            logging.info(f"Loading local file: {filepath}")
            return pd.read_csv(filepath)
        else:
            logging.warning(f"File not found: {filepath}")
            return None

    def save_to_processed(self, df, filename):
        """
        Saves a dataframe to the processed data directory.
        """
        filepath = os.path.join(self.processed_dir, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved processed data to {filepath}")

if __name__ == "__main__":
    loader = DataLoader()
    # Example usage
    # train, val, test = loader.load_liar_dataset()
