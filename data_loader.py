"""
Data loading and preprocessing for HAM10000 dataset.
Ensures proper sample alignment between image and tabular clients for VFL.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_image(img_path, target_size=(224, 224), augment=False):
    """
    Load and preprocess a single image.
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        augment (bool): Whether to apply data augmentation
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Load image using PIL for better compatibility
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        
        # Apply data augmentation if requested
        if augment:
            img_array = apply_augmentation(img_array)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        # Return a blank image if loading fails
        return np.zeros((*target_size, 3), dtype=np.float32)


def apply_augmentation(img_array):
    """
    Apply data augmentation to an image array.
    
    Args:
        img_array (np.ndarray): Input image array
    
    Returns:
        np.ndarray: Augmented image array
    """
    import random
    
    # Convert to TensorFlow tensor for augmentation
    img_tensor = tf.constant(img_array)
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)
    
    # Random brightness and contrast adjustment
    img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
    img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
    
    # Color jittering for better generalization on skin lesions
    img_tensor = tf.image.random_hue(img_tensor, max_delta=0.1)
    img_tensor = tf.image.random_saturation(img_tensor, lower=0.8, upper=1.2)
    
    # Random rotation using rot90 (90-degree increments)
    if random.random() > 0.7:
        k = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
        img_tensor = tf.image.rot90(img_tensor, k=k)
    
    return img_tensor.numpy()


class HAM10000DataLoader:
    """
    HAM10000 dataset loader with proper sample alignment for VFL.
    """
    
    def __init__(self, data_dir="data", test_size=0.2, val_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Data containers
        self.df = None
        self.image_paths = None
        self.tabular_features = None
        self.labels = None
        self.sensitive_attrs = None
        
        # Split indices for ensuring alignment
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_and_preprocess_data(self, data_percentage=1.0):
        """
        Load and preprocess the HAM10000 dataset.
        
        Args:
            data_percentage (float): Fraction of data to use (for testing)
        """
        print(f"Loading HAM10000 dataset from {self.data_dir}...")
        
        # Load metadata
        metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.df = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(self.df)} samples")
        
        # Clean and preprocess metadata
        self._preprocess_metadata()
        
        # Use subset of data if requested
        if data_percentage < 1.0:
            n_samples = int(len(self.df) * data_percentage)
            # Stratified sampling to maintain class distribution
            from sklearn.model_selection import train_test_split
            self.df, _ = train_test_split(
                self.df, 
                train_size=n_samples, 
                stratify=self.df['dx'], 
                random_state=self.random_state
            )
            print(f"Using {data_percentage*100:.1f}% of data: {len(self.df)} samples")
        
        # Create image paths and verify existence
        self._create_image_paths()
        
        # Prepare features and labels
        self._prepare_features_and_labels()
        
        # Create train/val/test splits with shared indices
        self._create_splits()
        
        print(f"Data loading complete:")
        print(f"  - Train samples: {len(self.train_indices)}")
        print(f"  - Validation samples: {len(self.val_indices)}")
        print(f"  - Test samples: {len(self.test_indices)}")
        print(f"  - Class distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
    
    def _preprocess_metadata(self):
        """Preprocess the metadata."""
        # Remove samples with unknown sex
        initial_count = len(self.df)
        self.df = self.df[self.df['sex'].isin(['male', 'female'])].copy()
        print(f"Removed {initial_count - len(self.df)} samples with unknown sex")
        
        # Encode sex as binary
        self.df['sex_encoded'] = self.df['sex'].map({'male': 1, 'female': 0})
        
        # Fill missing ages with median
        self.df['age'] = self.df['age'].fillna(self.df['age'].median())
        
        # Create age bins for sensitive attribute analysis
        self.df['age_bin'] = pd.cut(
            self.df['age'], 
            bins=[0, 30, 45, 60, 75, 120], 
            labels=[0, 1, 2, 3, 4], 
            include_lowest=True
        ).astype(int)
        
        # Encode localization
        localization_encoder = LabelEncoder()
        self.df['localization_encoded'] = localization_encoder.fit_transform(
            self.df['localization'].astype(str)
        )
        
        # Encode diagnosis labels
        self.label_encoder.fit(self.df['dx'])
        self.df['label_encoded'] = self.label_encoder.transform(self.df['dx'])
        
        print(f"Preprocessed metadata with {len(self.df)} valid samples")
        print(f"Diagnosis classes: {list(self.label_encoder.classes_)}")
    
    def _create_image_paths(self):
        """Create and verify image paths."""
        img_dir1 = os.path.join(self.data_dir, "HAM10000_images_part_1")
        img_dir2 = os.path.join(self.data_dir, "HAM10000_images_part_2")
        
        image_paths = []
        missing_images = 0
        
        for img_id in self.df['image_id']:
            path1 = os.path.join(img_dir1, f"{img_id}.jpg")
            path2 = os.path.join(img_dir2, f"{img_id}.jpg")
            
            if os.path.exists(path1):
                image_paths.append(path1)
            elif os.path.exists(path2):
                image_paths.append(path2)
            else:
                image_paths.append(None)
                missing_images += 1
        
        if missing_images > 0:
            print(f"Warning: {missing_images} images not found")
            # Remove rows with missing images
            valid_mask = [path is not None for path in image_paths]
            self.df = self.df[valid_mask].reset_index(drop=True)
            self.image_paths = [path for path in image_paths if path is not None]
        else:
            self.image_paths = image_paths
        
        print(f"Found {len(self.image_paths)} valid images")
    
    def _prepare_features_and_labels(self):
        """Prepare feature matrices and labels."""
        # Tabular features: age, sex, localization
        tabular_cols = ['age', 'sex_encoded', 'localization_encoded']
        self.tabular_features = self.df[tabular_cols].values.astype(np.float32)
        
        # Normalize tabular features
        self.tabular_features = self.scaler.fit_transform(self.tabular_features)
        
        # Labels
        self.labels = self.df['label_encoded'].values
        
        # Sensitive attributes for fairness analysis
        self.sensitive_attrs = self.df[['sex_encoded', 'age_bin']].values
    
    def _create_splits(self):
        """Create train/val/test splits with shared indices."""
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            range(len(self.df)),
            test_size=self.test_size,
            stratify=self.labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        train_labels_subset = self.labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_size,
            stratify=train_labels_subset,
            random_state=self.random_state
        )
        
        self.train_indices = np.array(train_idx)
        self.val_indices = np.array(val_idx)
        self.test_indices = np.array(test_idx)
    
    def get_image_client_data(self):
        """
        Get data for the image client.
        
        Returns:
            dict: Dictionary containing train/val/test splits for image client
        """
        return {
            'train': {
                'image_paths': np.array(self.image_paths)[self.train_indices],
                'labels': self.labels[self.train_indices],
                'sensitive_attrs': self.sensitive_attrs[self.train_indices],
                'indices': self.train_indices
            },
            'val': {
                'image_paths': np.array(self.image_paths)[self.val_indices],
                'labels': self.labels[self.val_indices],
                'sensitive_attrs': self.sensitive_attrs[self.val_indices],
                'indices': self.val_indices
            },
            'test': {
                'image_paths': np.array(self.image_paths)[self.test_indices],
                'labels': self.labels[self.test_indices],
                'sensitive_attrs': self.sensitive_attrs[self.test_indices],
                'indices': self.test_indices
            }
        }
    
    def get_tabular_client_data(self):
        """
        Get data for the tabular client.
        
        Returns:
            dict: Dictionary containing train/val/test splits for tabular client
        """
        return {
            'train': {
                'features': self.tabular_features[self.train_indices],
                'labels': self.labels[self.train_indices],
                'sensitive_attrs': self.sensitive_attrs[self.train_indices],
                'indices': self.train_indices
            },
            'val': {
                'features': self.tabular_features[self.val_indices],
                'labels': self.labels[self.val_indices],
                'sensitive_attrs': self.sensitive_attrs[self.val_indices],
                'indices': self.val_indices
            },
            'test': {
                'features': self.tabular_features[self.test_indices],
                'labels': self.labels[self.test_indices],
                'sensitive_attrs': self.sensitive_attrs[self.test_indices],
                'indices': self.test_indices
            }
        }
    
    def get_num_classes(self):
        """Get the number of classes in the dataset."""
        return len(self.label_encoder.classes_)
    
    def get_class_names(self):
        """Get the class names."""
        return list(self.label_encoder.classes_)
    
    def get_tabular_dim(self):
        """Get the dimensionality of tabular features."""
        return self.tabular_features.shape[1]


def create_data_generator(image_paths, tabular_features, labels, batch_size=32, 
                         target_size=(224, 224), augment=False, shuffle=True):
    """
    Create a data generator for training.
    
    Args:
        image_paths (np.ndarray): Array of image paths
        tabular_features (np.ndarray): Tabular feature array
        labels (np.ndarray): Label array
        batch_size (int): Batch size
        target_size (tuple): Target image size
        augment (bool): Whether to apply augmentation
        shuffle (bool): Whether to shuffle data
    
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    def generator():
        indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batch_images = []
            batch_tabular = []
            batch_labels = []
            
            for idx in batch_indices:
                # Load and preprocess image
                img = load_and_preprocess_image(
                    image_paths[idx], 
                    target_size=target_size, 
                    augment=augment
                )
                batch_images.append(img)
                batch_tabular.append(tabular_features[idx])
                batch_labels.append(labels[idx])
            
            yield (
                {'image_input': np.array(batch_images), 'tabular_input': np.array(batch_tabular)},
                np.array(batch_labels)
            )
    
    # Create dataset
    output_signature = (
        {
            'image_input': tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),
            'tabular_input': tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset 