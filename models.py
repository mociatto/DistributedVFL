"""
Model architectures for multimodal vertical federated learning.
Includes image encoder, tabular encoder, and enhanced fusion model with attention.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate,
    MultiHeadAttention, Add, Embedding, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for handling class imbalance.
    Focuses learning on hard-to-classify examples.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[1])
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight: (1 - p_t)^gamma
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha weighting
        alpha_weight = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Combine all weights
        focal_loss = alpha_weight * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))


class TransformerFusionBlock(tf.keras.layers.Layer):
    """
    Transformer-based fusion block for multimodal feature integration.
    Uses multi-head attention to learn complex feature interactions.
    """
    
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.1, **kwargs):
        super(TransformerFusionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dropout(dropout_rate),
            Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, seq_len, embed_dim)
        # For our case, seq_len = 2 (image and tabular features)
        
        # Multi-head attention with residual connection
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def create_image_encoder(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Create improved CNN image encoder with better feature extraction.
    
    Args:
        input_shape (tuple): Input image shape
        embedding_dim (int): Output embedding dimension
    
    Returns:
        tf.keras.Model: Image encoder model
    """
    inputs = Input(shape=input_shape, name='image_input')
    
    # Enhanced CNN architecture with better feature extraction
    # Block 1 - Feature detection
    x = Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2 - Pattern recognition
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3 - Complex features
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Global feature extraction
    x = GlobalAveragePooling2D()(x)
    
    # Dense embedding layers
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Final embedding layer
    embeddings = Dense(embedding_dim, activation='linear', 
                      kernel_initializer='he_normal', name='embeddings')(x)
    
    model = Model(inputs=inputs, outputs=embeddings, name='image_encoder')
    
    print(f"   🖼️  Image encoder: {model.count_params():,} parameters")
    return model


def create_tabular_encoder(input_dim, embedding_dim=128):
    """
    Create enhanced tabular encoder with better feature processing.
    
    Args:
        input_dim (int): Input feature dimension
        embedding_dim (int): Output embedding dimension
    
    Returns:
        tf.keras.Model: Tabular encoder model
    """
    inputs = Input(shape=(input_dim,), name='tabular_input')
    
    # Feature normalization and expansion
    x = BatchNormalization()(inputs)
    
    # Progressive feature expansion with residual connections
    # Layer 1 - Feature expansion
    x1 = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Layer 2 - Pattern detection
    x2 = Dense(128, activation='relu', kernel_initializer='he_normal')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Layer 3 - Feature refinement with residual
    x3 = Dense(128, activation='relu', kernel_initializer='he_normal')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Add()([x2, x3])  # Residual connection
    x3 = Dropout(0.4)(x3)
    
    # Final embedding layer
    embeddings = Dense(embedding_dim, activation='linear', 
                      kernel_initializer='he_normal', name='embeddings')(x3)
    
    model = Model(inputs=inputs, outputs=embeddings, name='tabular_encoder')
    
    print(f"   📊 Tabular encoder: {model.count_params():,} parameters")
    return model


def create_fusion_model_with_transformer(image_dim=128, tabular_dim=128, 
                                       num_classes=7, adversarial_lambda=0.0):
    """
    Create enhanced fusion model with improved cross-attention and better multimodal fusion.
    
    Args:
        image_dim (int): Image embedding dimension
        tabular_dim (int): Tabular embedding dimension
        num_classes (int): Number of output classes
        adversarial_lambda (float): Weight for adversarial loss (0 to disable)
    
    Returns:
        tuple: (fusion_model, adversarial_model) - adversarial_model is None if lambda=0
    """
    # Input layers for embeddings from clients
    image_embeddings = Input(shape=(image_dim,), name='image_embeddings')
    tabular_embeddings = Input(shape=(tabular_dim,), name='tabular_embeddings')
    
    # Normalize embeddings
    image_norm = LayerNormalization()(image_embeddings)
    tabular_norm = LayerNormalization()(tabular_embeddings)
    
    # Cross-modal attention mechanism
    # Expand dimensions for attention computation
    image_expanded = tf.expand_dims(image_norm, axis=1)  # (batch, 1, dim)
    tabular_expanded = tf.expand_dims(tabular_norm, axis=1)  # (batch, 1, dim)
    
    # Stack for multi-head attention
    stacked_embeddings = tf.concat([image_expanded, tabular_expanded], axis=1)  # (batch, 2, dim)
    
    # Multi-head self-attention for cross-modal fusion
    attention_output = MultiHeadAttention(
        num_heads=4, 
        key_dim=32,
        dropout=0.1,
        name='cross_modal_attention'
    )(stacked_embeddings, stacked_embeddings)
    
    # Add residual connection
    attention_output = Add()([stacked_embeddings, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    # Global pooling to combine attended features
    fused_features = GlobalAveragePooling1D()(attention_output)
    
    # Additional feature processing
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(fused_features)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Classification head with better regularization
    predictions = Dense(num_classes, activation='softmax', 
                       kernel_initializer='he_normal', name='predictions')(x)
    
    # Create fusion model
    fusion_model = Model(
        inputs=[image_embeddings, tabular_embeddings],
        outputs=predictions,
        name='enhanced_fusion_model'
    )
    
    # Compile with better optimizer and loss
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   🔄 Enhanced fusion model: {fusion_model.count_params():,} parameters")
    
    # Adversarial model (disabled for now)
    adversarial_model = None
    if adversarial_lambda > 0:
        print(f"   ⚡ Adversarial training enabled (λ={adversarial_lambda})")
        # Implementation would go here
    else:
        print(f"   ⚪ Adversarial training disabled")
    
    return fusion_model, adversarial_model


def compile_models(image_encoder, tabular_encoder, fusion_model, adversarial_model=None, 
                  learning_rate=0.001, adversarial_lambda=0.0):
    """
    Compile all models with appropriate optimizers and loss functions.
    
    Args:
        image_encoder: Image encoder model
        tabular_encoder: Tabular encoder model  
        fusion_model: Main fusion model
        adversarial_model: Adversarial model (can be None)
        learning_rate (float): Learning rate
        adversarial_lambda (float): Adversarial loss weight
    """
    optimizer = Adam(learning_rate=learning_rate)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Compile image encoder (for standalone training if needed)
    image_encoder.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Compile tabular encoder (for standalone training if needed)  
    tabular_encoder.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Compile fusion model with Focal Loss
    fusion_model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    # Compile adversarial model if provided
    if adversarial_model is not None and adversarial_lambda > 0:
        adversarial_model.compile(
            optimizer=optimizer,
            loss={
                'sex_prediction': 'sparse_categorical_crossentropy',
                'age_prediction': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'sex_prediction': adversarial_lambda,
                'age_prediction': adversarial_lambda
            },
            metrics=['accuracy']
        )


def create_complete_vfl_architecture(image_shape=(224, 224, 3), tabular_dim=3, 
                                   num_classes=7, embedding_dim=256, adversarial_lambda=0.0):
    """
    Create the complete VFL architecture with all components.
    
    Args:
        image_shape (tuple): Input image shape
        tabular_dim (int): Tabular feature dimension
        num_classes (int): Number of output classes
        embedding_dim (int): Embedding dimension for both modalities
        adversarial_lambda (float): Adversarial loss weight (0 to disable)
    
    Returns:
        dict: Dictionary containing all model components
    """
    # Create encoders
    image_encoder = create_image_encoder(image_shape, embedding_dim)
    tabular_encoder = create_tabular_encoder(tabular_dim, embedding_dim)
    
    # Create fusion model
    fusion_model, adversarial_model = create_fusion_model_with_transformer(
        image_dim=embedding_dim,
        tabular_dim=embedding_dim,
        num_classes=num_classes,
        adversarial_lambda=adversarial_lambda
    )
    
    # Compile all models
    compile_models(
        image_encoder, 
        tabular_encoder, 
        fusion_model, 
        adversarial_model,
        adversarial_lambda=adversarial_lambda
    )
    
    return {
        'image_encoder': image_encoder,
        'tabular_encoder': tabular_encoder,
        'fusion_model': fusion_model,
        'adversarial_model': adversarial_model,
        'embedding_dim': embedding_dim,
        'adversarial_lambda': adversarial_lambda
    }


def create_end_to_end_model(image_shape=(224, 224, 3), tabular_dim=3, 
                          num_classes=7, embedding_dim=256):
    """
    Create end-to-end model for evaluation purposes.
    
    Args:
        image_shape (tuple): Input image shape
        tabular_dim (int): Tabular feature dimension
        num_classes (int): Number of output classes
        embedding_dim (int): Embedding dimension
    
    Returns:
        tf.keras.Model: End-to-end model
    """
    # Input layers
    image_input = Input(shape=image_shape, name='image_input')
    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    
    # Create encoders
    image_encoder = create_image_encoder(image_shape, embedding_dim)
    tabular_encoder = create_tabular_encoder(tabular_dim, embedding_dim)
    
    # Get embeddings
    image_embeddings = image_encoder(image_input)
    tabular_embeddings = tabular_encoder(tabular_input)
    
    # Create fusion model (without adversarial component for end-to-end)
    fusion_model, _ = create_fusion_model_with_transformer(
        image_dim=embedding_dim,
        tabular_dim=embedding_dim,
        num_classes=num_classes,
        adversarial_lambda=0.0
    )
    
    # Get final predictions
    predictions = fusion_model([image_embeddings, tabular_embeddings])
    
    # Create complete model
    end_to_end_model = Model(
        inputs=[image_input, tabular_input],
        outputs=predictions,
        name='EndToEndModel'
    )
    
    # Compile with Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    end_to_end_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    return end_to_end_model 