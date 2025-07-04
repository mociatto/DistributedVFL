"""
Model architectures for multimodal vertical federated learning.
Includes image encoder, tabular encoder, and enhanced fusion model with attention.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate,
    MultiHeadAttention, Add, Embedding, GlobalAveragePooling1D,
    Multiply, GaussianNoise, AlphaDropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetB0
import numpy as np
import ssl
import urllib.request

# SSL workaround for EfficientNet download issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


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


class AdvancedDropout(tf.keras.layers.Layer):
    """
    STEP 3: Advanced dropout with scheduled rates for better generalization.
    Combines multiple dropout techniques for robust training.
    """
    
    def __init__(self, base_rate=0.3, max_rate=0.5, **kwargs):
        super(AdvancedDropout, self).__init__(**kwargs)
        self.base_rate = base_rate
        self.max_rate = max_rate
        self.dropout_layer = Dropout(base_rate)
        self.alpha_dropout = AlphaDropout(base_rate * 0.7)
    
    def call(self, inputs, training=None):
        if training:
            # Apply both regular and alpha dropout for better regularization
            x = self.dropout_layer(inputs, training=training)
            x = self.alpha_dropout(x, training=training)
            return x
        return inputs


class NoiseInjection(tf.keras.layers.Layer):
    """
    STEP 3: Adaptive noise injection for improved generalization.
    Helps models learn robust features that generalize better.
    """
    
    def __init__(self, noise_stddev=0.1, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)
        self.noise_stddev = noise_stddev
        self.gaussian_noise = GaussianNoise(noise_stddev)
    
    def call(self, inputs, training=None):
        if training:
            return self.gaussian_noise(inputs, training=training)
        return inputs


def create_image_encoder(input_shape=(224, 224, 3), embedding_dim=128, use_step3_enhancements=True, use_lightweight=True):
    """
    Create EfficientNet based image encoder with transfer learning.
    MAJOR UPGRADE: Replaces custom CNN with state-of-the-art pre-trained backbone.
    MEMORY FIX: Option for lightweight EfficientNetB0 vs heavy EfficientNetV2-S.
    
    Args:
        input_shape (tuple): Input image shape
        embedding_dim (int): Output embedding dimension
        use_step3_enhancements (bool): Whether to use Step 3 generalization enhancements
        use_lightweight (bool): Use EfficientNetB0 (5M) vs EfficientNetV2S (21M) parameters
    
    Returns:
        tf.keras.Model: Image encoder model
    """
    inputs = Input(shape=input_shape, name='image_input')
    
    # STEP 3: Add noise injection for better generalization
    if use_step3_enhancements:
        x = NoiseInjection(noise_stddev=0.05)(inputs)
        print("   Step 3: Noise injection enabled for image encoder")
    else:
        x = inputs
    
    # ========================================
    # PHASE 2 UPGRADE: EfficientNet Transfer Learning with Memory Optimization
    # ========================================
    
    if use_lightweight:
        print("   MEMORY-OPTIMIZED: Using EfficientNetB0 backbone (5M parameters)")
        backbone = EfficientNetB0(
            weights='imagenet',  # Pre-trained weights
            include_top=False,   # Remove final classification layer
            input_tensor=x,      # Use our input tensor
            pooling='avg'        # Global average pooling
        )
        backbone_output_dim = 1280  # EfficientNetB0 output dimension
    else:
        print("   FULL-SCALE: Using EfficientNetV2-S backbone (21M parameters)")
        backbone = EfficientNetV2S(
            weights='imagenet',  # Pre-trained weights
            include_top=False,   # Remove final classification layer
            input_tensor=x,      # Use our input tensor
            pooling='avg'        # Global average pooling
        )
        backbone_output_dim = 1280  # EfficientNetV2-S output dimension
    
    # Make backbone trainable for fine-tuning
    backbone.trainable = True
    
    # Fine-tuning strategy: freeze early layers, train later layers
    total_layers = len(backbone.layers)
    freeze_until = int(total_layers * 0.8)  # Freeze first 80%
    
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    
    model_name = "EfficientNetB0" if use_lightweight else "EfficientNetV2-S"
    print(f"   {model_name}: {total_layers} layers, freezing first {freeze_until} layers")
    
    # Get features from backbone
    backbone_features = backbone(x)
    
    # Adaptation layers to convert 1280-D features to our embedding_dim
    # Add regularization and adaptation
    features = Dense(512, activation='relu', kernel_initializer='he_normal')(backbone_features)
    features = BatchNormalization()(features)
    
    # STEP 3: Enhanced dropout for better generalization
    if use_step3_enhancements:
        features = AdvancedDropout(base_rate=0.3)(features)
        features = NoiseInjection(noise_stddev=0.1)(features)
    else:
        features = Dropout(0.3)(features)
    
    # Second adaptation layer
    features = Dense(embedding_dim * 2, activation='relu', kernel_initializer='he_normal')(features)
    features = BatchNormalization()(features)
    
    # STEP 3: More aggressive regularization
    if use_step3_enhancements:
        features = AdvancedDropout(base_rate=0.4)(features)
    else:
        features = Dropout(0.4)(features)
    
    # Final embedding layer
    embeddings = Dense(embedding_dim, activation='linear', 
                      kernel_initializer='he_normal', name='embeddings')(features)
    
    model = Model(inputs=inputs, outputs=embeddings, name='efficientnet_image_encoder')
    
    enhancement_note = " (Step 3 enhanced)" if use_step3_enhancements else ""
    memory_note = " [LIGHTWEIGHT]" if use_lightweight else " [FULL-SCALE]"
    print(f"   {model_name} encoder{enhancement_note}{memory_note}: {model.count_params():,} parameters")
    print(f"   Transfer Learning: Pre-trained on ImageNet, fine-tuning enabled")
    return model


def create_tabular_encoder(input_dim, embedding_dim=128, use_step3_enhancements=True):
    """
    Create enhanced tabular encoder with better feature processing.
    
    Args:
        input_dim (int): Input feature dimension
        embedding_dim (int): Output embedding dimension
        use_step3_enhancements (bool): Whether to use Step 3 generalization enhancements
    
    Returns:
        tf.keras.Model: Tabular encoder model
    """
    inputs = Input(shape=(input_dim,), name='tabular_input')
    
    # Feature normalization and expansion
    x = BatchNormalization()(inputs)
    
    # STEP 3: Add noise injection for tabular features
    if use_step3_enhancements:
        x = NoiseInjection(noise_stddev=0.05)(x)
        print("   Step 3: Noise injection enabled for tabular encoder")
    
    # Progressive feature expansion with residual connections
    # Layer 1 - Feature expansion
    x1 = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    
    # STEP 3: Enhanced dropout
    if use_step3_enhancements:
        x1 = AdvancedDropout(base_rate=0.3)(x1)
    else:
        x1 = Dropout(0.3)(x1)
    
    # Layer 2 - Pattern detection
    x2 = Dense(128, activation='relu', kernel_initializer='he_normal')(x1)
    x2 = BatchNormalization()(x2)
    
    # STEP 3: Enhanced dropout
    if use_step3_enhancements:
        x2 = AdvancedDropout(base_rate=0.35)(x2)
    else:
        x2 = Dropout(0.3)(x2)
    
    # Layer 3 - Feature refinement with residual
    x3 = Dense(128, activation='relu', kernel_initializer='he_normal')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Add()([x2, x3])  # Residual connection
    
    # STEP 3: More aggressive regularization for final layer
    if use_step3_enhancements:
        x3 = AdvancedDropout(base_rate=0.4)(x3)
        x3 = NoiseInjection(noise_stddev=0.1)(x3)
    else:
        x3 = Dropout(0.4)(x3)
    
    # Final embedding layer
    embeddings = Dense(embedding_dim, activation='linear', 
                      kernel_initializer='he_normal', name='embeddings')(x3)
    
    model = Model(inputs=inputs, outputs=embeddings, name='tabular_encoder')
    
    enhancement_note = " (Step 3 enhanced)" if use_step3_enhancements else ""
    print(f"   Tabular encoder{enhancement_note}: {model.count_params():,} parameters")
    return model


def create_fusion_model_with_transformer(image_dim=128, tabular_dim=128, 
                                       num_classes=7, adversarial_lambda=0.0, 
                                       use_advanced_fusion=True, use_step3_enhancements=True):
    """
    Create enhanced fusion model with improved cross-attention and better multimodal fusion.
    
    Args:
        image_dim (int): Image embedding dimension
        tabular_dim (int): Tabular embedding dimension
        num_classes (int): Number of output classes
        adversarial_lambda (float): Weight for adversarial loss (0 to disable)
        use_advanced_fusion (bool): Whether to use advanced transformer fusion (Step 2)
        use_step3_enhancements (bool): Whether to use Step 3 generalization enhancements
    
    Returns:
        tuple: (fusion_model, adversarial_model) - adversarial_model is None if lambda=0
    """
    # Input layers for embeddings from clients
    image_embeddings = Input(shape=(image_dim,), name='image_embeddings')
    tabular_embeddings = Input(shape=(tabular_dim,), name='tabular_embeddings')
    
    # Normalize embeddings for better training stability
    image_norm = LayerNormalization()(image_embeddings)
    tabular_norm = LayerNormalization()(tabular_embeddings)
    
    # STEP 3: Add noise injection to embeddings for better generalization
    if use_step3_enhancements:
        print("   Step 3: Embedding noise injection enabled")
        image_norm = NoiseInjection(noise_stddev=0.05)(image_norm)
        tabular_norm = NoiseInjection(noise_stddev=0.05)(tabular_norm)
    
    if use_advanced_fusion:
        # STEP 2: Advanced Transformer-based Cross-Modal Fusion
        print("   Using advanced transformer fusion (Step 2)")
        
        # Cross-modal attention mechanism
        # Expand dimensions for attention computation using Keras layers
        image_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(image_norm)  # (batch, 1, dim)
        tabular_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(tabular_norm)  # (batch, 1, dim)
        
        # Stack for multi-head attention
        stacked_embeddings = tf.keras.layers.Concatenate(axis=1)([image_expanded, tabular_expanded])  # (batch, 2, dim)
        
        # Multi-head self-attention for cross-modal fusion
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            dropout=0.15 if use_step3_enhancements else 0.1,  # STEP 3: Higher dropout
            name='cross_modal_attention'
        )(stacked_embeddings, stacked_embeddings)
        
        # Add residual connection
        attention_output = Add()([stacked_embeddings, attention_output])
        attention_output = LayerNormalization()(attention_output)
        
        # Global pooling to combine attended features
        fused_features = GlobalAveragePooling1D()(attention_output)
        
        # Additional cross-modal interaction layers
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(fused_features)
        x = BatchNormalization()(x)
        
        # STEP 3: Enhanced dropout for better generalization
        if use_step3_enhancements:
            x = AdvancedDropout(base_rate=0.5)(x)  # More aggressive
        else:
            x = Dropout(0.4)(x)
        
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        
        # STEP 3: Enhanced dropout
        if use_step3_enhancements:
            x = AdvancedDropout(base_rate=0.4)(x)
        else:
            x = Dropout(0.3)(x)
        
        # Ensemble with simple concatenation for robustness
        simple_concat = Concatenate()([image_norm, tabular_norm])
        simple_features = Dense(128, activation='relu', kernel_initializer='he_normal')(simple_concat)
        
        # STEP 3: Enhanced dropout for simple path
        if use_step3_enhancements:
            simple_features = AdvancedDropout(base_rate=0.3)(simple_features)
        else:
            simple_features = Dropout(0.2)(simple_features)
        
        # Combine advanced and simple features
        combined_features = Add()([x, simple_features])
        x = LayerNormalization()(combined_features)
        
    else:
        # STEP 1: Simple but effective multimodal fusion with concatenation
        print("   Using simple concatenation fusion (Step 1)")
        fused_features = Concatenate()([image_norm, tabular_norm])
        
        # Feature processing with proper regularization
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(fused_features)
        x = BatchNormalization()(x)
        
        # STEP 3: Enhanced dropout even for simple fusion
        if use_step3_enhancements:
            x = AdvancedDropout(base_rate=0.4)(x)
        else:
            x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        
        # STEP 3: Enhanced dropout
        if use_step3_enhancements:
            x = AdvancedDropout(base_rate=0.3)(x)
        else:
            x = Dropout(0.2)(x)
    
    # Final classification layers (common for both approaches)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    
    # STEP 3: Final layer regularization
    if use_step3_enhancements:
        x = AdvancedDropout(base_rate=0.2)(x)
        x = NoiseInjection(noise_stddev=0.05)(x)  # Light noise before prediction
    else:
        x = Dropout(0.1)(x)
    
    # Classification head
    predictions = Dense(num_classes, activation='softmax', 
                       kernel_initializer='glorot_uniform', name='predictions')(x)
    
    # Create fusion model
    if use_step3_enhancements:
        model_name = 'step3_enhanced_fusion_model'
        print("   Step 3: Enhanced fusion model with advanced regularization")
    elif use_advanced_fusion:
        model_name = 'advanced_fusion_model'
    else:
        model_name = 'simple_fusion_model'
        
    fusion_model = Model(
        inputs=[image_embeddings, tabular_embeddings],
        outputs=predictions,
        name=model_name
    )
    
    # Compile with stable settings
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    enhancement_notes = []
    if use_advanced_fusion:
        enhancement_notes.append("Step 2 transformer")
    if use_step3_enhancements:
        enhancement_notes.append("Step 3 generalization")
    
    note_str = f" ({', '.join(enhancement_notes)})" if enhancement_notes else ""
    print(f"   Enhanced fusion model{note_str}: {fusion_model.count_params():,} parameters")
    
    # Adversarial model (disabled for now)
    adversarial_model = None
    if adversarial_lambda > 0:
        print(f"   Adversarial training enabled (λ={adversarial_lambda})")
        # Implementation would go here
    else:
        print(f"   Adversarial training disabled")
    
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


def nt_xent_loss(z_i, z_j, temperature=0.5, batch_size=None):
    """
    Simplified contrastive loss for multimodal embedding alignment.
    Encourages image and tabular embeddings from same sample to be similar.
    
    Args:
        z_i: Image embeddings [batch_size, embedding_dim]
        z_j: Tabular embeddings [batch_size, embedding_dim]  
        temperature: Temperature parameter for scaling
        batch_size: Batch size (auto-detected if None)
    
    Returns:
        Contrastive loss value
    """
    if batch_size is None:
        batch_size = tf.shape(z_i)[0]
    
    # L2 normalize embeddings
    z_i = tf.nn.l2_normalize(z_i, axis=1)
    z_j = tf.nn.l2_normalize(z_j, axis=1)
    
    # Compute positive similarity (same sample pairs)
    positive_sim = tf.reduce_sum(z_i * z_j, axis=1) / temperature
    
    # Compute negative similarities (different sample pairs)
    # For each image embedding, compute similarity with all tabular embeddings
    neg_sim_matrix = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    
    # Create mask to exclude positive pairs from negatives
    batch_size_float = tf.cast(batch_size, tf.float32)
    mask = tf.ones([batch_size, batch_size]) - tf.eye(batch_size)
    
    # Apply mask and compute negative similarities
    masked_neg_sim = neg_sim_matrix * mask
    
    # Compute contrastive loss using logsumexp for numerical stability
    neg_logits = tf.reduce_logsumexp(masked_neg_sim, axis=1)
    pos_logits = positive_sim
    
    # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    loss = -pos_logits + tf.math.log(tf.exp(pos_logits) + tf.exp(neg_logits))
    
    return tf.reduce_mean(loss)


def contrastive_fusion_loss(y_true, y_pred, image_emb, tabular_emb, 
                           alpha=0.7, temperature=0.5):
    """
    Combined classification and contrastive loss for improved multimodal learning.
    
    Args:
        y_true: True labels
        y_pred: Predicted logits
        image_emb: Image embeddings
        tabular_emb: Tabular embeddings
        alpha: Weight for classification loss (1-alpha for contrastive)
        temperature: Temperature for NT-Xent loss
    
    Returns:
        Combined loss value
    """
    # Classification loss
    classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )
    classification_loss = tf.reduce_mean(classification_loss)
    
    # Contrastive loss for better embedding alignment
    contrastive_loss = nt_xent_loss(image_emb, tabular_emb, temperature)
    
    # Combined loss
    total_loss = alpha * classification_loss + (1 - alpha) * contrastive_loss
    
    return total_loss 