import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
import json
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer

def load_dataset(path):
    df = pd.read_parquet(path)
    return df['question'].tolist(), df['answer'].tolist()

def format_prompt(question, answer=None):
    if answer:
        return f"Question: {question}\nAnswer: {answer}"
    return f"Question: {question}"

def main():
    print("Loading datasets...")
    train_questions, train_answers = load_dataset('dataset/train-00000-of-00001.parquet')
    test_questions, test_answers = load_dataset('dataset/test-00000-of-00001.parquet')
    
    print("\nDataset info:")
    print(f"Train size: {len(train_questions)}")
    print(f"Test size: {len(test_questions)}")
    
    print("\nExample from training data:")
    print(f"Question: {train_questions[0]}")
    print(f"Answer: {train_answers[0]}")

    print("\nInitializing tokenizer...")
    with open('tokenizer.json', 'r') as f:
        tokenizer_config = json.load(f)
    
    # Extract and simplify tokenizer config
    tokenizer_cfg = tokenizer_config['config'].copy()
    tokenizer_cfg['dtype'] = 'int32'  # Simplify the dtype configuration
    tokenizer = GemmaTokenizer(**tokenizer_cfg)
    
    print("\nLoading Gemma model...")
    with open('config.json', 'r') as f:
        model_config = json.load(f)
    
    # Initialize model with config
    model = GemmaBackbone(**model_config['config'])
    
    # Load pre-trained weights
    model.load_weights('model.weights.h5')
    
    # Prepare training data
    print("\nPreparing training data...")
    train_texts = [format_prompt(q, a) for q, a in zip(train_questions, train_answers)]
    test_texts = [format_prompt(q, a) for q, a in zip(test_questions, test_answers)]
    
    # Tokenize the data
    max_length = 512  # Adjust based on your needs
    train_tokens = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    test_tokens = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    
    # Create training configuration
    training_config = keras.optimizers.experimental.AdamConfig(
        learning_rate=2e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.experimental.Adam(training_config),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create checkpoint callback
    checkpoint_path = "checkpoints/model_{epoch:02d}.h5"
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_tokens['input_ids'],
        train_tokens['input_ids'][:, 1:],  # Shift right for next token prediction
        validation_data=(
            test_tokens['input_ids'],
            test_tokens['input_ids'][:, 1:]
        ),
        epochs=3,
        batch_size=4,
        callbacks=[checkpoint_callback]
    )
    
    print("\nTraining completed!")
    
    # Save the final model
    print("\nSaving model...")
    os.makedirs("final_model", exist_ok=True)
    model.save_weights('final_model/model.weights.h5')
    print("Model saved to final_model/model.weights.h5")

if __name__ == "__main__":
    main()
