# AI Music Generation with BiLSTM

This project implements a Deep Learning model to generate piano music using MIDI files. It uses a **Bi-Directional LSTM (BiLSTM)** neural network to predict both the **Pitch** (melody) and **Duration** (rhythm) of notes.

## Features
- **Dual-Output Model**: Predicts note pitch and duration simultaneously.
- **Rhythm Support**: Unlike simple models that only predict notes, this model captures the rhythmic structure of the music.
- **Temperature Sampling**: Allows for varied and creative generation by adjusting the randomness.
- **Class-Based Structure**: Modular `MusicGenerator` class for easy extension.

## Requirements
- Python 3.x
- TensorFlow / Keras
- Music21
- NumPy
- TQDM

Install dependencies:
```bash
pip install tensorflow music21 numpy tqdm
```

## Usage

### 1. Prepare Data
Place your MIDI files in the `All Midi Files` directory. The script supports recursive loading from subdirectories.

### 2. Train the Model
Run the script to parse MIDI files and train the model:
```bash
python auto_music_gen.py
```
By default, it trains for 5 epochs. You can adjust this in the `__main__` block.

### 3. Generate Music
After training, the script automatically generates a new MIDI file named `improved_music.mid`.
You can also customize the generation:
```python
gen = MusicGenerator()
gen.model = load_model("best_model.h5")
# Generate with higher temperature for more randomness
music = gen.generate_music(num_notes=300, temperature=1.2)
gen.save_midi(music, "my_creation.mid")
```

## Model Architecture
- **Inputs**: Sequence of Pitches, Sequence of Durations.
- **Embeddings**: Learned embeddings for both Pitch and Duration vocabularies.
- **Backbone**: 2-layer Bi-Directional LSTM with Dropout.
- **Outputs**:
    - `pitch_head`: Softmax over pitch vocabulary.
    - `duration_head`: Softmax over duration vocabulary.
