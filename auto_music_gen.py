import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

class MusicGenerator:
    def __init__(self, seq_length=50):
        self.seq_length = seq_length
        self.notes = []
        self.durations = []
        self.pitch_vocab = {}
        self.duration_vocab = {}
        self.rev_pitch_vocab = {}
        self.rev_duration_vocab = {}
        self.model = None

    def parse_midi_files(self, file_pattern):
        """Parses MIDI files to extract pitch and duration."""
        files = glob.glob(file_pattern, recursive=True)
        print(f"Found {len(files)} files.")
        
        all_notes = []
        all_durations = []

        for file in tqdm(files, desc="Parsing MIDI"):
            try:
                midi = converter.parse(file)
                notes_to_parse = None
                
                try: # file has instrument parts
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse() 
                except: # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        all_notes.append(str(element.pitch))
                        all_durations.append(element.duration.quarterLength)
                    elif isinstance(element, chord.Chord):
                        all_notes.append('.'.join(str(n) for n in element.normalOrder))
                        all_durations.append(element.duration.quarterLength)
            except Exception as e:
                print(f"Error parsing {file}: {e}")

        self.notes = all_notes
        self.durations = all_durations
        print(f"Total notes parsed: {len(self.notes)}")

    def prepare_sequences(self):
        """Prepares sequences for training."""
        # Create Vocabularies
        unique_notes = sorted(list(set(self.notes)))
        unique_durations = sorted(list(set(self.durations)))
        
        self.pitch_vocab = {n: i for i, n in enumerate(unique_notes)}
        self.duration_vocab = {d: i for i, d in enumerate(unique_durations)}
        
        self.rev_pitch_vocab = {i: n for n, i in self.pitch_vocab.items()}
        self.rev_duration_vocab = {i: d for d, i in self.duration_vocab.items()}

        print(f"Pitch Vocab Size: {len(self.pitch_vocab)}")
        print(f"Duration Vocab Size: {len(self.duration_vocab)}")

        network_input_pitch = []
        network_input_duration = []
        network_output_pitch = []
        network_output_duration = []

        for i in range(0, len(self.notes) - self.seq_length):
            p_seq = self.notes[i:i + self.seq_length]
            d_seq = self.durations[i:i + self.seq_length]
            
            p_out = self.notes[i + self.seq_length]
            d_out = self.durations[i + self.seq_length]

            network_input_pitch.append([self.pitch_vocab[char] for char in p_seq])
            network_input_duration.append([self.duration_vocab[char] for char in d_seq])
            
            network_output_pitch.append(self.pitch_vocab[p_out])
            network_output_duration.append(self.duration_vocab[d_out])

        n_patterns = len(network_input_pitch)
        
        # Reshape and normalize input
        self.X_pitch = np.reshape(network_input_pitch, (n_patterns, self.seq_length))
        self.X_duration = np.reshape(network_input_duration, (n_patterns, self.seq_length))
        
        self.y_pitch = to_categorical(network_output_pitch, num_classes=len(self.pitch_vocab))
        self.y_duration = to_categorical(network_output_duration, num_classes=len(self.duration_vocab))

        return self.X_pitch, self.X_duration, self.y_pitch, self.y_duration

    def build_model(self):
        """Builds a multi-output LSTM model."""
        pitch_input = Input(shape=(self.seq_length,))
        dur_input = Input(shape=(self.seq_length,))

        # Embeddings
        pitch_embed = Embedding(len(self.pitch_vocab), 100)(pitch_input)
        dur_embed = Embedding(len(self.duration_vocab), 100)(dur_input)

        # Concatenate
        merge = Concatenate()([pitch_embed, dur_embed])

        # LSTM Backbone
        lstm_1 = LSTM(256, return_sequences=True)(merge)
        dropout_1 = Dropout(0.3)(lstm_1)
        lstm_2 = LSTM(256)(dropout_1)
        dropout_2 = Dropout(0.3)(lstm_2)

        # Output Heads
        pitch_dense = Dense(256, activation='relu')(dropout_2)
        pitch_out = Dense(len(self.pitch_vocab), activation='softmax', name='pitch')(pitch_dense)

        dur_dense = Dense(256, activation='relu')(dropout_2)
        dur_out = Dense(len(self.duration_vocab), activation='softmax', name='duration')(dur_dense)

        self.model = Model(inputs=[pitch_input, dur_input], outputs=[pitch_out, dur_out])
        self.model.compile(
            loss={'pitch': 'categorical_crossentropy', 'duration': 'categorical_crossentropy'},
            optimizer='rmsprop',
            metrics={'pitch': 'accuracy', 'duration': 'accuracy'}
        )
        self.model.summary()

    def train(self, epochs=50, batch_size=64):
        """Trains the model."""
        checkpoint = ModelCheckpoint(
            "best_model.h5", monitor='loss', verbose=0, save_best_only=True, mode='min'
        )
        self.model.fit(
            [self.X_pitch, self.X_duration],
            [self.y_pitch, self.y_duration],
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint]
        )

    def sample(self, preds, temperature=1.0):
        """Helper function to sample an index from a probability array."""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature  # Add epsilon to avoid log(0)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_music(self, num_notes=200, temperature=0.8):
        """Generates music using the trained model."""
        start = np.random.randint(0, len(self.X_pitch)-1)
        
        pattern_pitch = self.X_pitch[start]
        pattern_dur = self.X_duration[start]
        
        prediction_output = []

        for note_index in tqdm(range(num_notes), desc="Generating"):
            input_pitch = np.reshape(pattern_pitch, (1, len(pattern_pitch)))
            input_dur = np.reshape(pattern_dur, (1, len(pattern_dur)))

            pitch_pred, dur_pred = self.model.predict([input_pitch, input_dur], verbose=0)

            # Use temperature sampling instead of argmax
            index_pitch = self.sample(pitch_pred[0], temperature)
            index_dur = self.sample(dur_pred[0], temperature)

            result_pitch = self.rev_pitch_vocab[index_pitch]
            result_dur = self.rev_duration_vocab[index_dur]
            
            prediction_output.append((result_pitch, result_dur))

            pattern_pitch = np.append(pattern_pitch, index_pitch)
            pattern_pitch = pattern_pitch[1:]
            
            pattern_dur = np.append(pattern_dur, index_dur)
            pattern_dur = pattern_dur[1:]

        return prediction_output

    def save_midi(self, prediction_output, filename='generated_music.mid'):
        """Converts prediction output to MIDI file."""
        offset = 0
        output_notes = []

        for pattern, duration in prediction_output:
            # Handle Chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                new_chord.duration.quarterLength = duration
                output_notes.append(new_chord)
            # Handle Note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                new_note.duration.quarterLength = duration
                output_notes.append(new_note)

            offset += duration

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=filename)
        print(f"Music saved to {filename}")

if __name__ == "__main__":
    # Example Usage
    gen = MusicGenerator()
    
    # 1. Parse
    gen.parse_midi_files("All Midi Files/**/*.mid")
    
    # 2. Prepare
    gen.prepare_sequences()
    
    # 3. Build
    gen.build_model()
    
    # 4. Train (Use small epochs for testing)
    gen.train(epochs=5, batch_size=64) 
    
    # 5. Generate
    music = gen.generate_music()
    gen.save_midi(music, "improved_music.mid")
