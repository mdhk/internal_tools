import numpy as np
import pandas as pd
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class AnnotatedAudioDataset(Dataset):
    def __init__(self, 
                 annotations: str | Path | pd.DataFrame, 
                 audio_dir: str | Path, 
                 start_time_column: str = 'start_time', 
                 end_time_column: str = 'end_time', 
                 file_id_column: str = 'file_id', 
                 file_format: str = 'wav'
                ):
        self.audio_dir = Path(audio_dir)
        self._file_format = file_format

        if not isinstance(annotations, pd.DataFrame):
            annotations = pd.read_csv(annotations)

        for col in [start_time_column, end_time_column, file_id_column]:
            assert col in annotations.columns, \
                f"Annotations DataFrame has no column '{col}'. " +\
                "Specify start_time_column, end_time_column, and file_id_column with column names " +\
                "from the provided annotations file when initializing AnnotatedAudioDataset."
            
        file_ids = annotations[file_id_column].unique()
        audio_paths = [
            self.audio_dir / f'{fid}.{self._file_format}' for fid in file_ids
        ]

        # retrieve audio durations for sorting
        audio_durations = {}
        sampling_rates = []
        for f, fid in enumerate(file_ids):
            audio_path = audio_paths[f]
            assert audio_path.exists(), f"Cannot load audio for file id {fid}: " +\
            f"{str(audio_path)} does not exist"
            audio_info = sf.info(audio_path)
            audio_durations[fid] = audio_info.duration
            sampling_rates.append(audio_info.samplerate)
            
        assert len(np.unique(sampling_rates)) == 1, "All audio files should have the same sampling rate, " +\
        f"but found multiple: {np.unique(sampling_rates).tolist()}"

        # sort files by duration for batch processing (longest first)
        annotations['audio_file_duration'] = annotations[file_id_column].apply(lambda fid: audio_durations[fid])
        annotations = annotations.sort_values(by='audio_file_duration', ascending=False)

        # store sorted file ids + annotations
        self._file_ids = annotations[file_id_column].unique()
        self._audio_paths = [
            self.audio_dir / f'{fid}.{self._file_format}' for fid in self._file_ids
        ]
        self._file_annotation_indices = [
            annotations[annotations[file_id_column] == fid].index.values.tolist()
            for fid in self._file_ids
        ]
        self._file_start_times = [
            annotations[annotations[file_id_column] == fid][start_time_column].values.tolist()
            for fid in self._file_ids
        ]
        self._file_end_times = [
            annotations[annotations[file_id_column] == fid][end_time_column].values.tolist()
            for fid in self._file_ids
        ]

    def __len__(self):
        return len(self._file_ids)

    def __getitem__(self, idx):
        audio_path = self.audio_dir / f'{self._file_ids[idx]}.{self._file_format}'
        signal, sr = sf.read(audio_path)
        item = {
            'audio_path': str(audio_path),
            'audio_signal': signal,
            'audio_sampling_rate': sr,
            'annotation_indices': self._file_annotation_indices[idx],
            'annotation_start_times': self._file_start_times[idx],
            'annotation_end_times': self._file_end_times[idx]
        }
        return item
    
def aadl_collate_fn(batch):
    sr = np.unique([item['audio_sampling_rate'] for item in batch])
    assert len(sr) == 1, "All audio files should have the same sampling rate, " +\
    f"but found multiple: {sr}"
    
    batch_dict = {
        'audio_path': [item['audio_path'] for item in batch],
        'audio_signal': [item['audio_signal'] for item in batch],
        'audio_sampling_rate': sr.item(),
        'annotation_indices': [item['annotation_indices'] for item in batch],
        'annotation_start_times': [item['annotation_start_times'] for item in batch],
        'annotation_end_times': [item['annotation_end_times'] for item in batch]
    }
    return batch_dict

def get_annotated_audio_loader(
    annotations_file: str | Path, 
    audio_dir: str | Path,
    batch_size: int = 30,
    collate_fn = aadl_collate_fn,
    start_time_column: str = 'start_time', 
    end_time_column: str = 'end_time', 
    file_id_column: str = 'file_id', 
    file_format: str = 'wav'):
    
    annotated_audios = AnnotatedAudioDataset(
        annotations_file, 
        audio_dir,
        start_time_column=start_time_column,
        end_time_column=end_time_column,
        file_id_column=file_id_column,
        file_format=file_format
    )
    annotated_audio_loader = DataLoader(annotated_audios, batch_size=batch_size, collate_fn=collate_fn)
    
    return annotated_audio_loader