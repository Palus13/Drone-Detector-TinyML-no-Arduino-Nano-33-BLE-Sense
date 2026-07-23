# Python Pipeline

## Before running

All scripts assume they are run **from the repository root**, e.g.:

```
python python/01_prepare_dataset.py
```

They use relative paths (`dataset/`, `processed/`, `models/`,
`arduino/drone_detector/`). Install dependencies first:

```
pip install -r requirements.txt
```

## Pipeline (run in this order)

| # | Script | What it does |
|---|--------|---------------|
| 1 | `01_prepare_dataset.py` | Download the [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) into `dataset/yes_drone/` and `dataset/unknown/`, then extract MFCC features from the whole public dataset -> `processed/dataset.npz` |
| 2 | `02_train_base_model.py` | Trains the final architecture (DS-CNN with `strides=2`, RAM-optimized) on the public dataset -> `models/lightweight_drone_base_v2.keras` |
| 3 | `03_record_via_arduino.py` (+ `arduino/record_and_dump/`) | Records audio of your own drone **using the Arduino's own microphone** (avoids the domain gap between microphones) -> `dataset/gravacoes_arduino/*.wav` |
| 4 | `04_finetune_my_drone.py` | Fine-tunes the base model with the Arduino recordings (BatchNorm frozen) -> `models/finetuned_lightweight_drone_arduino_mic.keras` |
| 5 | `05_convert_to_tflite.py` | Converts the fine-tuned model to `.tflite` |
| 6 | `06_export_arduino_assets.py` | Generates `model.h` and `mfcc_tables.h` directly inside `arduino/drone_detector/` |

After step 6, open `arduino/drone_detector/` in the Arduino IDE and
upload — no manual file copying needed.
