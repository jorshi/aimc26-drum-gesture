# Creating Continuous Datasets

Record data you want to be recognized as continuous. 5-10min of data was sufficient for us. Make sure to include negative examples (i.e., gestures that shouldn't be marked as continuous).

Data for the continuous models are 1+ wav files and accompanying csv sidecar files.

The easiest (right now) is to use [Sonic Visualiser](https://www.sonicvisualiser.org/) to label your own data. 

1. Load your audio into Sonic Visualiser
2. Create a new regions annotation layer:

![SonicViz Region Layer](https://imagedelivery.net/ghxunFRIAYelWM8h7qZwcA/7c661041-b408-48a7-c144-71669e73b100/smaller)

3. Change to a segmentation plot type:

![SonicViz Segmentation](https://imagedelivery.net/ghxunFRIAYelWM8h7qZwcA/9887ff28-08bf-450b-23e2-49356f1bac00/smaller)

4. Draw in the regions of desired gestures with the draw tool.

![SonicViz Drawing Segments](https://imagedelivery.net/ghxunFRIAYelWM8h7qZwcA/f95c8315-0f35-490e-c446-e60e11605200/smaller)


5. Export the annotation layer `File > Export Annotation Layer ...`: save as csv, select the "Write times in audio sample frames" option when exporting.

6. Update or add a new data cfg yaml file. `cfg/data/your_dataset.yaml`. Easiest will be to either update one of the existing files or copy one as a starting point.

    An example cfg:
    ```
    _target_: drum_gesture.data.ContinuousGestureDataset
    num_data: 1000
    audio:
    - continuous/snare_buzz_2.wav
    - continuous/snare_hits.wav
    annotations:
    - continuous/snare_buzz_2-buzz.csv
    - null
    ```

    `num_data` is the number of examples per epoch. The dataset chops the input audio in 1sec segments and then samples from those during training.

    If an audio file doesn't have accompanying annotations, i.e., is only negative examples. You can include `null` for that annotation.

7. Update the base cfg with your new dataset. These are in the root of the `cfg/` folder.

    ```
    defaults:
    - data: your_new_dataset
    - feature: onset_signal
    - model: buzz_rnn
    - loss: bce
    - task: continuous_gesture
    ...
    ```

    Feature can have `onset_signal` (we used for buzz strokes) or `mel_spectrogram_log` (we used for brushing). You can use either, but `onset_signal` models run in `JEM_InputBuzz` and `mel_spectrogram_log` models run in `JEM_InputBrush`.