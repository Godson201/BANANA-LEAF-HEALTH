# Disease coverage and interpretation

| Condition | Default model | Expanded research model |
|---|---|---|
| Cordana | Predicted | Predicted |
| Healthy appearance | Predicted | Predicted |
| Pestalotiopsis dataset class | Predicted | Predicted |
| General Sigatoka class | Predicted | Predicted |
| Fusarium / Panama dataset class | Guidance only | Experimental prediction |
| Moko bacterial wilt | Guidance only | Experimental prediction |
| Banana bunchy top | Guidance only | Guidance only |
| Xanthomonas bacterial wilt | Guidance only | Guidance only |

The models are MobileNetV3 Small neural networks loaded from real saved PyTorch weights. The expanded classifier head was trained using frozen features from the existing model, class-weighted cross entropy, Adam, training-only augmentation and validation-based early stopping. It trained for 17 epochs and restored epoch 11.

There are 993 originals in the expanded split: 693 training, 149 validation and 151 test images. The original assignment split is retained. New-class image hashes and perceptual similarity groups are kept separate across splits. No publisher-provided augmented images enter evaluation. Images from one source can still share farm or capture-session information that is unavailable here; the reported score does not establish farm independence.

## Important limits

- There are just seven Fusarium/Panama and nine Moko test images.
- New-class images come from a different source, which can create source-specific shortcuts.
- Moko and Xanthomonas wilt are different diseases; the model must not substitute one for the other.
- The Fusarium/Panama label does not identify a race or confirm TR4.
- Bunchy top and Xanthomonas wilt need suitable expert-labeled images before training.
- Model scores are uncalibrated and do not establish disease severity or treatment.

Read the source-linked guidance in the dashboard. It uses FAO, IITA, ProMusa and the Australian agriculture department. These sources support general information; they do not validate this project's model. New-class testing on independent farms and expert review are needed before operational use.

## Reproduce the research experiment

From the project root, using the installed environment:

```powershell
.venv\Scripts\python tools/train_expanded_preview.py
.venv\Scripts\python tools/evaluate_expanded_preview.py
```

The saved split manifest and included original PNG images allow retraining without the downloaded candidate archives. The original four-class model is not overwritten. The research review notebook loads the saved experiment without retraining.
