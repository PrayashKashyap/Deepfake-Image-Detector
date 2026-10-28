# Limitations and responsible use

## What a prediction means

- The output is a **model score** (softmax for FAKE) compared with a threshold chosen on validation data. It is not a calibrated probability unless calibration has been evaluated, and it is never proof.
- **False positives** (authentic images flagged FAKE) and **false negatives** (fakes passed as REAL) occur. Their rates depend on the data and the threshold.
- Results must **not** be presented as legal, journalistic or forensic evidence.

## Known technical limitations

- **Distribution shift.** Detectors trained on some generators or manipulation methods often degrade on unseen ones (Experiments 2 and 3 measure this). A model trained on AI-generated scenes is not expected to detect face swaps, and vice versa; that is why the platform has two separate tracks.
- **Post-processing.** JPEG re-compression, resizing, blur, noise, screenshots and social-media pipelines can remove the artifacts a model relies on (Experiment 4).
- **Dataset biases.** GenImage real images come from ImageNet. Real and fake images can differ in format, size or compression in ways a model may exploit; those shortcuts do not generalize. FF++ contains a limited set of identities and sources.
- **Preprocessing.** Inputs are resized to the checkpoint's resolution (center crop by default for track 1), so small regions outside the crop are not seen.
- **Explainability.** Grad-CAM is coarse (7×7 for both backbones at 224 px) and qualitative. A heatmap on a face or object does not prove the model uses meaningful forensic evidence. Explainability does not improve accuracy.
- **Frequency analysis.** Spectral peaks can come from generator upsampling, but also from JPEG 8×8 blocks, resampling or textures. No spectral pattern uniquely identifies a fake.
- **Metadata.** Missing EXIF is normal (platforms strip it) and is **not** evidence of manipulation. Present metadata can be forged.
- **Face crops.** The OpenCV Haar detector misses profile and occluded faces. Missed faces are skipped during frame extraction, which can bias the dataset.
- **Statistics.** Small test sets give wide confidence intervals. Frame-level metrics on video datasets overstate the effective sample size; use the grouped bootstrap CIs.

## Responsible use

- The system does not identify, recognize or profile people, and it must not be extended to do so.
- Respect dataset licenses and terms of use; do not redistribute restricted data.
- Uploaded images may be personal data. Use `DFF_API_KEY` on shared machines, delete analyses that are no longer needed (History → delete removes the files), and do not upload images you have no right to process.
- Sensitive EXIF fields (GPS, serial numbers, owner/artist names and similar) are redacted in the API, the UI and reports.
