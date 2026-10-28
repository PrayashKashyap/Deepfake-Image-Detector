# Datasets: acquisition, preparation and leakage control

**No dataset is included or downloaded automatically.** Obtain each dataset yourself through its official channel and comply with its license and terms of use. Keep the data outside the Git repository; only manifests (small CSVs) live in `data/manifests/` (also git-ignored by default).

## Datasets actually used for the reported results

| Track | Dataset | Source | Size used | Manifest |
|---|---|---|---|---|
| Facial deepfakes | **DeepFakeFace (DFF)**, Song et al. 2023 (arXiv:2309.02218) | Authors' upload `OpenRL/DeepFakeFace` on Hugging Face (OpenRAIL; real images from IMDB-WIKI, non-commercial research) | all 120,000 images (30k real + 3 x 30k fakes) | `data/manifests/dff_v2.csv` |
| AI-generated images | **GenImage** official test set, Zhu et al. NeurIPS 2023 | Public mirror `jzousz/GenImage` (`genimage_test.zip`) | random subset, 1,000 real + 1,000 fake per generator x 8 = 16,000 | `data/manifests/genimage_sub_v1.csv` |

Kaggle was not used because it requires a personal API token. FaceForensics++, Celeb-DF and DFDC mirrors were deliberately **not** used, because their official terms require an access request.

Reproduce (PowerShell, venv active; data stored outside the repo in `C:\Users\<you>\datasets`):

```powershell
# DFF: download the 4 zips (5 GB) from https://huggingface.co/datasets/OpenRL/DeepFakeFace, extract to datasets\DeepFakeFace
python -m ml.data.deepfakeface --root C:\Users\<you>\datasets\DeepFakeFace --out data\manifests\dff_v2.csv `
    --dataset-id deepfakeface_v2 --merge-near-duplicates 2
# GenImage subset: selective extraction over HTTP range requests (no 25 GB download)
python -m ml.data.remote_zip --url https://huggingface.co/datasets/jzousz/GenImage/resolve/main/genimage_test.zip `
    --out C:\Users\<you>\datasets\GenImage_subset --per-group 1000 --chunk 50 --seed 42
python -m ml.data.genimage --root C:\Users\<you>\datasets\GenImage_subset --out data\manifests\genimage_sub_v1.csv `
    --layout genimage --dataset-id genimage_subset --max-per-class-per-source 1000 --val-frac 0.15 --test-frac 0.15 `
    --merge-near-duplicates 2 --seed 42
```

### Findings of the data audit (`reports/data_validation/`)

- **DFF**: no missing or corrupted files, but **165 sets (344 images) of byte-identical real photos filed under different identities**. With an identity-only split, 84 of them crossed splits, plus 19,949 cross-identity near-duplicate pairs (dHash at most 4). Fix: `--merge-near-duplicates 2` merges leakage groups across near-duplicates (30,000 to 18,511 groups) before splitting.
- **DFF size and format difference**: every fake is a 512x512 JPEG; no real image is 512x512 (median short side 440 px), and 1,048 real images are PNG, GIF or MPO. The source-normalisation option (`preprocess.normalize_short_side` / `normalize_jpeg_quality`) equalises this. In our experiments it did not reduce accuracy and strongly improved robustness.
- **GenImage subset**: no missing or corrupted files and no exact duplicates; the merge joined 47 near-duplicate ImageNet photos. Known bias: real images are JPEG and generated images are mostly PNG (Grommelt et al., 2024).

## Track 1: AI-generated images

**GenImage** (Zhu et al., NeurIPS 2023 Datasets & Benchmarks): about 1.3M real ImageNet images paired with images from 8 generators (Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN). Download it from the authors' official project page or repository. Check the license: the real images inherit ImageNet's terms (non-commercial research).

Expected layout after extraction (directory names vary by release; the scanner only requires `train|val` and `ai|nature` somewhere in each path, with the generator as the top-level folder):

```
GenImage\
  Midjourney\imagenet_midjourney\train\ai\*.png
  Midjourney\imagenet_midjourney\train\nature\*.JPEG
  Midjourney\imagenet_midjourney\val\ai\...
  stable_diffusion_v_1_4\...
```

Build a balanced, reproducible manifest:

```powershell
python -m ml.data.genimage --root D:\datasets\GenImage --out data\manifests\genimage_v1.csv `
    --layout genimage --max-per-class-per-source 2000 --val-frac 0.1 --seed 42
```

- GenImage `val` becomes our **test** split. GenImage `train` is split into **train/val** with the seed.
- `source` is the generator folder name, so cross-generator experiments use source filters (Experiment 2).
- `--max-per-class-per-source N` builds a balanced subset for rapid experiments.

**Any real/fake folder** (e.g. a CIFAKE-style dataset you have rights to) works with `--layout folder`:

```
root\real\[<source>\]...\*.jpg
root\fake\<generator>\...\*.png
```

## Track 2: facial deepfakes

**FaceForensics++** (Rössler et al., ICCV 2019). Access requires submitting the authors' request form and agreeing to their terms of use. You then receive the official download script. Use only the methods and compression levels you are permitted to use. **Never** bypass the access process.

1. Extract frames (OpenCV; optional largest-face crop with the bundled Haar cascade; lossless PNG output):

```powershell
python -m ml.data.faceforensics extract --root D:\datasets\FFpp --out D:\datasets\ffpp_frames `
    --compression c23 --frames-per-video 16 --face-crop
```

2. Build a leakage-aware manifest:

```powershell
python -m ml.data.faceforensics manifest --frames-root D:\datasets\ffpp_frames --out data\manifests\ffpp_c23.csv `
    --dataset-id ffpp_c23 --official-splits D:\datasets\FFpp\splits     # official train/val/test JSON (recommended)
```

Leakage control:
- A manipulated video `000_003` combines target 000 and source 003. A union-find links every video id that co-occurs, and each connected component becomes one **group**. A group (all frames of a person/video and every manipulation derived from it) is always assigned to a single split.
- With `--official-splits`, the official FF++ pair lists decide the split; the group invariant is then verified.
- Frames of the same video are never spread across train and test.
- `source` = manipulation method (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`) or `original`; `video` and `identity` hold video ids. No person is ever identified by name.

A generic pre-extracted layout `root\{real,fake}\<method>\<video_id>\<frames>` is also supported (same `manifest` command). Other legally accessible face datasets (e.g. Celeb-DF, DFDC) can be converted to this layout.

## Manifest schema

`path,label,split,source,group,identity,video,dataset_id`: see [`ml/data/manifest.py`](../ml/data/manifest.py). Labels: `0 = REAL`, `1 = FAKE`. Each build also writes `<manifest>.summary.json` (class balance per split and source).

## Validation (run before every training)

```powershell
python -m ml.data.validate --manifest data\manifests\genimage_v1.csv --root D:\datasets\GenImage `
    --near-duplicates --hamming 4 --out reports\data_validation\genimage_v1.json
```

The validator checks:
- label and split values;
- missing files;
- corrupted images (verify plus a full decode);
- exact duplicates (SHA-256), including duplicates with conflicting labels and duplicates **across splits**;
- perceptual near-duplicates (64-bit dHash, exact 8-band LSH, Hamming ≤ threshold), especially pairs across splits;
- group leakage;
- class balance.

Remove or reassign cross-split duplicates before training. GenImage real images come from ImageNet, so different generator subsets may contain the same or nearly identical photos.

## Registering datasets

Copy `ml\configs\datasets.example.yaml` to `ml\configs\datasets.local.yaml` (git-ignored) and set `root` and `manifest` for each name. Configs and the UI refer to datasets **by name** only, so machine-specific paths never appear in configs or API responses.

## Synthetic test fixture (software testing only)

`python -m ml.data.fixture --out data\fixtures\synthetic` generates tiny procedural images whose "fake" class has an artificial checkerboard pattern. It exists only so the pipeline and tests can run without real data. Its `dataset_id` starts with `SYNTHETIC_FIXTURE`. Every checkpoint and result derived from it is flagged `is_synthetic_fixture: true`, and the UI shows a warning. **Never report fixture metrics.**

## Responsible-use restrictions

- Use the datasets only for the research purposes their licenses allow; do not redistribute them.
- Do not use face data to identify or profile people.
- Cite the dataset papers in any publication.
