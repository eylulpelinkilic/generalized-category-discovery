# Running the official Open-VCLIP and TC-CLIP code (compatible machine)

The adapters in this repository run on CPU/MPS with current PyTorch. The **official** test
scripts need a Linux machine with an NVIDIA GPU (they call `.cuda()`). Use them to check that
the adapter reproduces the native configuration before trusting adapter numbers.

## Open-VCLIP environment (from the official INSTALL.md)

```bash
conda create -n openvclip python=3.8.13 pip && conda activate openvclip
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy pandas scikit-learn ftfy regex simplejson psutil opencv-python tensorboard iopath
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av -c conda-forge
git clone https://github.com/facebookresearch/pytorchvideo.git && pip install -e pytorchvideo
git clone https://github.com/facebookresearch/detectron2 detectron2_repo && pip install -e detectron2_repo
pip install 'git+https://github.com/facebookresearch/fairscale'
cd third_party/Open-VCLIP && export PYTHONPATH=$PWD/slowfast:$PYTHONPATH && python setup.py build develop
```

Downloads: the checkpoint folder in docs/backbones.md (`openvclip-b16/swa_2_22.pth`) and the
original `ViT-B-16.pt` (`python -c "import clip; clip.load('ViT-B/16')"` with the official
`openai/CLIP` package, or the URL in docs/backbones.md). Record `sha256sum` of both in
docs/backbones.md.

## On Kaggle

`kaggle/ovar_ucf101_openvclip_check.ipynb` runs the check below end to end. It builds the same
environment with `uv` (Python 3.8, PyTorch 1.11 + CUDA 11.3, pytorchvideo pinned to
`2ac6ed36`) and deviates from INSTALL.md in two import-only ways: detectron2 is replaced by a stub
(the official code imports it at module level for detection heads, the demo and visualisation;
the classification test path never calls it, and any use of the stub raises), and the repository
is put on `PYTHONPATH` instead of `setup.py build develop` (pure Python; its `install_requires`
names `detectron2`, `PIL` and `sklearn`, which pip cannot install). `pip freeze` of that
environment is saved with the outputs.

## Native check on a small subset (UCF101, 2 videos per class)

```bash
# 1. In this repository: export exactly the subset that `infer --per-class 2` uses.
python -m ovar export-native --backbone openvclip --datasets ucf101 --per-class 2 --out runs/export_ovc

# 2. In third_party/Open-VCLIP (openvclip environment): official native test.
python -W ignore -u tools/run_net.py --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml --opts \
  DATA.PATH_TO_DATA_DIR $REPO/runs/export_ovc/ucf101 DATA.PATH_PREFIX $UCF_ROOT DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $REPO/runs/export_ovc/ucf101/ucf101-index2cls.json \
  TRAIN.ENABLE False OUTPUT_DIR $REPO/runs/official_ovc TEST.BATCH_SIZE 8 NUM_GPUS 1 \
  DATA.DECODING_BACKEND pyav MODEL.NUM_CLASSES 101 MODEL.TEMPORAL_MODELING_TYPE expand_temporal_view \
  TEST.CUSTOM_LOAD True TEST.CUSTOM_LOAD_FILE $CKPT/openvclip-b16/swa_2_22.pth \
  TEST.SAVE_RESULTS_PATH preds.pkl TEST.NUM_ENSEMBLE_VIEWS 3 TEST.NUM_SPATIAL_CROPS 1 \
  TEST.PATCHING_MODEL True TEST.PATCHING_RATIO 0.5 TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt

# 3. Back here: adapter on the same subset, then compare video by video.
python -m ovar infer openvclip_vitb16__openvclip_native_8x3x1 --per-class 2 \
  --set 'datasets=[ucf101]' --set model.trust_pickle=true
python -m ovar compare-native --official runs/official_ovc/preds.pkl \
  --run-dir runs/openvclip_vitb16__openvclip_native_8x3x1 --dataset ucf101 --trust-pickle \
  --out runs/native_check/compare_ucf101_subset2.json   # optional: also lists disagreeing videos
```

Expect top-1 agreement close to 1.0. Remaining disagreements come from decoding (PyAV
selective decoding vs ffmpeg full decoding: frame windows can shift by a frame) and are
reported, not hidden. Repeat with `DATA.DECODING_BACKEND torchvision`, which the HMDB script
and the YAML use.

## TC-CLIP (optional)

Environment: follow `third_party/tc-clip/docs/INSTALL.md`. Download only the zero-shot
Kinetics-400 checkpoint (docs/backbones.md).

```bash
python -m ovar export-native --backbone tcclip --out runs/export_tcclip
```

Add a Hydra data config inside the TC-CLIP repository, e.g. `configs/data/xov_eval.yaml`
(one block per domain; `root` is the dataset root from configs/paths.yaml):

```yaml
#@package _global_
data:
  train: {dataset_name: k400_train, root: ${k400.root}, num_classes: 400,
          label_file: ${k400.label_files.default}, ann_file: ${k400.ann_files.train}}
  val: {dataset_name: k400_val, root: ${k400.root}, num_classes: 400,
        label_file: ${k400.label_files.default}, ann_file: ${k400.ann_files.val}}
  test:
    - name: xov_ucf101
      protocol: top1
      dataset_list:
        - {dataset_name: xov_ucf101, root: /data/UCF-101, num_classes: 101,
           label_file: /abs/path/runs/export_tcclip/ucf101/ucf101_labels.csv,
           ann_file: /abs/path/runs/export_tcclip/ucf101/test.txt}
    # ... hmdb51 (51), arid (11), necdrone (16)
```

Then `torchrun --nproc_per_node=1 main.py -cn zero_shot data=xov_eval output=<out> trainer=tc_clip
eval=test resume=<zero_shot_checkpoint> gather_filename=true`. Note that the native final test
uses two randomly jittered temporal views (`multi_view_inference`), so report the seed and
expect small run-to-run differences; `multi_view_inference=false` gives the deterministic
single-view (16 × 1 × 1) setting that matches the common protocol's frame count.
