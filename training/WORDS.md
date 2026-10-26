# Word level (signed words) + letter level, in one decoder

Signed words ("ME", "HELLO", "THANK YOU") and fingerspelling run side by side. The phone sends the hand landmarks it already
sends **plus the upper-body pose**; the server decides, sign by sign, which engine owns the motion.

```
phone: hands (21 pts x 2)  +  pose (nose ... hips, 25 pts, ~15 Hz)  -- websocket -->  server
server, every frame:
  letters   the existing hand model + LetterDecoder (unchanged parameters, unchanged logic)
  words     wrist motion in torso units -> SignSegmenter ("a sign started / ended")
            once a sign has FINISHED  ->  word CNN reads the whole segment ONCE
  arbiter   word accepted  -> `sign` event, the letters the hand model "read" meanwhile are dropped
            not accepted   -> the held-back letter frames are replayed: you get exactly what the letter engine alone would have typed
```

* **Features**: everything is measured in the signer's torso (origin = shoulder mid-point, unit = shoulder width), so "hand at the
  chest" means the same for any person at any distance. Per frame: 13 body points (head, shoulders, elbows, wrists), both hands
  (location, hand shape, 44 geometric features) and their frame-to-frame motion = 298 numbers. A sign is resampled to 32 frames.
* **Model**: a real 1-D temporal CNN (4 conv layers, mean+max pooling, dense), trained in plain numpy like your letter MLP
  (no torch), exported through `matos/onnx_min.py` to ONNX. ONNX Runtime runs it (OpenCV 5 DNN reproduces it too:
  `WordBundle(..., prefer="opencv")`). One extra class, *background*, is trained on "not a sign" motion (raising the hand to
  fingerspell, random gestures, nothing) built from your real letter shapes - that is what lets the letter engine take over.
* **Honest evaluation**: whole *signers* are held out, so the printed accuracy is for people the model never saw.
* The letter bundle is untouched; word bundles live in `models/words/`, word shards in `models/wshards/`. The letter code cannot
  see them (and vice versa), and with no word bundle everything behaves exactly as before.

## 1. Check the plumbing (5 minutes, no download, no camera)
```
cd training
python -m matos words-synth --out wsyn --words 8 --per-class 50 --signers 8 --islr-folder data/fake_islr
python -m matos words-train --shards wsyn --out wsmoke --steps 400 --channels 32,32,48,48 --dense 96 --letters-shards none
python -m matos words-extract --src fake=data/fake_islr --out wfake --workers 2     # the Google-format reader, on a fake folder
```
(synthetic signs: this proves the pipeline works on your machine; it says nothing about real accuracy.)

## 2. Real data: the Google ISLR set (PopSign signs)
`sohier/461054610546105` is not a Kaggle dataset I could find (the number looks like `46105` repeated). Sohier Dane is on Google's
sign-language team and the word-level data is the **competition** `asl-signs` ("Google - Isolated Sign Language Recognition"):
250 signs, 21 signers, MediaPipe landmarks of face / both hands / **pose**, as parquet. It is *not* a dataset, so it is fetched with
`kagglehub.competition_download` after you accept the rules once at kaggle.com/competitions/asl-signs/rules.
**It is ~40 GB (57 GB unpacked)** - start with a subset:
```
pip install -r requirements-words.txt
python scripts/kaggle_words.py --per-sign 40         # ~10k files, a few GB, resumable; prints the next command
python -m matos words-extract --src islr="<path it printed>" --out wshards --max-per-class 40 --seed 0
python -m matos words-train   --shards wshards --out words1
python -m matos words-eval    --bundle words1 --shards wshards --include islr
```
`words-extract` prints a report: files kept, and whether the dataset's left/right hand labels agree with the pose (if not:
`--swap-hands`). The frames' aspect ratio is *assumed* 3:4 (`--aspect`); training randomises x/y stretch by +-10 % to cover a wrong guess.
Training time scales with `--steps` (default 6000 x batch 256). On one CPU core a step takes roughly 0.3 s at the default size;
a multi-core machine is faster. `--features pose_hands` (no geometric block) and smaller `--channels` are the cheap levers.

## 3. Your own words - "ME" is not in the 250
The vocabulary is 250 child-oriented words (hello, mom, dad, yes, no, please, thankyou, where, who, why, minemy, yourself, ...). Record
what you need, ~20-30 repetitions each, in the place and light you will use:
```
python -m matos words-capture --words me,hello,background --reps 25 --signer yourname --out words_mine
python -m matos words-extract --src mine=words_mine --out wshards            # same folder as the Google shards: they merge
python -m matos words-train   --shards wshards --out words2
```
Only landmarks are stored. `background` is a special word: record things you do that are NOT signs (scratching, waving, reaching,
adjusting glasses) and they become the background class. Rare words are not drowned by the big dataset (class-balanced sampling).
Display text: `data/gloss_map.tsv` (`label<TAB>GLOSS`) overrides how a label is written (default: upper case; `thankyou` -> THANK YOU).

## 4. Run it
From the folder that contains `app/`, `server/`, `training/`:
```
python -m app                       # newest bundle in training/models/words; letters-only when there is none
python -m app --word-bundle words2 --mode auto|letters|words
```
The page gets a small *Words + letters / Letters / Words* switch and shows each recognised sign. Mixed sentences go through the
dictionary corrector (spelled parts only) and then T5 (it was trained on exactly this kind of gloss).
Phone URL flags: `?pose=off` (never load the pose model), `?pose=cpu` (force the CPU delegate). The pose model runs at most ~15 Hz and
slows itself down if it is slow, but **check the frame rate on your phone** - it is the one thing I could not measure.

## 5. Tuning (`models/words/<bundle>/config.json`, read when `python -m app` starts)
| symptom | change |
|---|---|
| a long sign is cut in two | `segmenter.still_s` or `segmenter.drop_s` up |
| small signs never start a segment | `segmenter.v_on` down (0.9 -> 0.6) |
| words appear that you did not sign | `hybrid.conf_on`, `hybrid.margin_min`, `hybrid.verifier_min` up; record `background` |
| real signs turn into stray letters | the same, down; record more of that word |
| a held hand after a sign types a letter | `hybrid.gate_s` up |
| a resting hand types letters | `hybrid.rest_y` down (torso units: 0 = shoulders, ~2.5 = hips) |
Sentence pause follows your letter bundle's `pause_s`.

## 6. Limits you should know
* Not verified on real data: no Kaggle access from where I built this. Expect less than the dataset authors' own 82 % (val) / 84 % (test)
  (2-layer BiLSTM, 250 signs, held-out signers, hand landmarks only) - and less again live: the data was made with MediaPipe *Holistic*, the phone runs
  two separate Tasks models, and your signing differs from 21 volunteers'. Recording your own words closes that gap fastest.
* Isolated signs only (the dataset authors list continuous signing as an unsuitable use): signs are cut by hand motion and read one by one.
  Fast chains with no pause between signs may merge.
* One signer in the frame; the body must be visible down to the shoulders (it is what the torso is measured from).
