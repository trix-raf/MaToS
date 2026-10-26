"""Word (sign) level engine: body + hand landmarks -> temporal CNN -> one word per finished sign, with the letter engine as the fallback.

    holistic    torso-normalised features (pose + both hands), augmentation, left/right hand assignment
    model       numpy 1-D temporal CNN (manual backprop, no torch) + ONNX export through matos.onnx_min
    sources     dataset adapters: Google ISLR (asl-signs) parquet, plain .npz sequences (your own recordings) -> word shards
    train       signer-held-out evaluation -> calibration -> verifier -> final fit -> bundle;  evaluation of an exported bundle
    engine      the exported bundle (ONNX Runtime / OpenCV) + the classifier of one finished sign
    segmenter   when did a sign start / end (wrist motion in torso units)
    hybrid      word engine + letter engine in ONE decoder: the word model decides first, the hand engine is the fallback
    gloss       dataset label -> the word that goes into the transcript (THANK YOU, FRENCH FRIES, ...)
    capture     record your own words with the webcam (landmarks only)
    synth       fake sign data / streams / a fake Google-format folder for smoke tests (says nothing about real accuracy)
"""
