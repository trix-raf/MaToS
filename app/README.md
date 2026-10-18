# app/ - phone bridge

```
phone browser  --(MediaPipe JS, already in js/)-->  landmarks  --wss-->  FastAPI  -->  matos (features -> bundle -> LetterDecoder -> autocorrect)
      ^                                                                                         |
      +-------------------- window.MaToS.appendTranscript(text)  <------------------------------+
```

Nothing in `server/frontend` or `training/matos` is edited. `app.html` is served with one extra `<script>` injected
on the fly; `bridge.js` taps MediaPipe's result inside the existing `HandTracker` (no second model on the phone).

## Run

```
pip install --dry-run -r app/requirements.txt -c <your pinned requirements file>   # preview: what would be added
pip install           -r app/requirements.txt -c <your pinned requirements file>   # `-c` freezes every package you already have
pip check                                                                          # should say: No broken requirements found
python -m app                       # from the project root (the folder with app/ server/ training/)
python -m app --bundle run1         # same names as `python -m matos live --bundle`; default = newest bundle
```

It prints the phone URL (and a QR code). Open `https://<pc-ip>:8000/app.html` on the phone, same wifi.

* **Certificate warning on first visit is expected** (self-signed). Accept it once. https is required: on plain
  `http://192.168...` the phone gives no camera access at all.
* **Windows firewall**: allow Python on *Private* networks when prompted (or open TCP 8000), otherwise the phone can't connect.
* The phone needs internet too: `handtracker.js` loads MediaPipe and the hand model from CDNs.
* Quick check from the phone: `https://<pc-ip>:8000/api/health` shows the loaded bundle / engine / text stage.

## Use it from anywhere (ngrok)

Same-wifi only is the default. To reach the server from **any network** (phone on mobile data, a friend's wifi, ...):

```
pip install pyngrok
python -m app --ngrok --ngrok-token <your authtoken>        # token: https://dashboard.ngrok.com/get-started/your-authtoken
```

* It prints a public `https://....ngrok-free.app/app.html?key=...` link and a QR code of it. Open that on the phone.
  ngrok supplies a real certificate, so **no certificate warning** and the camera works. The local server then runs on plain
  http and only listens on `127.0.0.1` (only the tunnel can reach it).
* **Access key**: the public URL is on the open internet, so with `--ngrok` every request needs a key. The link already
  contains it (it becomes a cookie on the first visit). It is random per run; set `--key mysecret` (or `MATOS_KEY`) to
  keep it fixed, `--no-auth` to turn it off. Don't post the link publicly.
* **Same link every run**: in the ngrok dashboard (Domains) claim your free static domain, then
  `python -m app --ngrok --ngrok-domain your-name.ngrok-free.app --key mysecret`.
* Free plan: the very first visit shows an ngrok notice page - tap **Visit Site**. The free plan also allows one agent
  session at a time, so close any other running `ngrok`.
* Save the token once with `ngrok config add-authtoken <token>` (or set `NGROK_AUTHTOKEN`) and just run `python -m app --ngrok`.
* Everything still goes through your PC, so the PC must stay on and running `python -m app`.

## Options

| flag | meaning |
|---|---|
| `--text-mode sentences` (default) | the transcript gets the autocorrected sentence after a ~1.5 s pause (`process_letter_chunks`) |
| `--text-mode words` | every word is appended as soon as it ends, uncorrected |
| `--polish` | T5 grammar polish on sentences (falls back to autocorrect only if the model folder is missing) |
| `--no-mirror` | same as `live.py --no-mirror` |
| `--engine auto/opencv/ort` | same as `matos live --engine` |
| `--no-https` | localhost only (camera won't work from a phone) |
| `--ngrok` | public https URL via ngrok, works from any network (see above) |
| `--ngrok-token`, `--ngrok-domain` | authtoken / fixed domain |
| `--key KEY`, `--no-auth` | access key (random per run with `--ngrok`) / disable it |
| `--port`, `--host`, `--frontend`, `--training` | self-explanatory |

Under the camera a small HUD shows the current letter + confidence, the hold-progress bar, the word being typed and
words waiting for the sentence to finish. If something fails, the HUD shows the connection state.

## Files

| file | job |
|---|---|
| `server.py` | FastAPI: static frontend, `app.html` injection, `/ws`, `/api/msg` (HTTP fallback), `/api/health` + the access-key check (`?key=` -> cookie) |
| `session.py` | one `LetterDecoder` per camera run; the frame handling is `live.py`'s loop (mirror -> `to_pixels` -> `normalize` -> `bundle.predict` -> `decoder.update`) |
| `engine.py` | loads the bundle once (`matos.engines.Bundle`), autocorrect / T5 |
| `certs.py` | self-signed cert for the LAN IP (regenerated if your IP changes) |
| `tunnel.py` | opens the ngrok tunnel (`--ngrok`) |
| `static/bridge.js`, `bridge.css` | browser side |

Transport: WebSocket first; if it can't open within 4 s (some phones refuse `wss://` to a self-signed cert) the page
switches to batched HTTP POSTs automatically. Force it with `?transport=http` on the URL.

Landmarks are sent unmirrored with the phone's own timestamps; the server flips x -> 1-x when `CAMERA_MIRROR` is on,
which is what `cv2.flip(frame, 1)` does in `live.py`. The decoder is time-based, so wifi jitter doesn't affect it.
