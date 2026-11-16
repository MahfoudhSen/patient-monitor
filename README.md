# AI Patient Stand-Up & Fall Monitor (free, runs locally)

A camera watches the patient. AI tracks their body posture and **alerts staff when the patient
sits up, stands up, leaves the bed, falls, or disappears from view.** Built for ER bays with
high-fall-risk patients (elderly, sedated, delirious, intoxicated).

**Everything is free and runs on one computer. No video leaves the machine.**

| Part | Tool (free / open source) |
|---|---|
| Body tracking | Google MediaPipe Pose (33 body points, works on a normal laptop CPU; model downloads automatically on first run) |
| Alert double-check (optional) | Local vision LLM via **Ollama** (`moondream` or `llava`) |
| Phone alerts | **ntfy** push notifications (free app for iPhone & Android) |
| Other alerts | Loud alarm sound, flashing red on-screen banner, CSV log, webhook |

## What it detects

| Event | How |
|---|---|
| **PATIENT SITTING UP** (early warning) | Torso goes from horizontal to upright |
| **PATIENT STANDING UP** | Torso upright + legs straight (knee angle > 150°) |
| **LEFT BED ZONE** | Hips leave the bed box you draw |
| **FALL DETECTED** | Hips drop fast (within 1 second) to lying/floor level. Always alerts immediately |
| **PATIENT OUT OF VIEW** | No person seen for 20 seconds |

## Setup (about 10 minutes)

1. Install Python 3.10 or newer, then:
   ```bash
   pip install -r requirements.txt
   ```
2. Point a USB webcam at the bed so the **whole bed and the floor beside it** are visible
   (mount it high on the wall at the foot of the bed if you can).
3. Run:
   ```bash
   python patient_monitor.py --room "ER Bay 12"
   ```
4. Press **b** and drag a box around the bed, then press ENTER.

### Phone alerts for nurses
1. Install the **ntfy** app on staff phones and subscribe to a hard-to-guess topic name, e.g. `er-bay-alerts-7f3k9`.
2. Run with `--ntfy-topic er-bay-alerts-7f3k9`.

Alerts contain only the room label and event, never images or patient names.

### Optional: free local AI to cut false alarms
```bash
# install Ollama from https://ollama.com, then:
ollama pull moondream          # small and fast; or: ollama pull llava
python patient_monitor.py --room "ER Bay 12" --llm moondream --llm-veto
```
When the pose detector raises an alert, the frame goes to the **local** LLM, which answers
LYING / SITTING / STANDING / FLOOR / NOBODY. With `--llm-veto`, an alert the LLM clearly disagrees
with is suppressed (and printed). **Falls are never delayed or vetoed.**

### Test without a patient
Record yourself lying on a couch, sitting up, standing and walking away, then run:
```bash
python patient_monitor.py --video test.mp4
```

## Keys
`b` draw bed zone · `c` clear zone · `p` privacy blur (shows the skeleton only) · `m` mute · `q` quit

## Tuning
| Flag | Default | Meaning |
|---|---|---|
| `--cooldown` | 30 | seconds before the same alert repeats |
| `--fall-drop` | 0.18 | how far (fraction of frame) the hips must drop in 1 s to count as a fall |
| `--floor-line` | 0.80 | hips below this line on screen = floor level (gray line) |
| `--no-alert-sitting` | – | turn off sit-up early warnings |
| `--model-complexity` | 1 | 0 = fastest, 2 = most accurate |
| `--headless` | – | no window; for a mini-PC in the room |
| `--webhook URL` | – | send JSON alerts to Teams/Slack or a nurse-call integration |

All alerts are logged to `alerts_log.csv`.

## ⚠️ Before using with real patients (e.g. at Montefiore Einstein)
This is a **prototype**, not an FDA-cleared medical device. It is meant for demos and
pilots, and it does not replace sitters, bed alarms or rounding. To move it into a hospital:
- **Get approval first:** Nursing leadership, Patient Safety, Clinical Engineering/Biomed, IT Security,
  Privacy/HIPAA Compliance and possibly the IRB (if it's run as a study).
- **Keep it on-premises:** use a self-hosted ntfy server (`--ntfy-server`) on the hospital network,
  not the public ntfy.sh. Don't record video.
- **Validate it:** run it alongside existing bed alarms for several weeks and measure how many real
  events it caught versus how many false alarms it raised.
- **Known limits:** heavy blankets, bed rails, poor lighting, several people in the frame (it tracks
  one person, so staff at the bedside can confuse it) and camera angle all affect accuracy. An IR
  night-vision camera helps at night.
