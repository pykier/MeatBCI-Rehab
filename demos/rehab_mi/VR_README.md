# MetaBCI VR rehab scene integration

This folder adds a browser-based VR/MR feedback client while keeping MetaBCI as the experiment controller.

## Roles

- `rehab_stim_demo.py`: Brainstim MI timing, cue display, LSL software marker.
- `online_neuracle_closed_loop.py`: Neuracle online EEG, MI prediction, robot-hand feedback.
- `vr_scene_server.py`: local web scene opened by the VR headset browser.
- `vr_event_client.py`: UDP event sender used by the MetaBCI scripts.

## Network model

Run the VR scene server on the PC. Keep the VR headset and PC on the same Wi-Fi/router. Open the printed LAN URL on the headset browser.

Default ports:

- HTTP web scene: `8766`
- UDP event input: `8765`

## Quick test

Terminal 1:

```powershell
cd <MetaBCI-Rehab project root>
conda activate metabci39
python demos\rehab_mi\vr_scene_server.py
```

Open the printed `http://<PC-IP>:8766` URL in the VR headset browser.

Terminal 2:

```powershell
python demos\rehab_mi\vr_event_client.py --event test --phase TEST --target left_hand --prediction right_hand --control left_hand --trial 1
```

The web page should update immediately.

## Brainstim + VR display

```powershell
python demos\rehab_mi\rehab_stim_demo.py --direct --nrep 2 --lsl-markers --lsl-source-id rehab_mi_marker_stream --vr-events
```

This sends trial phases to the VR page:

- initial 10 s ready period
- prompt, default 2 s
- motor imagery, default 3 s
- rest, default 5 s
- optional feedback

## Online EEG + robot + VR feedback

Run this before starting the stimulation script:

```powershell
python demos\rehab_mi\online_neuracle_closed_loop.py --model demos\rehab_mi\outputs\sub01_neuracle_mi_model.pkl --control-source target --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 4 --require-confirm --vr-events
```

Then run Brainstim:

```powershell
python demos\rehab_mi\rehab_stim_demo.py --direct --nrep 2 --lsl-markers --lsl-source-id rehab_mi_marker_stream --vr-events
```

For true prediction-driven control, change `--control-source target` to `--control-source prediction`.

## One-command online demo

After an offline model has been trained, the competition-site demo can be started from one terminal:

```powershell
python demos\rehab_mi\run_online_demo.py --vr --control-source target --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 6 --nrep 3
```

The launcher starts:

- VR web scene server
- Neuracle online EEG decoder
- Robot-hand controller
- Brainstim MI stimulation with LSL markers

For prediction-driven robot control, use:

```powershell
python demos\rehab_mi\run_online_demo.py --vr --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 6 --nrep 3
```

Default Brainstim timing in the launcher is:

- initial wait: 10 s
- prompt: 2 s
- motor imagery: 3 s
- rest: 5 s
- feedback: 0 s, because online feedback is sent by `online_neuracle_closed_loop.py`

Default custom asset names under `demos/rehab_mi/assets`:

- `left_robot_hand.png`, `right_robot_hand.png` for the Brainstim stimulus window
- `left_hand_motion.gif`, `right_hand_motion.gif` for optional action preview
- `left_vr_hand.png`, `right_vr_hand.png` for the VR web scene

## Competition wording

Use this wording in documents:

`The project extends MetaBCI with a browser-based VR rehabilitation scene. The VR scene is driven by Brainstim trial phases and Brainflow online prediction results, while MetaBCI remains responsible for experiment timing, LSL markers, online EEG processing and robot-hand control.`
