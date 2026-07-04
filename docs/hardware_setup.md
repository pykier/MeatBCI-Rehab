# Hardware Setup Guide

This guide describes the hardware wiring and preflight checks for the
rehabilitation MI closed-loop demo.

## Devices

- EEG: Neuracle EEG cap and Recorder software.
- EEG stream: Recorder DataService at `127.0.0.1:8712`.
- Sample rate: 250 Hz.
- Channels: 16 EEG channels + 1 marker/trigger channel.
- Left robot hand: `COM4`.
- Right robot hand: `COM3`.
- VR/MR headset: browser connected to the same LAN as the PC.

## EEG Setup

1. Connect the Neuracle amplifier and EEG cap.
2. Start the Neuracle Recorder software.
3. Confirm that EEG waveforms are visible in Recorder.
4. Enable or confirm Recorder DataService on port `8712`.
5. Keep only one MetaBCI acquisition client connected to DataService at a time.

Preflight command:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode offline --num-chans 17 --left-com COM4 --right-com COM3 --move-hands
```

Expected result:

- EEG samples are received.
- Channel count is 17.
- The first several channel means are printed.
- Both robot hands move once if `--move-hands` is enabled.

## Robot Hand Setup

1. Power on both robot hands.
2. Confirm Windows Device Manager shows the expected serial ports.
3. Use `COM4` for the left hand and `COM3` for the right hand unless the ports
   have changed.
4. Keep hands clear before running any command with serial feedback.

If a serial port is occupied, close previous Python terminals and retry. The
project uses asynchronous feedback and should continue EEG collection even if a
robot write fails, but a correct demo requires both ports to be available.

## VR/MR Setup

1. Connect the PC and headset to the same LAN.
2. Start the online demo or preflight VR server.
3. Open the printed headset URL in the VR browser, for example:

```text
http://192.168.1.xxx:8766
```

Online preflight command:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode online --num-chans 17 --left-com COM4 --right-com COM3 --move-hands --vr
```

Expected result:

- The PC prints a local URL and one or more LAN URLs.
- The VR browser loads the MetaBCI rehabilitation scene.
- The page updates when the online demo sends Brainstim events.

## Online Demo

Use a trained model bundle:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_eegnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

Expected result:

- Brainstim window and VR page show the same phase.
- Online worker waits for the configured MI time window.
- FEEDBACK prints actual instruction and predicted result.
- The corresponding robot hand moves.

## Safety Notes

- Keep hands clear of the robot devices before any command with
  `--robot-mode serial` or `--move-hands`.
- Electric stimulation is disabled by default. Do not enable real stimulation
  without reviewed device protocol, safe amplitude/frequency limits, and manual
  confirmation.
- Stop the demo with `Ctrl+C` if the subject feels uncomfortable or if hardware
  behaves unexpectedly.
