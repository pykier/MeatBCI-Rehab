Put custom rehabilitation stimulus assets here.

Recommended names:

- left_robot_hand.png: default left image for `rehab_stim_demo.py`
- right_robot_hand.png: default right image for `rehab_stim_demo.py`
- left_hand_motion.gif: optional left action preview GIF
- right_hand_motion.gif: optional right action preview GIF
- left_vr_hand.png: optional left image for `vr_scene_server.py`
- right_vr_hand.png: optional right image for `vr_scene_server.py`

You can also pass explicit paths to `rehab_stim_demo.py`:

```powershell
python demos\rehab_mi\rehab_stim_demo.py --left-image demos\rehab_mi\assets\left_robot_hand.png --right-image demos\rehab_mi\assets\right_robot_hand.png
python demos\rehab_mi\rehab_stim_demo.py --left-gif demos\rehab_mi\assets\left_hand_motion.gif --right-gif demos\rehab_mi\assets\right_hand_motion.gif
```

The VR web scene automatically serves files under this folder at `/assets/<filename>`.
