import json
with open('phase1_real_trajectories.json') as f:
    data = json.load(f)

# Show a few real trajectories
print('REAL PATIENT TRAJECTORIES (from actual tumor segmentation masks):\n')
for i, (pid, traj) in enumerate(list(data.items())[:5]):
    vols_str = ', '.join([f'{v:.0f}' for v in traj['volumes_mm3']])
    print(f"{pid} ({traj['grade']}):")
    print(f"  Timepoints: {traj['timepoints']}")
    print(f"  Days:       {traj['days_since_baseline']}")
    print(f"  Volumes:    {vols_str} mm3")
    start_v = traj['volumes_mm3'][0]
    end_v = traj['volumes_mm3'][-1]
    pct_change = ((end_v - start_v) / start_v) * 100 if start_v > 0 else 0
    print(f"  Change:     {start_v:.0f} -> {end_v:.0f} mm3 ({pct_change:+.1f}%)")
    print()
