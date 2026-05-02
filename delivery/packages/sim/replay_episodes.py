#!/usr/bin/env python3
"""Replay saved episodes as videos + web viewer.

Usage:
    python3 replay_episodes.py --episodes-dir /data/rl_output/episodes --output-dir /data/rl_output/replays

Outputs per episode: front.mp4, side.mp4, wrist.mp4, grid.mp4, meta.json
Web viewer on port 8890
"""
import argparse, os, sys, json, subprocess
import numpy as np
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

def make_video(frames, output_path, fps=15):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo","-s",f"{w}x{h}",
           "-pix_fmt","rgb24","-r",str(fps),"-i","-","-an","-vcodec","libx264",
           "-pix_fmt","yuv420p","-crf","23","-preset","fast",str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

def make_grid(front, side, wrist):
    h, w = front.shape[:2]
    hh, hw = h//2, w//2
    s = side[::2,::2][:hh,:hw]
    wr = wrist[::2,::2][:hh,:hw]
    grid = np.zeros((h+hh, w, 3), dtype=np.uint8)
    grid[:h,:w] = front
    grid[h:h+hh,:hw] = s
    grid[h:h+hh,hw:hw*2] = wr
    return grid

def process_episode(ep_dir, output_dir):
    ep_name = ep_dir.name
    print(f"  {ep_name}...", end=" ", flush=True)
    ff_list = sorted(ep_dir.glob("frame_*.npz"))
    fronts, sides, wrists, cubes = [], [], [], []
    for ff in ff_list:
        f = np.load(str(ff))
        if "cam_front" in f: fronts.append(f["cam_front"])
        if "cam_side" in f: sides.append(f["cam_side"])
        if "cam_wrist" in f: wrists.append(f["cam_wrist"])
        cubes.append(f.get("cube_pos", np.zeros(3)))
    if not fronts:
        print("no cam data"); return None
    ep_out = output_dir / ep_name
    ep_out.mkdir(parents=True, exist_ok=True)
    make_video(fronts, ep_out/"front.mp4")
    if sides: make_video(sides, ep_out/"side.mp4")
    if wrists: make_video(wrists, ep_out/"wrist.mp4")
    if sides and wrists and len(fronts)==len(sides)==len(wrists):
        grids = [make_grid(f,s,w) for f,s,w in zip(fronts,sides,wrists)]
        make_video(grids, ep_out/"grid.mp4")
    meta = {"episode":ep_name,"num_frames":len(fronts),"duration_s":round(len(fronts)/15.0,2),
            "max_lift_cm":round((max(c[2] for c in cubes)-0.056)*100,2) if cubes else 0}
    with open(ep_out/"meta.json","w") as mf: json.dump(meta,mf,indent=2)
    print(f"{len(fronts)}f, {meta['duration_s']}s, lift={meta['max_lift_cm']}cm")
    return meta

def gen_html(out_dir, metas):
    cards=""
    for m in metas:
        n=m["episode"]
        cards+=f'<div class="card"><h3>{n}</h3><p>{m["num_frames"]}f | {m["duration_s"]}s | lift={m["max_lift_cm"]}cm</p><div class="vids"><div><label>Front</label><video src="{n}/front.mp4" controls muted loop></video></div><div><label>Side</label><video src="{n}/side.mp4" controls muted loop></video></div><div><label>Wrist</label><video src="{n}/wrist.mp4" controls muted loop></video></div></div><div class="gv"><label>Grid</label><video src="{n}/grid.mp4" controls muted loop width="100%"></video></div></div>'
    html=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Episode Replay</title>
<style>body{{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:20px}}h1{{color:#58a6ff}}.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}.card h3{{color:#3fb950}}.vids{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:8px 0}}.vids video{{width:100%;border-radius:4px}}.vids label,.gv label{{color:#8b949e;font-size:12px;display:block;margin-bottom:4px}}.gv{{margin-top:8px}}.btn{{background:#238636;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;margin:4px}}.btn:hover{{background:#2ea043}}</style></head>
<body><h1>Episode Replay ({len(metas)} episodes)</h1>
<button class="btn" onclick="document.querySelectorAll('video').forEach(v=>v.play())">Play All</button>
<button class="btn" onclick="document.querySelectorAll('video').forEach(v=>{{v.pause();v.currentTime=0}})">Reset All</button>
{cards}</body></html>'''
    with open(out_dir/"index.html","w") as f: f.write(html)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes-dir",default="/data/rl_output/episodes")
    p.add_argument("--output-dir",default="/data/rl_output/replays")
    p.add_argument("--port",type=int,default=8890)
    p.add_argument("--no-serve",action="store_true")
    a=p.parse_args()
    ep_dir,out_dir=Path(a.episodes_dir),Path(a.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    eps=sorted([d for d in ep_dir.iterdir() if d.is_dir()])
    print(f"Found {len(eps)} episodes")
    metas=[m for d in eps if (m:=process_episode(d,out_dir))]
    gen_html(out_dir,metas)
    print(f"\nDone! {len(metas)} eps -> {out_dir}")
    if not a.no_serve:
        print(f"Serving http://0.0.0.0:{a.port}/")
        os.chdir(str(out_dir))
        HTTPServer(("0.0.0.0",a.port),SimpleHTTPRequestHandler).serve_forever()

if __name__=="__main__": main()
