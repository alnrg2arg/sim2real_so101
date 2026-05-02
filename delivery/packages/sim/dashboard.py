"""SAC Training Dashboard — Squint-exact with Chart + Stage + Lift Stats."""
import io, json, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np
from PIL import Image

cam_frames = {}
cam_lock = threading.Lock()
train_stats = {}
stats_lock = threading.Lock()


def _enc(img, q=70):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return b.getvalue()


def update_cameras(cm):
    for n, c in cm.items():
        try:
            r = c.data.output.get("rgb")
            if r is None:
                continue
            i = r[0, :, :, :3].cpu().numpy()
            if i.dtype != np.uint8:
                i = (i * 255).clip(0, 255).astype(np.uint8)
            with cam_lock:
                cam_frames[n] = _enc(i)
        except:
            pass


def _safe_json(obj):
    txt = json.dumps(obj, default=str)
    txt = txt.replace("NaN", "0").replace("Infinity", "999999").replace("-Infinity", "0")
    return txt


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/stats":
            with stats_lock:
                d = _safe_json(train_stats)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(d.encode())
        elif self.path.startswith("/cam/"):
            name = self.path.split("/cam/")[-1]
            with cam_lock:
                data = cam_frames.get(name)
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(204)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(_HTML.encode())


class S(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start(port):
    s = S(("0.0.0.0", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"[HTTP] Dashboard: http://0.0.0.0:{port}", flush=True)


_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>SAC Squint — SO101</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,'Segoe UI',monospace;padding:16px}
h1{font-size:20px;margin-bottom:16px;color:#a371f7}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.grid6{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}
.label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.val{font-size:26px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.blue{color:#58a6ff}.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}.purple{color:#a371f7}

.row2{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card-title{font-size:12px;color:#8b949e;text-transform:uppercase;margin-bottom:10px;letter-spacing:.5px}

.stage-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #21262d}
.stage-row:last-child{border-bottom:none}
.stage-icon{font-size:18px;width:28px;text-align:center}
.stage-name{font-size:13px;font-weight:600;flex:1}
.stage-bar-wrap{width:100px;height:7px;background:#21262d;border-radius:4px;overflow:hidden}
.stage-bar{height:100%;border-radius:4px;transition:width .5s}
.stage-val{font-size:12px;font-weight:700;width:55px;text-align:right;font-variant-numeric:tabular-nums}

.ep-list{max-height:180px;overflow-y:auto;font-size:12px}
.ep{display:flex;justify-content:space-between;padding:3px 6px;border-radius:4px;margin-bottom:2px}
.ep:nth-child(odd){background:#0d1117}
.ep-r{font-weight:600;font-variant-numeric:tabular-nums}

svg{width:100%;height:180px;display:block}
.progress-wrap{background:#21262d;border-radius:4px;height:8px;overflow:hidden;margin-bottom:12px}
.progress-fill{height:100%;border-radius:4px;background:#58a6ff;transition:width .5s}
#updated{font-size:11px;color:#484f58;margin-top:8px;text-align:right}
</style>
</head><body>
<h1>SAC Squint — SO101 LiftCube</h1>

<div class="grid">
<div class="box"><div class="label">Steps</div><div class="val blue" id="v-step">-</div></div>
<div class="box"><div class="label">Episodes</div><div class="val blue" id="v-ep">-</div></div>
<div class="box"><div class="label">Mean Reward</div><div class="val yellow" id="v-mean">-</div></div>
<div class="box"><div class="label">Saved</div><div class="val green" id="v-suc">0</div></div>
</div>
<div class="grid6">
<div class="box"><div class="label">Best Reward</div><div class="val green" id="v-best">-</div></div>
<div class="box"><div class="label">Best Lift</div><div class="val purple" id="v-lift">0.0 cm</div></div>
<div class="box"><div class="label">Best Dist-to-Rest</div><div class="val purple" id="v-d2r">-</div></div>
</div>

<div class="progress-wrap"><div class="progress-fill" id="p-bar" style="width:0%"></div></div>


<div style="display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-bottom:16px">
<div class="card">
  <div class="card-title">Reward History</div>
  <svg id="chart" viewBox="0 0 800 180" preserveAspectRatio="none"></svg>
</div>
<div class="card">
  <div class="card-title">Reward Stages (per step)</div>
  <div id="stages">
    <div class="stage-row"><div class="stage-icon">&#127919;</div><div class="stage-name">Reaching</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-reaching" style="width:0%;background:#58a6ff"></div></div><div class="stage-val blue" id="sv-reaching">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#128275;</div><div class="stage-name">Open</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-approachopen" style="width:0%;background:#a371f7"></div></div><div class="stage-val purple" id="sv-approachopen">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#128260;</div><div class="stage-name">Retry</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-retry" style="width:0%;background:#d2a8ff"></div></div><div class="stage-val purple" id="sv-retry">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#9994;</div><div class="stage-name">Grasped</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-grasped" style="width:0%;background:#d29922"></div></div><div class="stage-val yellow" id="sv-grasped">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#128170;</div><div class="stage-name">Lift</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-lifthold" style="width:0%;background:#3fb950"></div></div><div class="stage-val green" id="sv-lifthold">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#129518;</div><div class="stage-name">Fold (pan&lt;5°)</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-ps1" style="width:0%;background:#2ea043"></div></div><div class="stage-val green" id="sv-ps1">0</div><span style="font-size:10px;color:#8b949e;margin-left:4px" id="sv-fsum"></span></div>
    <div class="stage-row"><div class="stage-icon">&#128274;</div><div class="stage-name">Fold Hold (50% save / 95% target)</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-foldhold" style="width:0%;background:#238636"></div></div><div class="stage-val green" id="sv-foldhold">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#9888;</div><div class="stage-name">Table</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-table" style="width:0%;background:#f85149"></div></div><div class="stage-val red" id="sv-table">0.000</div></div>
    <div class="stage-row"><div class="stage-icon">&#11015;</div><div class="stage-name">Not-Lifted</div><div class="stage-bar-wrap"><div class="stage-bar" id="sb-notlifted" style="width:0%;background:#f0883e"></div></div><div class="stage-val" style="color:#f0883e" id="sv-notlifted">0.000</div></div>
  </div>
</div>
<div class="card">
  <div class="card-title">Cumulative Stages</div>
  <div id="cum-stages">
    <div class="stage-row"><div class="stage-icon">&#127919;</div><div class="stage-name">Reaching</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-reaching" style="width:0%;background:#58a6ff"></div></div><div class="stage-val blue" id="cv-reaching">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#128275;</div><div class="stage-name">Open</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-open" style="width:0%;background:#a371f7"></div></div><div class="stage-val purple" id="cv-open">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#128260;</div><div class="stage-name">Retry</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-retry" style="width:0%;background:#d2a8ff"></div></div><div class="stage-val purple" id="cv-retry">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#9994;</div><div class="stage-name">Grasped</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-grasped" style="width:0%;background:#d29922"></div></div><div class="stage-val yellow" id="cv-grasped">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#128170;</div><div class="stage-name">Lift</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-lift" style="width:0%;background:#3fb950"></div></div><div class="stage-val green" id="cv-lift">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#129518;</div><div class="stage-name">Fold (pan&lt;5°)</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-pn" style="width:0%;background:#2ea043"></div></div><div class="stage-val green" id="cv-pn">0</div></div>
    <div class="stage-row"><div class="stage-icon">&#128190;</div><div class="stage-name">Saved (fold 50% held)</div><div class="stage-bar-wrap"><div class="stage-bar" id="cb-saved" style="width:0%;background:#a371f7"></div></div><div class="stage-val purple" id="cv-saved">0</div></div>
  </div>
</div>
</div>

<div class="row2">
<div class="card">
  <div class="card-title">Recent Episodes</div>
  <div class="ep-list" id="ep-list"></div>
</div>
<div class="card">
  <div class="card-title">Stats</div>
  <div style="font-size:13px;line-height:2">
    <div>Total Steps: <span id="cs-steps" style="font-weight:700;color:#58a6ff">-</span></div>
    <div>Total Episodes: <span id="cs-eps" style="font-weight:700;color:#58a6ff">-</span></div>
    <div>All-time Best: <span id="cs-best" style="font-weight:700;color:#3fb950">-</span></div>
    <div>Best Lift: <span id="cs-lift" style="font-weight:700;color:#a371f7">-</span></div>
    <div>Best D2R: <span id="cs-d2r" style="font-weight:700;color:#a371f7">-</span></div>
    <div>Saved: <span id="cs-suc" style="font-weight:700;color:#3fb950">0</span></div>
  </div>
</div>
</div>

<div id="updated"></div>

<script>
// Cumulative stage counters
var _cumReaching=0,_cumOpen=0,_cumRetry=0,_cumGrasped=0,_cumPlace=0,_cumLifted=0,_cumFolded=0,_cumFoldHold=0,_cumTotal=0;
function n(v){return(typeof v==='number'&&isFinite(v))?v:0}

function drawChart(hist){
  var svg=document.getElementById('chart');
  if(!hist||hist.length<2){svg.innerHTML='<text x="400" y="90" text-anchor="middle" fill="#484f58" font-size="14">Waiting for data...</text>';return}
  var W=800,H=180,pad=5;
  var means=hist.map(function(h){return n(h.mean)});
  var maxes=hist.map(function(h){return n(h.max)});
  var all=means.concat(maxes);
  var yMin=Math.min.apply(null,all);var yMax=Math.max.apply(null,all);
  if(yMax-yMin<0.01){yMax=yMin+1}
  var xS=function(i){return pad+(W-2*pad)*i/(hist.length-1)};
  var yS=function(v){return H-pad-(H-2*pad)*(n(v)-yMin)/(yMax-yMin)};
  var mP=means.map(function(v,i){return xS(i)+','+yS(v)}).join(' ');
  var xP=maxes.map(function(v,i){return xS(i)+','+yS(v)}).join(' ');
  var gY='';var steps=4;
  for(var g=0;g<=steps;g++){var gy=pad+(H-2*pad)*g/steps;var gv=(yMax-(yMax-yMin)*g/steps).toFixed(2);
    gY+='<line x1="'+pad+'" y1="'+gy+'" x2="'+(W-pad)+'" y2="'+gy+'" stroke="#21262d"/>';
    gY+='<text x="'+(W-pad+4)+'" y="'+(gy+4)+'" fill="#484f58" font-size="9">'+gv+'</text>';}
  svg.innerHTML=gY+'<polyline points="'+xP+'" fill="none" stroke="#3fb950" stroke-width="1.5" opacity="0.5"/>'
    +'<polyline points="'+mP+'" fill="none" stroke="#a371f7" stroke-width="2.5"/>'
    +'<text x="10" y="14" fill="#a371f7" font-size="10">Mean</text>'
    +'<text x="60" y="14" fill="#3fb950" font-size="10">Best</text>';
}

function setStage(key,val){
  var v=n(val);var el=document.getElementById('sv-'+key);var bar=document.getElementById('sb-'+key);
  if(el)el.textContent=Math.abs(v)<0.001&&v!==0?v.toExponential(1):v.toFixed(4);
  if(bar){var pct=Math.min(Math.abs(v)*100,100);bar.style.width=pct+'%';}
}

function update(){
  fetch('/api/stats').then(function(r){return r.json()}).then(function(s){
    document.getElementById('v-step').textContent=n(s.iteration).toLocaleString();
    document.getElementById('v-ep').textContent=n(s.episode_count).toLocaleString();
    document.getElementById('v-mean').textContent=n(s.mean_reward).toFixed(2);
    document.getElementById('v-best').textContent=n(s.alltime_max_reward).toFixed(2);
    document.getElementById('v-suc').textContent=n(s.saved_count);
    document.getElementById('v-lift').textContent=n(s.best_lift_cm).toFixed(1)+' cm';
    var d2r=n(s.best_dist_to_rest);
    document.getElementById('v-d2r').textContent=d2r<100?d2r.toFixed(3):'--';

    var maxIter=n(s.max_iterations)||100000000000;
    var pct=Math.min(100,n(s.iteration)/maxIter*100);
    document.getElementById('p-bar').style.width=pct.toFixed(2)+'%';

    var rt=s.reward_terms||{};
    var rv=function(k){var t=rt[k];if(!t)return 0;return typeof t==='object'?n(t.value):n(t)};
    setStage('reaching',rv('reaching'));
    setStage('grasped',rv('grasped'));
    setStage('table',rv('table_penalty'));
    setStage('approachopen',rv('approach_open'));
    setStage('retry',rv('grasp_retry'));
    setStage('notlifted',rv('not_lifted'));
    setStage('lifthold',rv('lift_hold'));
    // Per-step fold stages (env count per stage)
    var fs=s.fold_stage||{};
    function setPS(id,v,total){
      var e=document.getElementById('sv-'+id);var b=document.getElementById('sb-'+id);
      if(e)e.textContent=v+'/'+total;
      if(b)b.style.width=Math.min(v/Math.max(total,1)*100,100)+'%';
    }
    var N=n(s.num_envs)||2048;
    setPS('ps1',n(fs.fold_ok),N);
    var fse=document.getElementById('sv-fsum'); if(fse)fse.textContent='pan:'+n(fs.pan_err).toFixed(3);
    setStage('fold',rv('fold'));
    setStage('foldhold',rv('fold_hold'));

    drawChart(s.reward_history||[]);

    var epList=document.getElementById('ep-list');
    var eps=s.recent_episodes||[];
    epList.innerHTML=eps.map(function(e){
      var r=n(e.reward);var cls=r>1.5?'color:#3fb950':r>0?'color:#d29922':'color:#f85149';
      return '<div class="ep"><span>EP #'+e.ep+'</span><span class="ep-r" style="'+cls+'">'+r.toFixed(3)+'</span></div>';
    }).join('');

    document.getElementById('cs-steps').textContent=n(s.iteration).toLocaleString();
    document.getElementById('cs-eps').textContent=n(s.episode_count).toLocaleString();
    document.getElementById('cs-best').textContent=n(s.alltime_max_reward).toFixed(3);
    document.getElementById('cs-lift').textContent=n(s.best_lift_cm).toFixed(1)+' cm';
    document.getElementById('cs-d2r').textContent=d2r<100?d2r.toFixed(3):'--';
    document.getElementById('cs-suc').textContent=n(s.saved_count);

    // Update cumulative stage counters
    // Server-side cumulative (persists across refresh)
    var cum=s.cumulative||{};
    var suc=n(s.saved_count);
    var _cr=n(cum.reaching),_co=n(cum.open),_crt=n(cum.retry),_cg=n(cum.grasped);
    var _cl=n(cum.lifted);
    var _cpn=n(cum.fold_ok);
    var total=n(s.iteration);
    var _cumTotal=n(cum.reaching)||1;
    function setCum(id,v){
      var e=document.getElementById(id);
      if(!e)return;
      var pct=(_cumTotal>0)?(v/_cumTotal*100).toFixed(1):'0';
      e.textContent=v.toLocaleString()+' ('+pct+'%)';
    }
    function setCumBar(id,v,mx){var e=document.getElementById(id);if(e)e.style.width=Math.min(v/Math.max(mx,1)*100,100)+'%'}
    var mx=Math.max(_cr,1);
    setCum('cv-reaching',_cr); setCumBar('cb-reaching',_cr,mx);
    setCum('cv-open',_co); setCumBar('cb-open',_co,mx);
    setCum('cv-retry',_crt); setCumBar('cb-retry',_crt,mx);
    setCum('cv-grasped',_cg); setCumBar('cb-grasped',_cg,mx);
    setCum('cv-lift',_cl); setCumBar('cb-lift',_cl,mx);
    setCum('cv-pn',_cpn); setCumBar('cb-pn',_cpn,mx);
    setCum('cv-saved',suc); setCumBar('cb-saved',suc,mx);

    document.getElementById('updated').textContent='Updated: '+new Date().toLocaleTimeString();
  }).catch(function(e){document.getElementById('updated').textContent='Error: '+e;});
}
function updateAllCams(){
  ['front','side','wrist'].forEach(function(name){
    var img=document.getElementById('cam-'+name);
    if(img) img.src='/cam/'+name+'?t='+Date.now();
  });
}
try{update()}catch(e){}
setInterval(function(){try{update()}catch(e){}},2000);
</script>
</body></html>"""
