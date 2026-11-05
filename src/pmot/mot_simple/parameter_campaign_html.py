"""Self-contained browser report; loading and t intervals recompute in JS."""
import json

from scipy.stats import t as student_t


def write_html(payloads, status):
    from . import parameter_campaign as campaign
    from .campaign_io import replace_with_retry
    data = dict(status=status, updated=campaign.now(), total=len(campaign.CONFIGURATIONS),
                configurations=[campaign.config_record(i) for i in range(len(campaign.CONFIGURATIONS))],
                scans=campaign.SCAN_INDICES, views=campaign.VIEWS,
                fixed={k:campaign.fixed_text(k) for k in campaign.SCAN_INDICES},
                payloads={str(i):{**p,"critical":float(student_t.ppf(.975,len(p["captured_counts"])-1))}
                          for i,p in payloads.items()},
                mass_kg=campaign.VaporLoadingParameters().rb87_mass_kg)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    campaign.HTML.parent.mkdir(parents=True,exist_ok=True)
    temporary=campaign.HTML.with_suffix(".html.tmp")
    temporary.write_text(html,encoding="utf-8")
    replace_with_retry(temporary, campaign.HTML)


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two-level MOT · Loading parameter scans</title>
<style>
:root{color:#26343a;background:#f6f5f0;font:16px system-ui,sans-serif}*{box-sizing:border-box}
body{max-width:1150px;margin:auto;padding:45px 24px 80px}h1{font-size:clamp(26px,4vw,42px);letter-spacing:-1px;margin:8px 0 12px}
h2{font-size:23px;font-weight:600;margin:0 0 8px}.eyebrow{color:#607e7b;text-transform:uppercase;font-size:12px;letter-spacing:2px;font-weight:700}
p{line-height:1.65}.muted,.fixed{color:#627079;font-size:14px}.badge{display:inline-block;padding:6px 12px;border-radius:20px;background:#dfece5;font-size:13px}
.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap;padding:22px 25px;background:#e8efeb;border-radius:15px;margin:25px 0}
input{width:115px;padding:9px;border:1px solid #a5b9b1;border-radius:7px;background:white;font:inherit}
.card{padding:28px;margin-top:25px;background:#fffefa;border:1px solid #e8e7df;border-radius:17px;box-shadow:0 4px 16px #25362605}
svg{width:100%;height:auto;overflow:visible}.legend{display:flex;gap:20px;font-size:13px;color:#627079}.swatch{display:inline-block;width:23px;height:11px;border-radius:3px;margin-right:6px}
details{margin-top:18px}summary{cursor:pointer;font-size:14px;color:#627079}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}td,th{text-align:right;padding:8px;border-bottom:1px solid #eee}th:first-child,td:first-child{text-align:left}
.warning{color:#916323}nav{display:flex;gap:15px;flex-wrap:wrap;margin-top:20px}a{color:#527d7d;text-decoration:none}a:hover{text-decoration:underline}
</style>
<header><div class="eyebrow">Rb-87 · Conventional MOT · Effective two-level model</div>
<h1>How the MOT loads</h1><p>Four independent parameter scans, five views. Shaded pastel bands show pointwise <strong>95% Student-t confidence intervals</strong> across independent direction discs.</p>
<span id="status" class="badge"></span><nav id="nav"></nav></header>
<div class="controls"><label>Vapor temperature <input id="temperature" type="number" min="0.001" step="1" value="296.483"> K</label>
<span>Fixed Rb-87 partial pressure: <strong>10⁻¹⁰ Torr</strong><br><small id="density"></small></span></div>
<p class="muted">Changing temperature reweights the saved capture cross sections for every disc and recomputes the loading mean and confidence band. No trajectories are rerun. The pressure stays fixed, so number density changes as p/(k<sub>B</sub>T).</p>
<main id="plots"></main>
<section class="card"><h2>Simulation and statistics</h2>
<p>Six Gaussian cooling beams along ±x, ±y, ±z, using <strong>780.241209686 nm</strong>; linewidth 6.065 MHz and saturation intensity 1.669 mW/cm². Gravity acts in −z; recoil diffusion is absent. Single-beam center saturation is s₀ = 2P/(πw²I_sat), with s_eff = s₀/[1+(2Δ/linewidth)²]. The force law uses the sum of all six local saturations.</p>
<p>Incident directions are uniform over the full sphere. Launch points are uniform in area on 15 mm-radius discs centered 15 mm from the origin. Velocities on each disc are parallel to its inward normal. Cross sections are direction-averaged projected areas, with no additional 4π factor. Results describe this finite launch aperture and plane.</p>
<p>Capture requires 5 ms continuous residence within the 2 mm core, or two core entries separated by an exit. Production uses 5 µs RK4, with force and field recomputed at every stage. Required smaller-step and longer-time capture audits retain nonmonotone velocity masks. Unresolved positive-speed nodes prevent publication as complete; an authenticated indeterminate zero-speed node contributes only the exact loading-integrand anchor g(0)=0.</p>
<p>The independent pilot selects a fixed production sample size and is excluded from the reported intervals. Intervals describe Monte Carlo sampling uncertainty; they do not include physical-model or residual numerical error. A mean of zero does not establish a zero physical cross section or a useful relative confidence bound. The raw and effective saturation views share the same power data.</p>
<p id="sampling" class="muted"></p><p id="timestamp" class="muted"></p></section>
<script>
const DATA=__DATA__;
const ns='http://www.w3.org/2000/svg', kb=1.380649e-23, pressure=1e-10*133.32;
function tag(name,attrs={},text=''){const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e}
function rates(p,T){const density=pressure/(kb*T), scale=2*kb*T/DATA.mass_kg,
 pref=density*4*Math.PI*Math.pow(DATA.mass_kg/(2*Math.PI*kb*T),1.5);
 const values=p.captured_counts.map(row=>{let sum=0,oldV=0,oldG=0;
   p.velocity_m_per_s.forEach((v,j)=>{if(v>0&&row[j]===null)throw Error('Unresolved positive-speed capture node');
   const g=v===0?0:row[j]*p.disc_area_m2/p.points_per_disc*v*v*v*Math.exp(-v*v/scale);
   sum+=(v-oldV)*(g+oldG)/2;oldV=v;oldG=g;});return pref*sum;});
 const mean=values.reduce((a,b)=>a+b,0)/values.length,
 variance=values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1),
 half=p.critical*Math.sqrt(variance/values.length);return {mean,lower:mean-half,upper:mean+half,relative:mean>0?half/mean:null};}
function fmt(x){return Number(x).toLocaleString(undefined,{maximumSignificantDigits:5})}
function draw(view,T){const [key,family,field,xlabel,title,color,pastel]=view,card=document.createElement('section');card.className='card';card.id=key;
 const heading=document.createElement('h2');heading.textContent=title;card.append(heading);
 const fixed=document.createElement('p');fixed.className='fixed';fixed.textContent=DATA.fixed[family]+`; λ = 780.241209686 nm; vapor T = ${T} K; Rb-87 pressure = 10⁻¹⁰ Torr`;card.append(fixed);
 const rows=DATA.scans[family].filter(i=>DATA.payloads[i]).map(i=>({i,x:DATA.configurations[i][field],...rates(DATA.payloads[i],T)})).sort((a,b)=>a.x-b.x);
 const svg=tag('svg',{viewBox:'0 0 1000 495',role:'img','aria-label':title});
 const L=90,R=970,U=family==='power'?70:35,B=415;
 let min=rows.length?Math.min(...rows.map(r=>r.x)):0,max=rows.length?Math.max(...rows.map(r=>r.x)):1;
 if(min===max){min-=.5;max+=.5}const span=max-min;min-=span*.03;max+=span*.03;
 const ymax=Math.max(1,...rows.map(r=>r.upper/1e6))*1.12;
 const X=x=>L+(x-min)/(max-min)*(R-L),Y=y=>B-y/ymax*(B-U);
 for(let j=0;j<=5;j++){const y=ymax*j/5;svg.append(tag('line',{x1:L,x2:R,y1:Y(y),y2:Y(y),stroke:'#e8e9e4'}));svg.append(tag('text',{x:L-12,y:Y(y)+4,'text-anchor':'end',fill:'#627079','font-size':12},fmt(y)));}
 for(let j=0;j<=6;j++){const x=min+(max-min)*j/6;svg.append(tag('text',{x:X(x),y:B+25,'text-anchor':'middle',fill:'#627079','font-size':12},fmt(x)));}
 svg.append(tag('line',{x1:L,x2:R,y1:B,y2:B,stroke:'#a0aca9'}));
 svg.append(tag('text',{x:(L+R)/2,y:475,'text-anchor':'middle',fill:'#33454a','font-size':15},xlabel));
 svg.append(tag('text',{transform:'translate(22,240) rotate(-90)','text-anchor':'middle',fill:'#33454a','font-size':15},'Loading rate [10⁶ atoms/s]'));
 if(family==='power'&&rows.length){const factor=DATA.configurations[rows[0].i][field]/DATA.configurations[rows[0].i].power_mw_per_beam;
 svg.append(tag('text',{x:(L+R)/2,y:16,'text-anchor':'middle',fill:'#627079','font-size':13},'Cooling power per traveling beam [mW]'));
 for(let j=0;j<=6;j++){const x=min+(max-min)*j/6;svg.append(tag('text',{x:X(x),y:43,'text-anchor':'middle',fill:'#627079','font-size':12},fmt(x/factor)));}}
 if(rows.length){const poly=rows.map(r=>`${X(r.x)},${Y(Math.max(0,r.lower)/1e6)}`).concat([...rows].reverse().map(r=>`${X(r.x)},${Y(r.upper/1e6)}`)).join(' ');
 svg.append(tag('polygon',{points:poly,fill:pastel,'fill-opacity':.6}));svg.append(tag('polyline',{points:rows.map(r=>`${X(r.x)},${Y(r.mean/1e6)}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5}));
 rows.forEach(r=>{const circle=tag('circle',{cx:X(r.x),cy:Y(r.mean/1e6),r:4,fill:color});circle.append(tag('title',{},`${xlabel}: ${fmt(r.x)}\nLoading: ${fmt(r.mean)} atoms/s\n95% CI: ${fmt(r.lower)} to ${fmt(r.upper)}\nRelative half-width: ${r.relative===null?'undefined':fmt(100*r.relative)+'%'}`));svg.append(circle);});}
 card.append(svg);const legend=document.createElement('div');legend.className='legend';legend.innerHTML=`<span><i class="swatch" style="background:${color}"></i>Mean loading rate</span><span><i class="swatch" style="background:${pastel};opacity:.65"></i>95% Student-t interval</span>`;card.append(legend);
 const progress=document.createElement('p');progress.className='muted';const under=rows.filter(r=>r.relative!==null&&r.relative<.1).length;
 progress.textContent=`${rows.length}/${DATA.scans[family].length} scan points complete · ${under}/${rows.length} have relative 95% half-width below 10%.`;card.append(progress);
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='View plotted values and precision';details.append(summary);
 const table=document.createElement('table');table.innerHTML='<thead><tr><th>Scan value</th><th>Loading [atoms/s]</th><th>95% lower</th><th>95% upper</th><th>Relative half-width</th></tr></thead>';
 for(const r of rows){const tr=document.createElement('tr');[fmt(r.x),fmt(r.mean),fmt(r.lower),fmt(r.upper),r.relative===null?'undefined':fmt(r.relative*100)+'%'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});table.append(tr)}details.append(table);card.append(details);document.getElementById('plots').append(card);}
function update(){const T=Number(document.getElementById('temperature').value);if(!(T>0))return;
document.getElementById('plots').replaceChildren();for(const v of DATA.views)draw(v,T);
document.getElementById('density').textContent=`Rb-87 number density: ${fmt(pressure/(kb*T))} m⁻³`;}
document.getElementById('status').textContent=`${DATA.status.replaceAll('_',' ')} · ${Object.keys(DATA.payloads).length}/${DATA.total} unique configurations`;
for(const view of DATA.views){const a=document.createElement('a');a.href='#'+view[0];a.textContent=view[4].replace('Loading rate vs ','');document.getElementById('nav').append(a)}
const first=Object.values(DATA.payloads)[0],discCounts=Object.values(DATA.payloads).map(p=>p.captured_counts.length);document.getElementById('sampling').textContent=first?`Production geometry: ${Math.min(...discCounts)}–${Math.max(...discCounts)} discs × ${first.points_per_disc} points/disc. Each point uses its own Student-t degrees of freedom (disc count minus one). All geometries are prefixes of one saved master geometry.`:'Production geometry will appear after the timing pilot.';
document.getElementById('timestamp').textContent='Report updated '+DATA.updated+'. Refresh the file to read newly completed scan points.';
document.getElementById('temperature').addEventListener('input',update);update();
</script></html>'''
