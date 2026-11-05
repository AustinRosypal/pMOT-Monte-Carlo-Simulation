"""Self-contained browser renderer for physical and paired MOT reports."""

TEMPLATE=r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched MOT loading scans</title>
<style>
:root{color:#28343b;background:#f6f5f0;font:16px system-ui,sans-serif}*{box-sizing:border-box}
body{max-width:1200px;margin:auto;padding:42px 25px 80px}h1{font-size:clamp(27px,4vw,42px);letter-spacing:-1px;margin:10px 0}
h2{font-size:22px;font-weight:600;margin:0 0 10px}p{line-height:1.65}.eyebrow{font-size:12px;letter-spacing:2px;color:#776285;font-weight:700;text-transform:uppercase}
.muted,.fixed{font-size:14px;color:#64717b}.badge{display:inline-block;background:#e8e0ed;border-radius:25px;padding:7px 12px;font-size:13px}
.controls{margin:25px 0;padding:22px;background:#e9e4ee;border-radius:15px;display:flex;gap:25px;align-items:center;flex-wrap:wrap}
input[type=number]{font:inherit;width:120px;padding:9px;border:1px solid #baa6c3;border-radius:7px}
nav{display:flex;gap:15px;flex-wrap:wrap;margin-top:20px}a{color:#6d5481;text-decoration:none}a:hover{text-decoration:underline}
.card{background:#fffefa;border:1px solid #e7e5dd;border-radius:17px;padding:28px;margin-top:25px;box-shadow:0 4px 15px #25362605}
svg{width:100%;height:auto;overflow:visible}.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;color:#62717b}
.swatch{display:inline-block;width:24px;height:11px;border-radius:3px;margin-right:6px}summary{cursor:pointer;color:#62717b;font-size:14px}
details{margin-top:18px}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:13px}td,th{text-align:right;padding:8px;border-bottom:1px solid #ecebe5}td:first-child,th:first-child{text-align:left}
.hint{font-size:13px;color:#776285}.scale{font-size:13px;color:#62717b;float:right}
</style>
<header><div class="eyebrow">Rb-87 · Conventional MOT · Matched launch geometry</div><h1 id="title"></h1>
<p id="intro"></p><span id="status" class="badge"></span><nav id="nav"></nav></header>
<section class="controls"><label>Vapor temperature <input type="number" id="temperature" value="296.483" min="0.001" step="1"> K</label>
<div>Fixed Rb-87 partial pressure: <strong>10⁻¹⁰ Torr</strong><br><small id="density"></small></div></section>
<p class="muted">Temperature changes reweight every saved direction-disc spectrum and recompute its loading rate and pointwise 95% Student-t interval. Number density changes as p/(k<sub>B</sub>T); trajectories are reused.</p>
<p class="hint">Cooling wavelength: 780.241209686 nm. All powers are per traveling beam; diameters are Gaussian 1/e² intensity diameters. Hover over a point for its values, or open its data table.</p>
<main id="plots"></main>
<section class="card"><h2>Physics, sampling, and comparison</h2>
<p>The physical multilevel model resolves 8 ground and 16 excited hyperfine-Zeeman states. ARC supplies dipoles, transition frequencies, pairwise spontaneous rates and Landé factors before trajectory calculations. Stimulated coefficients are W = Γ<sub>e</sub>|Ω|²/(Γ<sub>e</sub>²+4Δ²); saturation arises from coupled populations. Each beam force uses the net absorption-minus-stimulated-emission rate W(p<sub>g</sub>−p<sub>e</sub>).</p>
<p>Cooling detuning is relative to F=2 → F′=3. Repump detuning is relative to F=1 → F′=2; zero means on resonance. The six repump components follow the same Cartesian directions as cooling. Their default power is 0.1 mW each and diameter 12.7 mm. The two cooling-diameter studies hold repump diameter fixed or match it to cooling diameter, while holding both powers fixed.</p>
<p>Common scans reuse a predeclared subset of the exact two-level launch coordinates. Both models are reanalyzed on the same direction discs and points. Independent pilot and held-out validation discs are excluded from production confidence intervals. Directions cover the full sphere; launch points are uniform in area on 15 mm-radius discs whose centers are 15 mm from the origin. Velocities on each disc are parallel to the inward normal. The launch aperture and plane remain finite. No extra octant or 4π factor is applied.</p>
<p>Gravity is enabled; recoil diffusion is absent. Adaptive Dormand–Prince integration recomputes the full physical problem at every stage. Fixed RK4 with the original dual-step ladder audits capture boundaries, disagreements, exact zero speed and subsequent confinement. Full fixed velocity masks on independent held-out rays validate every parameter setting. Capture uses 5 ms continuous residence within the 2 mm core OR two entries separated by an exit. Capture boundaries and exceptional direct velocity masks receive the recorded dual-timestep and longer-duration audits.</p>
<p>The two saturation plots are nominal single-beam center quantities using the benchmark I_sat = 1.669 mW/cm² and linewidth 6.065 MHz. They share the same power trajectories. The physical kernel retains ARC-derived state-dependent decay rates and resolved Zeeman/polarization physics; its restoring propagation-frame helicities are sigma+ on x/y and sigma− on z in both directions. These are the physical realization of the two-level model’s effective axis signs.</p>
<p>Intervals describe Monte Carlo uncertainty across independent direction discs, not model error or residual numerical error. A zero sampled mean does not establish a zero physical cross section or a useful relative error bound. The physical model assumes instantaneous local steady-state populations and incoherent beams. Frozen-environment event checks validate its rate kernel, not a recoil trajectory or adiabatic population-relaxation model. Weak-wing population-relaxation limitations remain relevant. No atomic-temperature or equilibrium-temperature claim is made.</p>
<p id="timestamp" class="muted"></p></section>
<script>
const DATA=__DATA__;
const ns='http://www.w3.org/2000/svg',kb=1.380649e-23,pressure=1e-10*133.32;
function tag(name,attrs={},text=''){const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e}
function rates(p,T){if(!p._counts){const raw=atob(p.counts_base64);p._counts=Uint8Array.from(raw,c=>c.charCodeAt(0));}
 const v=p.velocity_m_per_s,nv=v.length,pref=pressure/(kb*T)*4*Math.PI*Math.pow(DATA.mass_kg/(2*Math.PI*kb*T),1.5),thermal=2*kb*T/DATA.mass_kg;
 const values=[];for(let d=0;d<p.disc_count;d++){let integral=0,oldV=0,oldG=0;
 for(let j=0;j<nv;j++){const count=p._counts[d*nv+j],speed=v[j];if(speed>0&&count===255)throw Error('Unresolved positive-speed node');
 const g=speed===0?0:count*p.disc_area_m2/p.points_per_disc*speed**3*Math.exp(-(speed**2)/thermal);
 integral+=(speed-oldV)*(g+oldG)/2;oldV=speed;oldG=g;}values.push(pref*integral);}
 const mean=values.reduce((a,b)=>a+b,0)/values.length,variance=values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1),half=p.critical*Math.sqrt(variance/values.length);
 return {mean,lower:mean-half,upper:mean+half,relative:mean>0?half/mean:null};}
function fmt(x){return Number(x).toLocaleString(undefined,{maximumSignificantDigits:5})}
function niceStep(value){const power=10**Math.floor(Math.log10(value)),ratio=value/power;return (ratio<=1?1:ratio<=2?2:ratio<=2.5?2.5:ratio<=5?5:10)*power;}
function draw(view,T){const card=document.createElement('section');card.className='card';card.id=view.id;
 const heading=document.createElement('h2');heading.textContent=view.title;card.append(heading);
 const fixed=document.createElement('p');fixed.className='fixed';fixed.textContent=view.fixed+`; vapor T = ${T} K; Rb-87 pressure = 10⁻¹⁰ Torr`;card.append(fixed);
 const series=view.series.map(s=>({...s,rows:s.ids.filter(i=>DATA.datasets[s.model][i]).map(i=>({i,x:DATA.points[i][view.field],...rates(DATA.datasets[s.model][i],T)})).sort((a,b)=>a.x-b.x)}));
 const all=series.flatMap(s=>s.rows),svg=tag('svg',{viewBox:'0 0 1000 500',role:'img','aria-label':view.title});
 const L=92,R=970,U=view.family==='cooling_power'?75:35,B=412;
 let min=all.length?Math.min(...all.map(r=>r.x)):0,max=all.length?Math.max(...all.map(r=>r.x)):1;if(min===max){min-=.5;max+=.5}
 const span=max-min;min-=span*.03;max+=span*.03;
 const ystep=niceStep(Math.max(1,...all.map(r=>r.upper/1e6))*1.1/5),ymax=Math.ceil(Math.max(1,...all.map(r=>r.upper/1e6))*1.1/ystep)*ystep;
 const X=x=>L+(x-min)/(max-min)*(R-L),Y=y=>B-y/ymax*(B-U);
 for(let y=0;y<=ymax+ystep*.01;y+=ystep){svg.append(tag('line',{x1:L,x2:R,y1:Y(y),y2:Y(y),stroke:'#e9e9e3'}));svg.append(tag('text',{x:L-12,y:Y(y)+4,'text-anchor':'end',fill:'#62717b','font-size':12},fmt(y)));}
 const xstep=niceStep((max-min)/6);for(let x=Math.ceil(min/xstep)*xstep;x<=max+xstep*.001;x+=xstep){svg.append(tag('text',{x:X(x),y:B+25,'text-anchor':'middle',fill:'#62717b','font-size':12},fmt(x)));}
 svg.append(tag('line',{x1:L,x2:R,y1:B,y2:B,stroke:'#a2aca8'}));svg.append(tag('text',{x:(L+R)/2,y:476,'text-anchor':'middle',fill:'#33454a','font-size':15},view.label));
 svg.append(tag('text',{transform:'translate(23,240) rotate(-90)','text-anchor':'middle',fill:'#33454a','font-size':15},'Loading rate [10⁶ atoms/s]'));
 if(view.family==='cooling_power'){const first=DATA.points[view.series[0].ids[0]],factor=first[view.field]/first.cooling_power_mw,pstep=niceStep((max-min)/factor/6);
 svg.append(tag('text',{x:(L+R)/2,y:17,'text-anchor':'middle',fill:'#62717b','font-size':13},'Cooling power per traveling beam [mW]'));
 for(let p=Math.ceil(min/factor/pstep)*pstep;p<=max/factor;p+=pstep)svg.append(tag('text',{x:X(p*factor),y:46,'text-anchor':'middle',fill:'#62717b','font-size':12},fmt(p)));}
 const legend=document.createElement('div');legend.className='legend';
 for(const s of series){const [color,pastel,label]=DATA.colors[s.model],rows=s.rows;if(!rows.length)continue;
 const polygon=rows.map(r=>`${X(r.x)},${Y(Math.max(0,r.lower)/1e6)}`).concat([...rows].reverse().map(r=>`${X(r.x)},${Y(r.upper/1e6)}`)).join(' ');
 svg.append(tag('polygon',{points:polygon,fill:pastel,'fill-opacity':.55}));svg.append(tag('polyline',{points:rows.map(r=>`${X(r.x)},${Y(r.mean/1e6)}`).join(' '),fill:'none',stroke:color,'stroke-width':2.4}));
 for(const r of rows){const dot=tag('circle',{cx:X(r.x),cy:Y(r.mean/1e6),r:3.8,fill:color});dot.append(tag('title',{},`${label}\n${view.label}: ${fmt(r.x)}\nLoading: ${fmt(r.mean)} atoms/s\n95% CI: ${fmt(r.lower)} to ${fmt(r.upper)}\nRelative half-width: ${r.relative===null?'undefined':fmt(r.relative*100)+'%'}`));svg.append(dot)}
 const item=document.createElement('span');item.innerHTML=`<i class="swatch" style="background:${color}"></i>${label} · shaded 95% t interval`;legend.append(item);}
 card.append(svg,legend);const physical=series.find(s=>s.model==='multilevel'),progress=document.createElement('p');progress.className='muted';
 progress.textContent=`${physical.rows.length}/${physical.ids.length} multilevel scan points complete · ${physical.rows.filter(r=>r.relative!==null&&r.relative<.1).length}/${physical.rows.length} have relative 95% half-width below 10%.`;card.append(progress);
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='View plotted values and precision';details.append(summary);
 const table=document.createElement('table');table.innerHTML='<thead><tr><th>Model</th><th>Scan value</th><th>Loading [atoms/s]</th><th>95% lower</th><th>95% upper</th><th>Relative half-width</th></tr></thead>';
 for(const s of series)for(const r of s.rows){const tr=document.createElement('tr');[DATA.colors[s.model][2],fmt(r.x),fmt(r.mean),fmt(r.lower),fmt(r.upper),r.relative===null?'undefined':fmt(r.relative*100)+'%'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});table.append(tr)}details.append(table);card.append(details);document.getElementById('plots').append(card);}
function update(){const T=Number(document.getElementById('temperature').value);if(!(T>0))return;document.getElementById('plots').replaceChildren();for(const view of DATA.views)draw(view,T);document.getElementById('density').textContent=`Rb-87 number density: ${fmt(pressure/(kb*T))} m⁻³`;}
document.getElementById('title').textContent=DATA.comparison?'Two-level vs physical multilevel MOT':'Physical multilevel MOT loading';
document.getElementById('intro').textContent=DATA.comparison?'Direct comparisons on the exact saved benchmark launch samples. Each model has its own pointwise 95% Student-t band; repump-only studies are in the multilevel report.':'Eight independent parameter sweeps and nine views, including cooling size with fixed or matched repump size and the three repump studies.';
document.getElementById('status').textContent=`${DATA.status.replaceAll('_',' ')} · ${DATA.completed_total}/${DATA.expected} physical configurations`;
for(const view of DATA.views){const a=document.createElement('a');a.href='#'+view.id;a.textContent=view.title.replace('Loading vs ','');document.getElementById('nav').append(a)}
document.getElementById('timestamp').textContent=`Report updated ${DATA.updated_utc}. Refresh the file for newly completed scan points. ${DATA.eta?`Estimated completion: ${new Date(DATA.eta.earliest_completion_utc).toLocaleString()} to ${new Date(DATA.eta.latest_completion_utc).toLocaleString()} (local browser time).`:``}`;
document.getElementById('temperature').addEventListener('input',update);update();
</script></html>'''
