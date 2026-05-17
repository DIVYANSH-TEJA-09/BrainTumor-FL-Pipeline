/* ═══ NeuroAI Presentation — Interactions ═══ */

document.addEventListener('DOMContentLoaded',()=>{
  initScrollReveal();
  initNavDots();
  initNavScroll();
  initFLAnimation();
  initTumorGrowth();
  initSegLayers();
  initCountUp();
});

/* ─── Scroll Reveal ─── */
function initScrollReveal(){
  const els=document.querySelectorAll('.reveal,.reveal-left,.reveal-right,.stagger');
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');obs.unobserve(e.target)}});
  },{threshold:0.15,rootMargin:'0px 0px -60px 0px'});
  els.forEach(el=>obs.observe(el));
}

/* ─── Nav Dots ─── */
function initNavDots(){
  const sections=document.querySelectorAll('section[id]');
  const dots=document.querySelectorAll('.nav-dot');
  const counter=document.getElementById('navCounter');
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        const id=e.target.id;
        dots.forEach((d,i)=>{
          d.classList.toggle('active',d.dataset.target===id);
          if(d.dataset.target===id&&counter)counter.textContent=String(i+1).padStart(2,'0')+' / '+dots.length;
        });
      }
    });
  },{threshold:0.4});
  sections.forEach(s=>obs.observe(s));
  dots.forEach(d=>d.addEventListener('click',()=>{
    const t=document.getElementById(d.dataset.target);
    if(t)t.scrollIntoView({behavior:'smooth'});
  }));
}

/* ─── Nav Background ─── */
function initNavScroll(){
  const nav=document.querySelector('nav');
  window.addEventListener('scroll',()=>{nav.classList.toggle('scrolled',window.scrollY>60)},{passive:true});
}

/* ─── FL Training Animation ─── */
function initFLAnimation(){
  const container=document.getElementById('flViz');
  if(!container)return;
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach(e=>{if(e.isIntersecting){runFLAnimation(container);obs.unobserve(container)}});
  },{threshold:0.3});
  obs.observe(container);
}

function runFLAnimation(container){
  const clients=container.querySelectorAll('.fl-client');
  const server=container.querySelector('.fl-server');
  const statusEl=document.getElementById('flStatus');
  let round=0;
  const maxRounds=5;

  function animateRound(){
    if(round>=maxRounds)return;
    round++;
    if(statusEl)statusEl.textContent=`Round ${round}/${maxRounds} — Local training...`;
    
    // Pulse clients (training)
    clients.forEach(c=>{c.style.boxShadow='0 0 0 6px rgba(212,160,23,.3)';c.style.borderColor='var(--gold)'});
    
    setTimeout(()=>{
      if(statusEl)statusEl.textContent=`Round ${round}/${maxRounds} — Sending weights...`;
      // Create packets flying to server
      clients.forEach((c,i)=>{
        const pkt=document.createElement('div');
        pkt.className='fl-packet';
        pkt.style.cssText=`left:${c.offsetLeft+30}px;top:${c.offsetTop+30}px;transition:all 1s cubic-bezier(.4,0,.2,1)`;
        container.appendChild(pkt);
        setTimeout(()=>{
          pkt.style.left=(server.offsetLeft+40)+'px';
          pkt.style.top=(server.offsetTop+40)+'px';
        },50);
        setTimeout(()=>pkt.remove(),1200);
      });
    },800);

    setTimeout(()=>{
      if(statusEl)statusEl.textContent=`Round ${round}/${maxRounds} — QPSO aggregation...`;
      server.style.boxShadow='0 0 0 12px rgba(212,160,23,.3),0 0 0 24px rgba(212,160,23,.1)';
    },2000);

    setTimeout(()=>{
      if(statusEl)statusEl.textContent=`Round ${round}/${maxRounds} — Broadcasting global model`;
      server.style.boxShadow='0 0 0 8px rgba(212,160,23,.15),0 0 0 16px rgba(212,160,23,.06)';
      clients.forEach(c=>{c.style.boxShadow='var(--shadow)';c.style.borderColor='var(--border)'});
    },3000);

    setTimeout(animateRound,4000);
  }
  animateRound();
}

/* ─── Tumor Growth on Scroll ─── */
function initTumorGrowth(){
  const dot=document.getElementById('tumorDot');
  const alert=document.getElementById('tumorAlert');
  if(!dot)return;
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        setTimeout(()=>{dot.classList.add('grown')},600);
        setTimeout(()=>{if(alert)alert.classList.add('visible')},2200);
      }else{
        dot.classList.remove('grown');
        if(alert)alert.classList.remove('visible');
      }
    });
  },{threshold:0.5});
  const wrap=dot.closest('.tumor-viz');
  if(wrap)obs.observe(wrap);
}

/* ─── Segmentation Layers hover ─── */
function initSegLayers(){
  document.querySelectorAll('.seg-layer').forEach(l=>{
    l.addEventListener('mouseenter',()=>{l.style.transform='perspective(600px) rotateY(0deg) scale(1.08)'});
    l.addEventListener('mouseleave',()=>{l.style.transform='perspective(600px) rotateY(15deg) scale(1)'});
  });
}

/* ─── Count Up Animation ─── */
function initCountUp(){
  const nums=document.querySelectorAll('[data-count]');
  const obs=new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        const el=e.target;
        const target=parseFloat(el.dataset.count);
        const suffix=el.dataset.suffix||'';
        const prefix=el.dataset.prefix||'';
        const decimals=el.dataset.decimals?parseInt(el.dataset.decimals):0;
        animateCount(el,0,target,1200,prefix,suffix,decimals);
        obs.unobserve(el);
      }
    });
  },{threshold:0.5});
  nums.forEach(n=>obs.observe(n));
}

function animateCount(el,start,end,duration,prefix,suffix,decimals){
  const startTime=performance.now();
  function update(now){
    const elapsed=now-startTime;
    const progress=Math.min(elapsed/duration,1);
    const eased=1-Math.pow(1-progress,3);
    const current=start+(end-start)*eased;
    el.textContent=prefix+current.toFixed(decimals)+suffix;
    if(progress<1)requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}
