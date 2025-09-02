const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const fileDetails = document.getElementById('file-details');
const uploadForm = document.getElementById('upload-form');
const uploadBtn = document.getElementById('uploadBtn');
const resultSection = document.getElementById('result');
const resultCard = document.getElementById('result-card');
let selectedFile = null;

['dragenter','dragover','dragleave','drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
});
function preventDefaults(e){ e.preventDefault(); e.stopPropagation(); }

['dragenter','dragover'].forEach(ev => {
  dropArea.addEventListener(ev, ()=> dropArea.classList.add('dragover'), false)
});
['dragleave','drop'].forEach(ev => {
  dropArea.addEventListener(ev, ()=> dropArea.classList.remove('dragover'), false)
});

dropArea.addEventListener('drop', handleDrop, false);
function handleDrop(e){
  const dt = e.dataTransfer;
  const files = dt.files;
  if(files && files.length>0) handleFiles(files);
}

fileElem.addEventListener('change', (e)=>{
  handleFiles(e.target.files);
});

function handleFiles(files){
  selectedFile = files[0];
  if(!selectedFile) return;
  fileDetails.textContent = `${selectedFile.name} — ${Math.round(selectedFile.size/1024)} KB`;
}

uploadForm.addEventListener('submit', async (e)=>{
  e.preventDefault();

  if(!selectedFile){ 
    alert('Lütfen önce bir video seçin.'); 
    return; 
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Analiz ediliyor...';
  resultSection.hidden = true;
  resultCard.innerHTML = '';

  const fd = new FormData();
  fd.append('video', selectedFile);

  try{
    const res = await fetch('/Detect', { method: 'POST', body: fd });
    const data = await res.json();
    if(res.ok){
      showResult(data);
    } else {
      showError(data.error || 'Sunucudan hata alındı');
    }
  } catch(err){
    showError(err.message);
  } finally{
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Analiz Et';
  }
});

function showResult(data){
  resultSection.hidden = false;
  if(data.error){
    resultCard.innerHTML = `<div class="card"><p class="small">Hata: ${data.error}</p></div>`;
    return;
  }
  const label = data.result;
  const conf = data.confidence;
  const badgeClass = label === 'FAKE' ? 'fake' : 'real';
  resultCard.innerHTML = `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3 style="margin:0">${label}</h3>
        <div class="badge ${badgeClass}">${conf}%</div>
      </div>
      <p class="small">Ortalama güven.</p>
      <hr style="opacity:0.06;margin:12px 0">
      <details>
        <summary class="small">Kare bazlı sonuçları göster</summary>
        <pre style="white-space:pre-wrap;margin-top:8px">${JSON.stringify(data.per_frame, null, 2)}</pre>
      </details>
    </div>
  `;
}

function showError(msg){
  resultSection.hidden = false;
  resultCard.innerHTML = `<div class="card"><p class="small">Hata: ${msg}</p></div>`;
}
