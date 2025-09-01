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
})
function preventDefaults(e){ e.preventDefault(); e.stopPropagation(); }

['dragenter','dragover'].forEach(ev => {
  dropArea.addEventListener(ev, ()=> dropArea.classList.add('dragover'), false)
})
['dragleave','drop'].forEach(ev => {
  dropArea.addEventListener(ev, ()=> dropArea.classList.remove('dragover'), false)
})

dropArea.addEventListener('drop', handleDrop, false)
function handleDrop(e){
  const dt = e.dataTransfer
  const files = dt.files
  if(files && files.length>0) handleFiles(files)
}

fileElem.addEventListener('change', (e)=>{
  handleFiles(e.target.files)
})

function handleFiles(files){
  selectedFile = files[0]
  if(!selectedFile) return
  fileDetails.textContent = `${selectedFile.name} — ${Math.round(selectedFile.size/1024)} KB`
}

uploadForm.addEventListener('submit', async (e)=>{
  e.preventDefault()
  if(!selectedFile){ alert('Lütfen önce bir video seçin.'); return }
  uploadB