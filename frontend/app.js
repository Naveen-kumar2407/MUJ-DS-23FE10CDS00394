const API_BASE = "http://localhost:8000/api";

// Elements
const uploadZone = document.getElementById("uploadZone");
const videoUpload = document.getElementById("videoUpload");
const uploadProgress = document.getElementById("uploadProgress");
const progressBar = document.querySelector(".progress-bar");
const uploadStatus = document.getElementById("uploadStatus");

const geminiKeyInput = document.getElementById("geminiKey");
const enableRAGToggle = document.getElementById("enableRAG");

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");

const loader = document.getElementById("loader");
const loaderText = document.getElementById("loaderText");
const resultsGrid = document.getElementById("resultsGrid");
const emptyState = document.getElementById("emptyState");
const ragBox = document.getElementById("ragBox");
const ragContent = document.getElementById("ragContent");

const videoCountEl = document.getElementById("videoCount");
const segmentCountEl = document.getElementById("segmentCount");

// Format seconds to MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
}

// Initial stats load
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/videos`);
        const data = await res.json();
        videoCountEl.innerText = data.indexed_videos.length;
        segmentCountEl.innerText = data.total_segments || 0;
    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}
fetchStats();

// API Key logic
geminiKeyInput.addEventListener("input", (e) => {
    enableRAGToggle.disabled = e.target.value.trim().length === 0;
    if (enableRAGToggle.disabled) {
        enableRAGToggle.checked = false;
    }
});

// Upload logic
uploadZone.addEventListener("click", () => videoUpload.click());
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--primary)";
});
uploadZone.addEventListener("dragleave", () => {
    uploadZone.style.borderColor = "var(--border-color)";
});
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--border-color)";
    if (e.dataTransfer.files.length) {
        handleUpload(e.dataTransfer.files[0]);
    }
});
videoUpload.addEventListener("change", (e) => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
});

async function handleUpload(file) {
    if (!file.name.match(/\.(mp4|mov|avi)$/i)) {
        alert("Please upload a valid video file (.mp4, .mov, .avi)");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", file);
    
    uploadProgress.classList.remove("hidden");
    progressBar.style.width = "50%";
    uploadStatus.innerText = "Extracting frames and audio embeddings...";
    
    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        
        if (res.ok) {
            progressBar.style.width = "100%";
            uploadStatus.innerText = data.message;
            setTimeout(() => {
                uploadProgress.classList.add("hidden");
                progressBar.style.width = "0%";
                uploadStatus.innerText = "";
                fetchStats();
            }, 3000);
        } else {
            throw new Error(data.detail);
        }
    } catch (e) {
        progressBar.style.width = "0%";
        uploadStatus.innerText = `Error: ${e.message}`;
        uploadStatus.style.color = "red";
    }
}

// Search Logic
async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    
    // UI Reset
    emptyState.classList.add("hidden");
    ragBox.classList.add("hidden");
    resultsGrid.innerHTML = "";
    loader.classList.remove("hidden");
    loaderText.innerText = "Searching multi-modal space...";
    
    try {
        const res = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}`);
        const data = await res.json();
        
        loader.classList.add("hidden");
        
        if (!data.results || data.results.length === 0) {
            resultsGrid.innerHTML = `<div class="empty-state"><p>No results found for "${query}"</p></div>`;
            return;
        }
        
        renderResults(data.results);
        
        // Trigger RAG if enabled
        if (enableRAGToggle.checked && geminiKeyInput.value) {
            triggerRAG(query, data.results);
        }
        
    } catch (e) {
        loader.classList.add("hidden");
        alert(`Search failed: ${e.message}`);
    }
}

function renderResults(results) {
    resultsGrid.innerHTML = results.map(r => `
        <div class="result-card">
            <div class="result-header">
                <div class="result-title">#${r.rank} — ${r.video_name}</div>
                <div class="badges">
                    <span class="badge time">⏱ ${formatTime(r.timestamp_start)}</span>
                    <span class="badge attn">Attn: ${r.attn_score.toFixed(3)}</span>
                </div>
            </div>
            ${r.filename ? `
                <video class="video-player" controls preload="metadata">
                    <source src="${API_BASE}/videos/${encodeURIComponent(r.filename)}#t=${r.timestamp_start}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            ` : `<div style="padding: 2rem; text-align:center; background: rgba(0,0,0,0.2); border-radius: 8px;">Source video file not found on disk.</div>`}
        </div>
    `).join("");
}

async function triggerRAG(query, hits) {
    ragBox.classList.remove("hidden");
    ragContent.innerHTML = `<span class="cursor"></span>`;
    
    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                gemini_key: geminiKeyInput.value,
                hits: hits
            })
        });
        
        if (!response.ok) {
            const err = await response.json();
            ragContent.innerHTML = `<span style="color: #ef4444">Error: ${err.detail}</span>`;
            return;
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        ragContent.innerHTML = "";
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Parse Server-Sent Events (SSE)
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || "";
            
            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const text = line.substring(6);
                    if (text === "[DONE]") {
                        return; // Finished
                    } else if (text.startsWith("[ERROR]")) {
                        ragContent.innerHTML += `<span style="color: #ef4444">${text}</span>`;
                        return;
                    } else {
                        // Very simple markdown to HTML (just for line breaks)
                        const formattedText = text.replace(/\n/g, "<br>");
                        ragContent.innerHTML += formattedText;
                    }
                }
            }
        }
    } catch (e) {
        ragContent.innerHTML = `<span style="color: #ef4444">Connection failed: ${e.message}</span>`;
    }
}

searchBtn.addEventListener("click", performSearch);
searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") performSearch();
});
