const docStatus = document.getElementById("docStatus");
const reindexBtn = document.getElementById("reindexBtn");
const chatForm = document.getElementById("chatForm");
const messages = document.getElementById("messages");
const ragStatus = document.getElementById("ragStatus");
const ragToggleBtn = document.getElementById("ragToggleBtn");
const uploadForm = document.getElementById("uploadForm");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");

let currentDocuments = [];
let ragEnabled = true;

function appendMessage(role, title, body, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = title;

  const content = document.createElement("div");
  content.className = "message-body";
  content.textContent = body;

  article.append(label, content);

  if (sources.length) {
    const list = document.createElement("div");
    list.className = "source-list";
    sources.forEach((source) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const sourceTitle = document.createElement("strong");
      sourceTitle.textContent = source.source;
      const sourcePreview = document.createElement("div");
      sourcePreview.textContent = source.preview;
      item.append(sourceTitle, sourcePreview);
      list.appendChild(item);
    });
    article.appendChild(list);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function renderDocumentList(files) {
  currentDocuments = files || [];
  documentList.innerHTML = "";

  if (!currentDocuments.length) {
    const empty = document.createElement("div");
    empty.className = "document-empty";
    empty.textContent = "当前还没有文档。";
    documentList.appendChild(empty);
    return;
  }

  currentDocuments.forEach((file) => {
    const row = document.createElement("div");
    row.className = "document-row";

    const name = document.createElement("div");
    name.className = "document-name";
    name.textContent = file;

    const removeBtn = document.createElement("button");
    removeBtn.className = "danger-btn";
    removeBtn.type = "button";
    removeBtn.textContent = "删除";
    removeBtn.dataset.file = file;

    row.append(name, removeBtn);
    documentList.appendChild(row);
  });
}

function renderDocSummary(data) {
  const notes = data.notes?.length ? `\n提示:\n- ${data.notes.join("\n- ")}` : "";
  docStatus.textContent =
    `文件数: ${data.file_count}\n片段数: ${data.chunk_count}\n索引时间: ${data.indexed_at || "尚未建立"}\n文档: ${
      data.files.length ? data.files.join(", ") : "暂无"
    }${notes}`;
  renderDocumentList(data.files);
}

function renderRagStatus() {
  ragStatus.textContent = ragEnabled
    ? "当前状态：已开启 RAG。\n聊天时会先检索知识库，再把片段送给模型。"
    : "当前状态：已关闭 RAG。\n聊天时将直接调用模型，不附带知识库片段。";
  ragToggleBtn.textContent = ragEnabled ? "关闭 RAG" : "开启 RAG";
}

async function refreshDocuments() {
  docStatus.textContent = "正在读取索引状态...";
  try {
    const response = await fetch("/api/documents");
    const data = await response.json();
    renderDocSummary(data);
  } catch (error) {
    docStatus.textContent = `读取失败: ${error.message}`;
  }
}

async function refreshRagStatus() {
  ragStatus.textContent = "正在读取 RAG 状态...";
  try {
    const response = await fetch("/api/rag");
    const data = await response.json();
    ragEnabled = Boolean(data.rag_enabled);
    renderRagStatus();
  } catch (error) {
    ragStatus.textContent = `读取失败: ${error.message}`;
  }
}

async function rebuildIndex() {
  reindexBtn.disabled = true;
  docStatus.textContent = "正在重建索引...";
  try {
    const response = await fetch("/api/documents/reindex", { method: "POST" });
    const data = await response.json();
    renderDocSummary(data.summary);
  } catch (error) {
    docStatus.textContent = `重建失败: ${error.message}`;
  } finally {
    reindexBtn.disabled = false;
  }
}

async function toggleRag() {
  ragToggleBtn.disabled = true;
  try {
    const response = await fetch("/api/rag", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ rag_enabled: !ragEnabled }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "切换失败");
    }
    ragEnabled = Boolean(data.rag_enabled);
    renderRagStatus();
  } catch (error) {
    ragStatus.textContent = `切换失败: ${error.message}`;
  } finally {
    ragToggleBtn.disabled = false;
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const submitButton = uploadForm.querySelector('button[type="submit"]');
  const fileInput = document.getElementById("uploadFile");
  const file = fileInput.files?.[0];
  if (!file) {
    uploadStatus.textContent = "请选择一个文档。";
    return;
  }

  submitButton.disabled = true;
  uploadStatus.textContent = "正在上传并重建索引...";

  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "上传失败");
    }
    uploadStatus.textContent = data.message;
    renderDocSummary(data.summary);
    uploadForm.reset();
  } catch (error) {
    uploadStatus.textContent = `上传失败: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
}

async function removeDocument(file) {
  uploadStatus.textContent = `正在删除 ${file} 并重建索引...`;
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(file)}`, {
      method: "DELETE",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "删除失败");
    }
    uploadStatus.textContent = data.message;
    renderDocSummary(data.summary);
  } catch (error) {
    uploadStatus.textContent = `删除失败: ${error.message}`;
  }
}

reindexBtn.addEventListener("click", rebuildIndex);
ragToggleBtn.addEventListener("click", toggleRag);
uploadForm.addEventListener("submit", uploadDocument);

documentList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.dataset.file) {
    return;
  }
  await removeDocument(target.dataset.file);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const submitButton = chatForm.querySelector('button[type="submit"]');
  submitButton.disabled = true;

  const question = document.getElementById("question").value.trim();
  const topK = Number(document.getElementById("topK").value || "4");
  if (!question) {
    submitButton.disabled = false;
    return;
  }

  appendMessage("user", "用户", question);
  appendMessage("assistant", "系统", "正在调用模型，请稍候...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: topK,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "服务返回错误");
    }

    messages.removeChild(messages.lastElementChild);
    appendMessage("assistant", "回答", data.answer, data.sources || []);
  } catch (error) {
    messages.removeChild(messages.lastElementChild);
    appendMessage("assistant", "错误", `请求失败: ${error.message}`);
  } finally {
    submitButton.disabled = false;
  }
});

Promise.all([refreshDocuments(), refreshRagStatus()]);
