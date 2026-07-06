const $ = s => document.querySelector(s);
const chat = $("#chat");
let attachFiles = [];   // File[] (uploaded or grabbed-as-blob images)
let docIds = [];        // multiple documents
let accessGranted = false;   // request-based access to restricted images

// ---------- panels / tabs ----------
document.querySelectorAll(".nav-btn").forEach(b => b.onclick = () => {
  const p = b.dataset.panel;
  document.querySelectorAll(".nav-btn").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach(x =>
    x.classList.toggle("active", x.dataset.panel === p));
});

// ---------- status ----------
fetch("/api/status").then(r => r.json()).then(s => {
  $("#side-status").innerHTML =
    `${s.device} · ${s.bank.toLocaleString()} imgs · ${s.kb.toLocaleString()} KB<br>LLM: ${s.llm}`;
}).catch(() => $("#side-status").textContent = "offline");

// ---------- chat sessions (localStorage) ----------
const KEY = "mma_sessions";
let sessions = JSON.parse(localStorage.getItem(KEY) || "[]");
let current = null;

function persist() { localStorage.setItem(KEY, JSON.stringify(sessions)); renderHistory(); }
function renderHistory() {
  const h = $("#history"); h.innerHTML = "";
  sessions.forEach(s => {
    const row = document.createElement("div");
    row.className = "hist-row" + (current && s.id === current.id ? " active" : "");
    const b = document.createElement("button");
    b.className = "hist-item"; b.textContent = s.title || "New chat";
    b.onclick = () => loadChat(s.id);
    const del = document.createElement("button");
    del.className = "hist-del"; del.textContent = "🗑"; del.title = "Delete chat";
    del.onclick = e => { e.stopPropagation(); deleteSession(s.id); };
    row.appendChild(b); row.appendChild(del); h.appendChild(row);
  });
}
function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  localStorage.setItem(KEY, JSON.stringify(sessions));
  if (current && current.id === id) current = null;
  if (!sessions.length) newChat();
  else if (!current) loadChat(sessions[0].id);
  else renderHistory();
}
function titleize(msg) {
  let t = (msg || "").toLowerCase().trim();
  for (const f of ["can you", "could you", "please", "give me", "i want", "i need",
    "i meant", "show me", "tell me about", "research papers on", "research paper on",
    "papers on", "find papers on", "research on"]) {
    if (t.startsWith(f)) t = t.slice(f.length).trim();
  }
  t = t.replace(/[?.!]+$/, "").split(/\s+/).slice(0, 6).join(" ");
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : "New chat";
}
function newChat() {
  current = { id: Date.now().toString(36), title: "New chat", messages: [] };
  sessions.unshift(current); docIds = []; clearAttach();
  const ds = $("#doc-status"); if (ds) ds.hidden = true;
  chat.innerHTML = ""; $("#chat-title").textContent = "New chat";
  greet(); persist();
}
function loadChat(id) {
  current = sessions.find(s => s.id === id); if (!current) return;
  chat.innerHTML = ""; $("#chat-title").textContent = current.title;
  if (!current.messages.length) greet();
  current.messages.forEach(m => render(m.role, m.text, m.note, m.images, false, m.download, m.preview));
  renderHistory();
  document.querySelector('.nav-btn[data-panel="chat"]').click();
}
function greet() {
  render("bot", "**Welcome.** I can search real medical images, analyze an uploaded/"
    + "grabbed image, study a document (PDF), explain concepts, and find **research "
    + "papers** (PubMed/Scholar). Try “show brain MRI”, “research papers on TB”, or "
    + "“what is a glioma?”.", null, null, false);
}

// ---------- rendering ----------
function render(role, text, note, images, save = true) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const av = role === "user" ? "🧑‍⚕️" : "🩺";
  let imgs = "";
  if (images && images.length) {
    if (images.length === 1 && !images[0].source) {
      imgs = `<div class="solo"><img src="${images[0].url}"/></div>`;
    } else {
      imgs = `<div class="grid">` + images.map(im => `
        <div class="card" data-ref="${im.ref || ''}"><img src="${im.url}"/>
        <div class="cap"><b>${im.score ?? ''}</b> ${(im.caption || '').slice(0, 68)}
        ${im.source ? `<br>📖 <a href="${im.source_url}" target="_blank">${im.source}</a>` : ''}</div></div>`).join("") + `</div>`;
    }
  }
  let actions = "";
  if (arguments[4] && arguments[4].url) {   // download object passed as 5th arg
    const d = arguments[4];
    actions = `<div class="report-actions">
      <a class="dl" href="${d.url}" download="${d.name}">⬇ Download PDF</a>
      <button class="pv" data-md="${encodeURIComponent(arguments[5] || text)}" data-url="${d.url}">👁 Preview</button></div>`;
  }
  wrap.innerHTML = `<div class="avatar">${av}</div><div class="bubble">${marked.parse(text || "")}${imgs}${actions}${note ? `<div class="note">${note}</div>` : ""}</div>`;
  chat.appendChild(wrap);
  wrap.querySelectorAll(".card[data-ref]").forEach(c => {
    if (c.dataset.ref) c.onclick = () => grab(c.dataset.ref, c.querySelector("img").src);
  });
  const pv = wrap.querySelector(".pv");
  if (pv) pv.onclick = () => openPreview(decodeURIComponent(pv.dataset.md), pv.dataset.url);
  // access gate (arg 7 = access methods object)
  const access = arguments[6];
  if (access) renderAccessGate(wrap.querySelector(".bubble"), access);
  chat.scrollTop = chat.scrollHeight;
  if (save && current) {
    current.messages.push({ role, text, note, images, download: arguments[4], preview: arguments[5] });
    if (role === "user" && (current.title === "New chat" || !current.title))
      current.title = titleize(text);
    persist();
  }
  return wrap;
}

// ---------- attachments (multiple images) ----------
async function grab(ref, url) {            // fetch a result image as a File
  try {
    const b = await (await fetch(url)).blob();
    attachFiles.push(new File([b], (ref.split("/").pop() || "image.png"),
                              { type: b.type || "image/png" }));
    renderChips();
  } catch (e) {}
}
function renderChips() {
  const c = $("#attach-chip");
  if (!attachFiles.length) { c.hidden = true; c.innerHTML = ""; return; }
  c.hidden = false;
  c.innerHTML = attachFiles.map((f, i) =>
    `<span class="chip-item"><img src="${URL.createObjectURL(f)}"/>${f.name.slice(0, 12)}
     <button data-i="${i}">×</button></span>`).join("");
  c.querySelectorAll("button").forEach(b =>
    b.onclick = () => { attachFiles.splice(+b.dataset.i, 1); renderChips(); });
}
function clearAttach() { attachFiles = []; $("#file-img").value = ""; renderChips(); }
$("#btn-attach").onclick = () => $("#file-img").click();
$("#file-img").onchange = e => { for (const f of e.target.files) attachFiles.push(f); renderChips(); };

// ---------- document upload (multiple) ----------
$("#btn-doc").onclick = () => $("#file-doc").click();
$("#file-doc").onchange = async e => {
  const ds = $("#doc-status");
  for (const f of e.target.files) {
    ds.hidden = false; ds.innerHTML = `<span class="spinner"></span> ingesting “${f.name}”…`;
    const fd = new FormData(); fd.append("file", f);
    try {
      const r = await (await fetch("/api/upload_doc", { method: "POST", body: fd })).json();
      if (r.error) throw new Error(r.error);
      docIds.push(r.doc_id);
    } catch (err) { ds.innerHTML = `⚠️ ${err.message}`; return; }
  }
  renderDocs();
};
function renderDocs() {
  const ds = $("#doc-status");
  if (!docIds.length) { ds.hidden = true; return; }
  ds.hidden = false;
  ds.innerHTML = `📄 <b>${docIds.length}</b> document(s) ready — ask “generate a report” or a question. <a href="#" id="clrd">clear</a>`;
  $("#clrd").onclick = ev => { ev.preventDefault(); docIds = []; ds.hidden = true; };
}

// ---------- restricted-access request gate (two methods + proof) ----------
function renderAccessGate(bubble, access) {
  if (!bubble) return;
  const subj = encodeURIComponent("MultiMedAI restricted-image access request");
  const body = encodeURIComponent(
    "Reason for access:\nRole (medical student / doctor):\nInstitution:\n" +
    "Registration/Enrolment ID:\n(attach proof of credentials)\n");
  const mail = `mailto:${access.email}?subject=${subj}&body=${body}`;
  const form_link = access.form_url
    ? `<a class="mini" href="${access.form_url}" target="_blank">🌐 External form</a>` : "";
  const div = document.createElement("div");
  div.className = "access";
  div.innerHTML = `
    <div class="access-req">Requires: ${access.required}</div>
    <div class="access-methods">
      <a class="mini" href="${mail}">✉️ Method 1 — Email us</a>
      ${form_link}
      <button class="mini afbtn">📝 Method 2 — Verify in-app</button>
    </div>
    <div class="access-form" hidden>
      <input class="txt af-reason" placeholder="Clinical reason for access"/>
      <select class="txt af-role"><option value="">Role…</option>
        <option>Medical student</option><option>Doctor / physician</option>
        <option>Resident</option><option>Nurse</option><option>Researcher</option></select>
      <input class="txt af-proof" placeholder="Institution + registration/enrolment ID"/>
      <input type="file" class="af-file"/>
      <button class="primary af-submit">Submit request</button>
      <div class="muted af-status"></div>
    </div>`;
  bubble.appendChild(div);
  const form = div.querySelector(".access-form");
  div.querySelector(".afbtn").onclick = () => { form.hidden = !form.hidden; };
  div.querySelector(".af-submit").onclick = async () => {
    const fd = new FormData();
    fd.append("reason", div.querySelector(".af-reason").value);
    fd.append("role", div.querySelector(".af-role").value);
    fd.append("proof", div.querySelector(".af-proof").value);
    const f = div.querySelector(".af-file").files[0];
    if (f) fd.append("credential", f);
    const st = div.querySelector(".af-status");
    st.innerHTML = `<span class="spinner"></span> submitting…`;
    try {
      const r = await (await fetch("/api/request_access", { method: "POST", body: fd })).json();
      st.textContent = r.message;
      if (r.granted) { accessGranted = true; form.hidden = true;
        st.textContent += " Re-run your search to view the images."; }
    } catch (e) { st.textContent = "⚠️ " + e.message; }
  };
}

// ---------- report preview drawer ----------
function openPreview(md, url) {
  $("#preview-body").innerHTML = marked.parse(md || "");
  $("#preview-dl").href = url || "#";
  $("#preview").classList.add("open");
}
$("#preview-close").onclick = () => $("#preview").classList.remove("open");

// ---------- send ----------
const input = $("#input");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 150) + "px"; });
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("#btn-send").onclick = send;

async function send() {
  const msg = input.value.trim(); if (!msg && !attachFiles.length) return;
  const hist = (current?.messages || []).slice(-8).map(m => ({ role: m.role, text: m.text }));
  const uimg = attachFiles.length ? attachFiles.map(f => ({ url: URL.createObjectURL(f) })) : null;
  render("user", msg || "(image)", null, uimg);
  input.value = ""; input.style.height = "auto";
  const fd = new FormData();
  fd.append("message", msg);
  fd.append("allow_gen", $("#allow-gen").checked);
  fd.append("scholar", $("#use-scholar")?.checked || false);
  fd.append("doc_ids", docIds.join(","));
  fd.append("history", JSON.stringify(hist));
  fd.append("access_granted", accessGranted);
  for (const f of attachFiles) fd.append("images", f);
  const think = render("bot", "", null, null, false);
  think.querySelector(".bubble").innerHTML = `<span class="spinner"></span> thinking…`;
  try {
    const r = await (await fetch("/api/chat", { method: "POST", body: fd })).json();
    think.remove();
    let note = r.note || "";
    if (r.corrected && r.corrected.trim() && r.corrected.trim() !== msg.trim())
      note = `✏️ Interpreted as: “${r.corrected}”. ` + note;
    render("bot", r.text || "(no response)", note, r.images, r.download, r.preview, r.access);
  } catch (err) { think.remove(); render("bot", "⚠️ " + err.message); }
  // attachments are CONSUMED by the message: clear the image chips and collapse the
  // document standby bar so nothing lingers "on standby" above the composer. Image
  // content is preserved in the conversation history (the server reuses it for
  // content follow-ups like "report on the images above"); uploaded documents stay
  // active in memory (docIds) for follow-up questions, just without the standby chip.
  clearAttach();
  const ds = $("#doc-status"); if (ds) ds.hidden = true;
}

// ---------- settings: gemini key ----------
$("#gem-save").onclick = async () => {
  const key = $("#gem-key").value.trim(); if (!key) return;
  const st = $("#gem-status"); st.innerHTML = `<span class="spinner"></span> verifying…`;
  const fd = new FormData(); fd.append("key", key);
  const r = await (await fetch("/api/set_gemini_key", { method: "POST", body: fd })).json();
  st.textContent = r.ok ? "✅ Saved — responses now use Gemini (fast)." : "❌ " + (r.message || "invalid key");
  if (r.ok) fetch("/api/status").then(x => x.json()).then(s => $("#side-status").innerHTML =
    `${s.device} · ${s.bank.toLocaleString()} imgs · ${s.kb.toLocaleString()} KB<br>LLM: ${s.llm}`);
};

// ---------- feedback ----------
$("#fb-send").onclick = async () => {
  const t = $("#fb-text").value.trim(); if (!t) return;
  const fd = new FormData(); fd.append("text", t);
  await fetch("/api/feedback", { method: "POST", body: fd });
  $("#fb-text").value = ""; $("#fb-status").textContent = "✅ Thanks — feedback recorded.";
};

// ---------- boot ----------
$("#new-chat").onclick = newChat;
if (sessions.length) loadChat(sessions[0].id); else newChat();
