/**
 * 高考志愿填报系统 · Web版 前端逻辑
 * 对接后端 Flask API，实现与桌面端等效功能
 */

const API = "";  // 同源，路径前缀为空

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 全局状态
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const State = {
  user: null,           // { username, user_type }
  regions: [],          // 所有省份列表
  majorsList: [],       // 所有专业列表
  selectedRegions: null,// null=未选, []=全选, [...]=部分
  selectedMajors:  null,
  searchResults: [],    // 当前搜索结果
  schemes: {},          // { tabId: { name, items:[] } }
  activeTab: null,
  tabCounter: 0,
  deleteHistory: [],    // 最近5条删除记录 { tabId, index, item }
  advConditions: [],    // 高级筛选条件
  sortSettings: [],     // 排序设置
  dragItem: null,       // 当前拖拽中的数据
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 工具函数
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function setStatus(msg) {
  document.getElementById("statusBar").textContent = msg;
}

function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.style.cssText = `
    position:fixed; bottom:30px; left:50%; transform:translateX(-50%);
    background:${type === "error" ? "#e74c3c" : "#2c3e50"}; color:#fff;
    padding:10px 24px; border-radius:8px; font-size:.9rem; z-index:99999;
    box-shadow:0 4px 16px rgba(0,0,0,.3); pointer-events:none;
  `;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2800);
}

async function apiFetch(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `HTTP ${res.status}`);
  }
  return res.json();
}

function getUserRank() {
  const v = document.getElementById("rankInput").value;
  return v && !isNaN(v) && parseInt(v) > 0 ? parseInt(v) : null;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 登录
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function doLogin() {
  const username = document.getElementById("loginUser").value.trim();
  const password = document.getElementById("loginPass").value.trim();
  const errEl = document.getElementById("loginError");
  errEl.textContent = "";
  if (!username || !password) { errEl.textContent = "请输入用户名和密码"; return; }
  try {
    const data = await apiFetch("/api/login", "POST", { username, password });
    State.user = data;
    document.getElementById("loginOverlay").classList.add("hidden");
    document.getElementById("mainApp").classList.remove("hidden");
    initApp();
  } catch (e) {
    errEl.textContent = "用户名或密码错误";
  }
}

document.getElementById("loginBtn").addEventListener("click", doLogin);
document.getElementById("loginPass").addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });
document.getElementById("loginUser").addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 应用初始化
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function initApp() {
  // 更新用户信息
  const u = State.user;
  document.getElementById("usernameEl").textContent = u.username;
  const badgeMap = { guest: "游客", vip: "VIP会员", dev: "开发者" };
  document.getElementById("userBadge").textContent = badgeMap[u.user_type] || u.user_type;
  const avatarEl = document.getElementById("avatarEl");
  avatarEl.className = "avatar" + (u.user_type === "vip" ? " avatar-vip" : u.user_type === "dev" ? " avatar-dev" : "");

  // 拉取基础数据
  document.getElementById("resultList").innerHTML = '<div class="loading-spinner">加载数据中…</div>';
  try {
    const [regions, majors] = await Promise.all([
      apiFetch("/api/regions"),
      apiFetch("/api/majors_list"),
    ]);
    State.regions = regions;
    State.majorsList = majors;
  } catch (e) {
    showToast("数据加载失败: " + e.message, "error");
  }

  // 创建初始方案
  addTab("方案1");

  // 绑定事件
  bindEvents();

  // 默认加载全部数据
  await doSearch();

  setStatus(`欢迎，${u.username}（${badgeMap[u.user_type]}）| 拖动左侧条目到右侧方案`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 事件绑定
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function bindEvents() {
  document.getElementById("searchBtn").addEventListener("click", doSearch);
  document.getElementById("rankInput").addEventListener("change", refreshTagsInSchemes);
  document.getElementById("regionBtn").addEventListener("click", () => showMultiSelect("地区", State.regions, State.selectedRegions, (sel) => {
    State.selectedRegions = sel;
    document.getElementById("regionLabel").textContent = formatSelection(sel, "地区");
  }));
  document.getElementById("majorBtn").addEventListener("click", () => showMultiSelect("专业", State.majorsList, State.selectedMajors, (sel) => {
    State.selectedMajors = sel;
    document.getElementById("majorLabel").textContent = formatSelection(sel, "专业");
  }));
  document.getElementById("advFilterBtn").addEventListener("click", showFilterModal);
  document.getElementById("sortBtn").addEventListener("click", showSortModal);
  document.getElementById("addTabBtn").addEventListener("click", () => addTab());
  document.getElementById("undoBtn").addEventListener("click", undoDelete);
  document.getElementById("exportBtn").addEventListener("click", exportCSV);
  document.getElementById("aiBtn").addEventListener("click", showAIAnalysis);
  document.getElementById("switchUserBtn").addEventListener("click", () => {
    document.getElementById("loginOverlay").classList.remove("hidden");
    document.getElementById("loginUser").value = "";
    document.getElementById("loginPass").value = "";
    document.getElementById("loginError").textContent = "";
  });

  // 模态关闭
  document.getElementById("modalClose").addEventListener("click", closeModal);
  document.getElementById("modal").addEventListener("click", e => { if (e.target.id === "modal") closeModal(); });
  document.getElementById("aiModalClose").addEventListener("click", () => document.getElementById("aiModal").classList.add("hidden"));
  document.getElementById("aiCloseBtn").addEventListener("click", () => document.getElementById("aiModal").classList.add("hidden"));
  document.getElementById("aiCopyBtn").addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("aiResult").innerText);
    showToast("已复制到剪贴板");
  });
  document.getElementById("filterModalClose").addEventListener("click", () => document.getElementById("filterModal").classList.add("hidden"));
  document.getElementById("cancelFilterBtn").addEventListener("click", () => document.getElementById("filterModal").classList.add("hidden"));
  document.getElementById("applyFilterBtn").addEventListener("click", applyAdvFilter);
  document.getElementById("addConditionBtn").addEventListener("click", addFilterConditionRow);
  document.getElementById("clearConditionsBtn").addEventListener("click", () => {
    document.getElementById("filterConditions").innerHTML = "";
    addFilterConditionRow();
  });
  document.getElementById("sortModalClose").addEventListener("click", () => document.getElementById("sortModal").classList.add("hidden"));
  document.getElementById("cancelSortBtn").addEventListener("click", () => document.getElementById("sortModal").classList.add("hidden"));
  document.getElementById("applySortBtn").addEventListener("click", applySort);
  document.getElementById("addSortBtn").addEventListener("click", addSortConditionRow);
}

function formatSelection(sel, label) {
  if (sel === null) return "未选择";
  if (!sel || sel.length === 0) return "全部";
  if (sel.length <= 3) return sel.join("、");
  return sel.slice(0, 3).join("、") + ` 等${sel.length}个${label}`;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 搜索
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function doSearch() {
  if (State.selectedRegions === null) {
    document.getElementById("regionLabel").textContent = "未选择";
  }
  const body = {
    user_rank: getUserRank(),
    regions: State.selectedRegions || [],
    majors: State.selectedMajors || [],
    conditions: State.advConditions,
    sort: State.sortSettings,
  };
  document.getElementById("resultList").innerHTML = '<div class="loading-spinner">搜索中…</div>';
  try {
    const data = await apiFetch("/api/search", "POST", body);
    State.searchResults = data.data;
    renderResultList();
    document.getElementById("resultCount").textContent = `共 ${data.total} 条结果`;
    setStatus(`搜索完成：找到 ${data.total} 条，已添加到方案的会高亮显示`);
  } catch (e) {
    showToast("搜索失败: " + e.message, "error");
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 渲染结果列表
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderResultList() {
  const container = document.getElementById("resultList");
  if (!State.searchResults.length) {
    container.innerHTML = '<div class="loading-spinner" style="padding:40px">无搜索结果</div>';
    return;
  }

  const addedIds = getAddedIdsInActiveTab();
  container.innerHTML = "";

  State.searchResults.forEach(item => {
    const el = document.createElement("div");
    const tagAccent = item.tag ? ` ri-${getTagClass(item.tag)}` : '';
    el.className = "result-item" + (addedIds.has(item.id) ? " already-added" : "") + tagAccent;
    el.draggable = true;
    el.dataset.id = item.id;

    const tags = buildTagChips(item);
    const cspTag = item.tag ? `<span class="tag-chip ${getTagClass(item.tag)}">${item.tag}${item.prob ? ' ' + item.prob + '%' : ''}</span>` : "";

    el.innerHTML = `
      <div class="ri-title">${item.region} · ${item.major}</div>
      <div class="ri-meta">最低:${item.min_rank} &nbsp;|&nbsp; 中位:${item.median_rank} &nbsp;|&nbsp; 最高:${item.max_rank}</div>
      <div class="ri-tags">${tags}${cspTag}</div>
    `;

    el.addEventListener("dragstart", e => {
      State.dragItem = item;
      el.classList.add("dragging");
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData("text/plain", JSON.stringify(item));
    });
    el.addEventListener("dragend", () => el.classList.remove("dragging"));

    // 双击添加
    el.addEventListener("dblclick", () => addItemToActiveScheme(item));

    container.appendChild(el);
  });
}

function buildTagChips(item) {
  let html = "";
  if (item.SF985) html += `<span class="tag-chip tag-985">985</span>`;
  if (item.SF211) html += `<span class="tag-chip tag-211">211</span>`;
  if (item.SFSYL) html += `<span class="tag-chip tag-syl">双一流</span>`;
  return html;
}

function getTagClass(tag) {
  if (tag === "冲") return "tag-chong";
  if (tag === "保") return "tag-bao";
  return "tag-wen";
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 方案标签页管理
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function addTab(name) {
  State.tabCounter++;
  const id = "tab_" + State.tabCounter;
  if (!name) {
    const existing = Object.values(State.schemes).map(s => s.name);
    let n = 1;
    while (existing.includes("方案" + n)) n++;
    name = "方案" + n;
  }
  State.schemes[id] = { name, items: [] };
  State.activeTab = id;
  renderTabBar();
  renderActiveScheme();
}

function switchTab(id) {
  State.activeTab = id;
  renderTabBar();
  renderActiveScheme();
  renderResultList();  // 刷新左侧已添加标记
}

function closeTab(id, e) {
  e.stopPropagation();
  const names = Object.keys(State.schemes);
  if (names.length === 1) { showToast("至少保留一个方案"); return; }
  if (!confirm(`确定要删除方案「${State.schemes[id].name}」吗？`)) return;
  delete State.schemes[id];
  if (State.activeTab === id) {
    State.activeTab = Object.keys(State.schemes)[0];
  }
  renderTabBar();
  renderActiveScheme();
}

function renderTabBar() {
  const bar = document.getElementById("tabBar");
  bar.innerHTML = "";
  Object.entries(State.schemes).forEach(([id, scheme]) => {
    const tab = document.createElement("div");
    tab.className = "tab-item" + (id === State.activeTab ? " active" : "");
    tab.innerHTML = `
      <span class="tab-name" title="${scheme.name}">${scheme.name}</span>
      <button class="tab-close" title="关闭">✕</button>
    `;
    tab.querySelector(".tab-name").addEventListener("click", () => switchTab(id));
    tab.querySelector(".tab-close").addEventListener("click", (e) => closeTab(id, e));
    // 双击重命名
    tab.querySelector(".tab-name").addEventListener("dblclick", () => {
      const newName = prompt("请输入新方案名称：", scheme.name);
      if (newName && newName.trim()) {
        State.schemes[id].name = newName.trim();
        renderTabBar();
      }
    });
    bar.appendChild(tab);
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 渲染当前方案
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderActiveScheme() {
  const container = document.getElementById("schemeContainer");
  const scheme = State.schemes[State.activeTab];

  // 拖放事件
  container.ondragover = e => { e.preventDefault(); container.classList.add("drag-over"); };
  container.ondragleave = () => container.classList.remove("drag-over");
  container.ondrop = e => {
    e.preventDefault();
    container.classList.remove("drag-over");
    try {
      const data = JSON.parse(e.dataTransfer.getData("text/plain"));
      addItemToActiveScheme(data);
    } catch {}
  };

  if (!scheme.items.length) {
    container.innerHTML = `<div class="scheme-empty"><span class="arrow">⬅️</span>从左侧搜索结果中拖拽专业到这里<br><small>或双击左侧条目添加</small></div>`;
    updateStats();
    return;
  }

  container.innerHTML = "";
  const rank = getUserRank();

  scheme.items.forEach((item, idx) => {
    const tag = item._tag || (rank ? computeTag(rank, item) : "稳");
    const prob = item._prob || (rank ? computeProb(rank, item) : null);

    const el = document.createElement("div");
    el.className = "scheme-item" + (tag ? ` si-${getTagClass(tag)}` : '');
    el.innerHTML = `
      <div class="si-number">${idx + 1}</div>
      <div class="si-content">
        <div class="si-title">${item.region} · ${item.major}</div>
        <div class="si-meta">最低:${item.min_rank} &nbsp;|&nbsp; 中位:${item.median_rank} &nbsp;|&nbsp; 最高:${item.max_rank}</div>
        <div class="si-tags">${buildTagChips(item)}</div>
      </div>
      <div class="si-right">
        <select class="tag-select ${getTagClass(tag)}" data-idx="${idx}">
          <option value="冲"${tag === "冲" ? " selected" : ""}>冲${prob && tag === "冲" ? " " + prob + "%" : ""}</option>
          <option value="稳"${tag === "稳" ? " selected" : ""}>稳${prob && tag === "稳" ? " " + prob + "%" : ""}</option>
          <option value="保"${tag === "保" ? " selected" : ""}>保${prob && tag === "保" ? " " + prob + "%" : ""}</option>
        </select>
        <button class="si-delete" data-idx="${idx}">✕</button>
      </div>
    `;

    const sel = el.querySelector(".tag-select");
    sel.addEventListener("change", () => {
      const i = parseInt(sel.dataset.idx);
      State.schemes[State.activeTab].items[i]._tag = sel.value;
      sel.className = `tag-select ${getTagClass(sel.value)}`;
      updateStats();
    });

    el.querySelector(".si-delete").addEventListener("click", () => {
      const i = parseInt(el.querySelector(".si-delete").dataset.idx);
      deleteFromScheme(i);
    });

    container.appendChild(el);
  });

  updateStats();
}

// 不可达阈值：与后端保持一致，位次差 > 100% 则不可达
const UNREACHABLE_THRESHOLD = 1.0;
// 单个方案最大志愿数
const MAX_SCHEME_ITEMS = 20;

function computeTag(rank, item) {
  const minRank = item.min_rank || 0;
  if (!minRank) return "稳";
  const diff = (rank - minRank) / minRank;
  if (diff > UNREACHABLE_THRESHOLD) return null;  // 不可达
  if (diff >= 0) return "冲";
  if (diff >= -0.15) return "稳";
  return "保";
}

function computeProb(rank, item) {
  const minRank = item.min_rank || 0;
  if (!minRank) return 50;
  const diff = (rank - minRank) / minRank;
  if (diff > UNREACHABLE_THRESHOLD) return null;  // 不可达
  if (diff >= 0.30) return 20;
  if (diff >= 0.15) return 30;
  if (diff >= 0) return 40;
  if (diff >= -0.10) return 60;
  if (diff >= -0.15) return 75;
  if (diff >= -0.30) return 85;
  return 95;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 方案条目操作
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function addItemToActiveScheme(item) {
  const scheme = State.schemes[State.activeTab];
  if (scheme.items.find(i => i.id === item.id)) {
    showToast(`「${item.major}」已在方案中`);
    return;
  }
  if (scheme.items.length >= MAX_SCHEME_ITEMS) {
    showToast(`每个方案最多添加 ${MAX_SCHEME_ITEMS} 个志愿，请新建方案或删除已有志愿`, "error");
    return;
  }
  const rank = getUserRank();
  const clone = { ...item };
  if (rank) {
    const tag = computeTag(rank, clone);
    if (tag === null) {
      showToast(`该专业与您的排名差距过大，不在可选范围内`, "error");
      return;
    }
    clone._tag = clone.tag || tag;
    clone._prob = clone.prob || computeProb(rank, clone);
  }
  scheme.items.push(clone);
  renderActiveScheme();
  renderResultList();
  updateStats();
  setStatus(`已添加：${item.region} ${item.major}（共 ${scheme.items.length}/${MAX_SCHEME_ITEMS} 个）`);
}

function deleteFromScheme(idx) {
  const scheme = State.schemes[State.activeTab];
  const removed = scheme.items.splice(idx, 1)[0];
  if (State.deleteHistory.length >= 5) State.deleteHistory.shift();
  State.deleteHistory.push({ tabId: State.activeTab, index: idx, item: removed });
  renderActiveScheme();
  renderResultList();
  setStatus(`已删除：${removed.major}`);
}

function undoDelete() {
  if (!State.deleteHistory.length) { showToast("没有可撤回的操作"); return; }
  const { tabId, index, item } = State.deleteHistory.pop();
  if (!State.schemes[tabId]) { showToast("原方案已关闭，无法撤回", "error"); return; }
  const items = State.schemes[tabId].items;
  items.splice(Math.min(index, items.length), 0, item);
  if (State.activeTab !== tabId) switchTab(tabId);
  else { renderActiveScheme(); renderResultList(); }
  setStatus(`已撤回：${item.major}`);
}

function getAddedIdsInActiveTab() {
  const scheme = State.schemes[State.activeTab];
  return scheme ? new Set(scheme.items.map(i => i.id)) : new Set();
}

function refreshTagsInSchemes() {
  const rank = getUserRank();
  Object.values(State.schemes).forEach(scheme => {
    scheme.items.forEach(item => {
      if (rank) {
        item._tag = computeTag(rank, item);
        item._prob = computeProb(rank, item);
      } else {
        item._tag = null;
        item._prob = null;
      }
    });
  });
  renderActiveScheme();
  renderResultList();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 统计更新
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function updateStats() {
  const scheme = State.schemes[State.activeTab];
  if (!scheme) return;
  const items = scheme.items;
  let chong = 0, wen = 0, bao = 0;
  items.forEach(item => {
    const t = item._tag || "稳";
    if (t === "冲") chong++;
    else if (t === "保") bao++;
    else wen++;
  });
  document.getElementById("countChong").textContent = chong;
  document.getElementById("countWen").textContent = wen;
  document.getElementById("countBao").textContent = bao;
  // 显示 X/20 进度
  const totalEl = document.getElementById("countTotal");
  totalEl.textContent = `${items.length}/${MAX_SCHEME_ITEMS}`;
  totalEl.style.color = items.length >= MAX_SCHEME_ITEMS ? "#e74c3c" : "";
  // 更新冲稳保进度条
  const total = items.length;
  const barC = document.getElementById("barChong");
  const barW = document.getElementById("barWen");
  const barB = document.getElementById("barBao");
  if (barC && barW && barB) {
    barC.style.width = total ? `${(chong / total) * 100}%` : '0%';
    barW.style.width = total ? `${(wen   / total) * 100}%` : '0%';
    barB.style.width = total ? `${(bao   / total) * 100}%` : '0%';
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 多选对话框（地区 / 专业）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function showMultiSelect(label, allItems, selected, callback) {
  const titleEl = document.getElementById("modalTitle");
  const bodyEl  = document.getElementById("modalBody");
  const footerEl = document.getElementById("modalFooter");

  titleEl.textContent = `选择${label}（可多选）`;

  // 当前选中状态（复制）
  let cur = selected === null ? [] : (selected.length === 0 ? allItems.slice() : selected.slice());

  bodyEl.innerHTML = `
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <input class="ms-search" id="msSearch" placeholder="搜索…">
      <button class="btn btn-sm btn-ghost" id="msSelectAll">全选</button>
      <button class="btn btn-sm btn-ghost" id="msClearAll">清除</button>
    </div>
    <div class="multi-select-wrap" id="msWrap"></div>
    <div style="color:#7f8c8d;font-size:.8rem;margin-top:8px">已选择 <b id="msCnt">0</b> 项</div>
  `;

  function renderItems(filter) {
    const wrap = document.getElementById("msWrap");
    wrap.innerHTML = "";
    const filtered = allItems.filter(it => !filter || it.toLowerCase().includes(filter.toLowerCase()));
    filtered.forEach(it => {
      const row = document.createElement("label");
      row.className = "ms-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = it;
      cb.checked = cur.includes(it);
      cb.addEventListener("change", () => {
        if (cb.checked) { if (!cur.includes(it)) cur.push(it); }
        else cur = cur.filter(x => x !== it);
        updateCnt();
      });
      row.appendChild(cb);
      row.append(" " + it);
      wrap.appendChild(row);
    });
    updateCnt();
  }

  function updateCnt() {
    const el = document.getElementById("msCnt");
    if (el) el.textContent = cur.length;
  }

  renderItems("");

  document.getElementById("msSearch").addEventListener("input", e => renderItems(e.target.value));
  document.getElementById("msSelectAll").addEventListener("click", () => {
    cur = allItems.slice();
    renderItems(document.getElementById("msSearch").value);
  });
  document.getElementById("msClearAll").addEventListener("click", () => {
    cur = [];
    renderItems(document.getElementById("msSearch").value);
  });

  footerEl.innerHTML = "";
  const okBtn = document.createElement("button");
  okBtn.className = "btn btn-sm btn-primary";
  okBtn.textContent = "✓ 确定";
  okBtn.addEventListener("click", () => {
    // cur.length === allItems.length → 视为全选 → []
    const result = cur.length === allItems.length ? [] : cur;
    closeModal();
    callback(result);
  });
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn btn-sm";
  cancelBtn.textContent = "取消";
  cancelBtn.addEventListener("click", closeModal);
  footerEl.appendChild(okBtn);
  footerEl.appendChild(cancelBtn);

  document.getElementById("modal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 高级筛选
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function showFilterModal() {
  const container = document.getElementById("filterConditions");
  container.innerHTML = "";
  if (State.advConditions.length) {
    State.advConditions.forEach(c => addFilterConditionRow(c));
  } else {
    addFilterConditionRow();
  }
  document.getElementById("filterModal").classList.remove("hidden");
}

function addFilterConditionRow(cond = null) {
  const container = document.getElementById("filterConditions");
  const row = document.createElement("div");
  row.className = "filter-condition-row";

  const logicSel = `<select class="fc-logic">
    <option value="AND"${cond && cond.logic === "AND" ? " selected" : ""}>AND</option>
    <option value="OR"${cond && cond.logic === "OR" ? " selected" : ""}>OR</option>
  </select>`;
  const fieldSel = `<select class="fc-field">
    ${["最低排名","中位数","地区","专业","标签"].map(f => `<option${cond && cond.field === f ? " selected" : ""}>${f}</option>`).join("")}
  </select>`;
  const opSel = `<select class="fc-op">
    ${["=","≠","<",">","≤","≥","包含","不包含"].map(o => `<option${cond && cond.operator === o ? " selected" : ""}>${o}</option>`).join("")}
  </select>`;
  const valIn = `<input type="text" class="fc-val" placeholder="条件值" value="${cond ? cond.value || "" : ""}">`;

  row.innerHTML = `
    <span class="fc-label">逻辑</span>${logicSel}
    <span class="fc-label">字段</span>${fieldSel}
    <span class="fc-label">操作</span>${opSel}
    <span class="fc-label">值</span>${valIn}
    <button class="btn btn-sm btn-red" style="padding:4px 8px">删除</button>
  `;
  row.querySelector("button").addEventListener("click", () => row.remove());
  container.appendChild(row);
}

function applyAdvFilter() {
  const rows = document.querySelectorAll(".filter-condition-row");
  State.advConditions = Array.from(rows).map(row => ({
    logic:    row.querySelector(".fc-logic").value,
    field:    row.querySelector(".fc-field").value,
    operator: row.querySelector(".fc-op").value,
    value:    row.querySelector(".fc-val").value.trim(),
  })).filter(c => c.value);
  document.getElementById("filterModal").classList.add("hidden");
  doSearch();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 自定义排序
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const SORT_FIELDS = [
  { label: "最低排名", value: "min_rank" },
  { label: "最高排名", value: "max_rank" },
  { label: "中位数排名", value: "median_rank" },
  { label: "地区", value: "region" },
  { label: "专业", value: "major" },
];

function showSortModal() {
  const container = document.getElementById("sortConditions");
  container.innerHTML = "";
  if (State.sortSettings.length) {
    State.sortSettings.forEach(s => addSortConditionRow(s));
  } else {
    addSortConditionRow();
  }
  document.getElementById("sortModal").classList.remove("hidden");
}

function addSortConditionRow(cond = null) {
  const container = document.getElementById("sortConditions");
  const row = document.createElement("div");
  row.className = "filter-condition-row";

  const fieldSel = `<select class="sc-field">
    ${SORT_FIELDS.map(f => `<option value="${f.value}"${cond && cond.field === f.value ? " selected" : ""}>${f.label}</option>`).join("")}
  </select>`;
  const dirSel = `<select class="sc-dir">
    <option value="asc"${cond && cond.ascending !== false ? " selected" : ""}>升序 ↑</option>
    <option value="desc"${cond && cond.ascending === false ? " selected" : ""}>降序 ↓</option>
  </select>`;

  row.innerHTML = `
    <span class="fc-label">排序字段</span>${fieldSel}
    <span class="fc-label">方向</span>${dirSel}
    <button class="btn btn-sm btn-red" style="padding:4px 8px">删除</button>
  `;
  row.querySelector("button").addEventListener("click", () => row.remove());
  container.appendChild(row);
}

function applySort() {
  const rows = document.querySelectorAll(".sc-field");
  const dirs  = document.querySelectorAll(".sc-dir");
  State.sortSettings = Array.from(rows).map((f, i) => ({
    field: f.value,
    ascending: dirs[i].value === "asc",
  }));
  document.getElementById("sortModal").classList.add("hidden");
  doSearch();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 导出 CSV
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function exportCSV() {
  if (State.user.user_type === "guest") {
    showToast("游客用户无导出权限，请升级为 VIP", "error"); return;
  }
  const scheme = State.schemes[State.activeTab];
  if (!scheme.items.length) { showToast("当前方案为空，无需导出"); return; }

  try {
    const data = await apiFetch("/api/export", "POST", {
      scheme_name: scheme.name,
      user_rank: getUserRank(),
      data: scheme.items,
    });
    const blob = new Blob(["\uFEFF" + data.csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = scheme.name + ".csv";
    a.click();
    URL.revokeObjectURL(url);
    showToast("导出成功！");
  } catch (e) {
    showToast("导出失败: " + e.message, "error");
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Markdown 渲染（AI 输出格式化）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function applyInline(text) {
  return text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function renderMarkdown(raw) {
  if (!raw) return '';
  const esc = raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const lines = esc.split('\n');
  const out = [];
  let inList = false;
  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<div style="margin:5px 0"></div>');
    } else if (t.startsWith('## ')) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<h3 class="ai-section-title">${applyInline(t.slice(3))}</h3>`);
    } else if (t.startsWith('### ')) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<h4 class="ai-sub-title">${applyInline(t.slice(4))}</h4>`);
    } else if (/^[-•*] /.test(t)) {
      if (!inList) { out.push('<ul class="ai-list">'); inList = true; }
      out.push(`<li>${applyInline(t.slice(2))}</li>`);
    } else if (t.startsWith('⚠️')) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<div class="ai-risk">${applyInline(t)}</div>`);
    } else if (t.startsWith('✅')) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<div class="ai-suggest">${applyInline(t)}</div>`);
    } else if (/综合评分[：:]\s*[\d.]+\/10/.test(t)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<div class="ai-score">${applyInline(t)}</div>`);
    } else {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push(`<p class="ai-para">${applyInline(t)}</p>`);
    }
  }
  if (inList) out.push('</ul>');
  return out.join('\n');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AI 分析
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function showAIAnalysis() {
  if (State.user.user_type === "guest") {
    showToast("游客用户无AI分析权限，请升级为 VIP", "error"); return;
  }
  const scheme = State.schemes[State.activeTab];
  if (!scheme.items.length) { showToast("当前方案为空，请先添加志愿"); return; }

  document.getElementById("aiResult").innerHTML = '<div class="loading-spinner" style="padding:60px">正在调用 AI 分析，请稍候…</div>';
  document.getElementById("aiModal").classList.remove("hidden");

  try {
    const data = await apiFetch("/api/ai_analyze", "POST", { scheme_data: scheme.items });
    document.getElementById("aiResult").innerHTML = renderMarkdown(data.result);
  } catch (e) {
    document.getElementById("aiResult").innerHTML = `<div class="ai-risk">⚠️ AI分析出错：${e.message}</div>`;
  }
}
