const state = {
  user: null,
  cursor: 0,
  loading: false,
  hasMore: true,
  feedCount: 0,
  items: new Map(),
  sessionByItem: new Map(),
  clickedItemIds: new Set(),
  clickingItemIds: new Set(),
  feedGeneration: 0,
};

const $ = (selector, root = document) => root.querySelector(selector);
const loginView = $("#login-view");
const appView = $("#app-view");
const feedGrid = $("#feed-grid");
const toast = $("#toast");

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({ message: "服务返回了不可解析的数据" }));
  if (!response.ok || payload.code !== 0) {
    const error = new Error(payload.message || `请求失败 (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload.data;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function hueFor(value) {
  let hash = 0;
  for (const char of String(value)) hash = (hash * 31 + char.charCodeAt(0)) % 360;
  return hash;
}

function codeFor(item) {
  const words = (item.name || item.item_id).replace(/[^\u4e00-\u9fa5A-Za-z0-9 ]/g, "").split(/\s+/).filter(Boolean);
  return words.slice(0, 2).map((word) => word[0]).join("").toUpperCase() || String(item.item_id).slice(-2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSkeletons(count = 8) {
  const template = $("#skeleton-template");
  for (let i = 0; i < count; i += 1) feedGrid.append(template.content.cloneNode(true));
}

function productCard(item, index, rank) {
  const article = document.createElement("article");
  article.className = "product-card";
  article.dataset.itemId = item.item_id;
  article.style.setProperty("--index", index % 12);
  article.style.setProperty("--hue", hueFor(item.item_id));
  article.style.setProperty("--tilt", `${(hueFor(item.name) % 20) - 10}deg`);
  article.innerHTML = `
    <button class="product-main" type="button" aria-label="查看 ${escapeHtml(item.name)}">
      <div class="product-visual" data-code="${escapeHtml(codeFor(item))}">
        <span class="rank-tag">NO.${String(rank).padStart(2, "0")}</span>
        <span class="score-tag">${Number(item.score || 0).toFixed(3)}</span>
      </div>
      <div class="card-body">
        <p class="category">${escapeHtml(item.category)}</p>
        <h3>${escapeHtml(item.name)}</h3>
        <p class="reason">${escapeHtml(item.reason || "为你精选")}</p>
      </div>
    </button>
    <div class="card-actions">
      <button type="button" data-feedback="like">♡ 喜欢</button>
      <button type="button" data-feedback="dislike">⊘ 不感兴趣</button>
    </div>`;
  $(".product-main", article).addEventListener("click", () => handleProductClick(item, article));
  article.querySelectorAll("[data-feedback]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(item, button.dataset.feedback, article, button));
  });
  return article;
}

async function loadRecommendations({ reset = false, replace = false } = {}) {
  if (state.loading || (!state.hasMore && !reset && !replace)) return;
  state.loading = true;
  const loadButton = $("#load-more-button");
  loadButton.disabled = true;
  loadButton.textContent = "正在整理你的清单…";

  if (reset || replace) {
    if (reset || !state.hasMore) state.cursor = 0;
    state.hasMore = true;
    state.items.clear();
    state.sessionByItem.clear();
    state.clickedItemIds.clear();
    state.clickingItemIds.clear();
    state.feedGeneration += 1;
    state.feedCount = state.cursor;
    feedGrid.innerHTML = "";
    renderSkeletons();
  }

  try {
    const data = await api(`/api/recommendations?cursor=${state.cursor}&limit=12`);
    feedGrid.querySelectorAll(".skeleton").forEach((node) => node.remove());
    const fragment = document.createDocumentFragment();
    data.items.forEach((item, index) => {
      if (state.items.has(item.item_id)) return;
      state.items.set(item.item_id, item);
      state.sessionByItem.set(item.item_id, data.session_id);
      state.feedCount += 1;
      fragment.append(productCard(item, index, state.feedCount));
    });
    feedGrid.append(fragment);
    state.cursor = data.next_cursor ?? state.cursor;
    state.hasMore = data.has_more;
    loadButton.textContent = state.hasMore ? "继续向下发现" : "这次先看到这里";
    loadButton.disabled = !state.hasMore;
  } catch (error) {
    feedGrid.querySelectorAll(".skeleton").forEach((node) => node.remove());
    if (error.status === 401) return showLogin();
    showToast(error.message);
    loadButton.textContent = "重新加载";
    loadButton.disabled = false;
  } finally {
    state.loading = false;
  }
}

function renderRail(anchor, anchorItem, items, generation) {
  if (generation !== state.feedGeneration || !anchor.isConnected) return;
  const seen = new Set();
  const freshItems = items.filter((item) => {
    const itemId = String(item.item_id);
    if (seen.has(itemId) || state.clickedItemIds.has(itemId) || state.items.has(itemId)) return false;
    seen.add(itemId);
    return true;
  });
  const existingRail = [...feedGrid.querySelectorAll(".similar-rail")]
    .find((node) => node.dataset.anchorItemId === String(anchorItem.item_id));
  if (!freshItems.length) {
    existingRail?.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  const rail = document.createElement("section");
  rail.className = "similar-rail";
  rail.dataset.anchorItemId = String(anchorItem.item_id);
  rail.innerHTML = `
    <header><p class="eyebrow">BECAUSE YOU VIEWED</p><h3>因为你看过《${escapeHtml(anchorItem.name)}》</h3><p>为你找到了 ${freshItems.length} 件来自相似关系的好物。</p></header>
    <div class="rail-items"></div>`;
  const list = $(".rail-items", rail);
  freshItems.forEach((item) => {
    state.items.set(item.item_id, item);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rail-item";
    button.innerHTML = `<i class="similar-badge">相似</i><b>${escapeHtml(item.name)}</b><span>${escapeHtml(item.category)} · ${Number(item.score).toFixed(3)}</span>`;
    button.addEventListener("click", () => openDetailWithSimilar(item));
    list.append(button);
  });
  if (existingRail) existingRail.replaceWith(rail);
  else anchor.insertAdjacentElement("afterend", rail);
  rail.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function handleProductClick(item, article) {
  const itemId = String(item.item_id);
  if (state.clickingItemIds.has(itemId)) return;
  const generation = state.feedGeneration;
  state.clickedItemIds.add(itemId);
  state.clickingItemIds.add(itemId);
  article.classList.add("is-opening");
  try {
    const data = await api("/api/events/click", {
      method: "POST",
      body: JSON.stringify({
        item_id: item.item_id,
        session_id: state.sessionByItem.get(item.item_id),
        visible_item_ids: [...new Set([...state.items.keys(), ...state.clickedItemIds])],
        client_time: new Date().toISOString(),
        idempotency_key: crypto.randomUUID(),
      }),
    });
    if (generation !== state.feedGeneration) return;
    renderRail(article, item, data.items, generation);
    openDetail(item, data.items);
  } catch (error) {
    showToast(error.message);
  } finally {
    state.clickingItemIds.delete(itemId);
    article.classList.remove("is-opening");
  }
}

async function openDetailWithSimilar(item) {
  if (!item) return;
  openDetail(item, null);
  try {
    const data = await api(`/api/items/${encodeURIComponent(item.item_id)}/similar?limit=6`);
    openDetail(item, data.items);
  } catch (error) {
    openDetail(item, []);
    showToast(error.message);
  }
}

function openDetail(item, similar = []) {
  const dialog = $("#detail-dialog");
  const hue = hueFor(item.item_id);
  const similarMarkup = similar === null
    ? '<p class="detail-loading">正在沿着商品关系寻找相似好物…</p>'
    : similar.map((row) => `<button type="button" data-similar-id="${escapeHtml(row.item_id)}"><b>${escapeHtml(row.name)}</b><br><small>${escapeHtml(row.category)}</small></button>`).join("") || "<p>暂无更多相似商品</p>";
  $("#detail-content").innerHTML = `
    <section class="detail-hero">
      <div class="detail-visual" style="--hue:${hue}"><span>${escapeHtml(codeFor(item))}</span></div>
      <div class="detail-copy">
        <p class="eyebrow">ITEM / ${escapeHtml(item.item_id)}</p>
        <h2>${escapeHtml(item.name)}</h2>
        <p>${escapeHtml(item.description || "由你的偏好轨迹筛选出的个性化商品。")}</p>
        <div class="detail-meta">
          <div><span>类目</span><b>${escapeHtml(item.category)}</b></div>
          <div><span>推荐分</span><b>${Number(item.score || 0).toFixed(4)}</b></div>
          <div><span>推荐来源</span><b>${escapeHtml(item.source_type || "ItemCF")}</b></div>
          <div><span>商品编号</span><b>${escapeHtml(item.item_id)}</b></div>
        </div>
      </div>
    </section>
    <section class="detail-similar">
      <p class="eyebrow">SIMILAR FINDS</p><h3>相似商品推荐</h3>
      <div class="detail-similar-grid">${similarMarkup}</div>
    </section>`;
  $("#detail-content").querySelectorAll("[data-similar-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextItem = state.items.get(button.dataset.similarId)
        || similar.find((row) => row.item_id === button.dataset.similarId);
      openDetailWithSimilar(nextItem);
    });
  });
  if (!dialog.open) dialog.showModal();
}

async function sendFeedback(item, feedbackType, article, button) {
  try {
    await api("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        item_id: item.item_id,
        session_id: state.sessionByItem.get(item.item_id),
        feedback_type: feedbackType,
        client_time: new Date().toISOString(),
        idempotency_key: crypto.randomUUID(),
      }),
    });
    button.classList.add("is-active");
    if (feedbackType === "dislike") {
      article.animate([{ opacity: 1 }, { opacity: 0, transform: "scale(.96)" }], { duration: 260, fill: "forwards" });
      window.setTimeout(() => article.remove(), 270);
      showToast("已减少此类推荐");
    } else {
      button.textContent = "♥ 已喜欢";
      showToast("已记录你的喜欢");
    }
  } catch (error) {
    showToast(error.message);
  }
}

function showApp(user) {
  state.user = user;
  loginView.classList.add("is-hidden");
  appView.classList.remove("is-hidden");
  $("#user-name").textContent = user.display_name;
  $("#user-id-label").textContent = `USER ${String(user.user_id).replace(".0", "")}`;
  loadRecommendations({ reset: true });
}

function showLogin() {
  state.user = null;
  state.items.clear();
  state.clickedItemIds.clear();
  state.clickingItemIds.clear();
  state.feedGeneration += 1;
  feedGrid.innerHTML = "";
  appView.classList.add("is-hidden");
  loginView.classList.remove("is-hidden");
}

async function loadDemoAccounts() {
  try {
    const accounts = await api("/api/auth/demo-accounts");
    const list = $("#account-list");
    list.innerHTML = "";
    accounts.forEach((account, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "account-chip";
      button.innerHTML = `<b>${escapeHtml(account.display_name)}</b><span>@${escapeHtml(account.username)}</span>`;
      button.addEventListener("click", () => {
        list.querySelectorAll(".account-chip").forEach((node) => node.classList.remove("selected"));
        button.classList.add("selected");
        $("#username").value = account.username;
        $("#password").value = account.password;
      });
      list.append(button);
      if (index === 0) button.click();
    });
  } catch (error) {
    $("#login-error").textContent = error.message;
  }
}

async function loadSystemInfo() {
  try {
    const info = await api("/api/system");
    $("#online-score").textContent = Number(info.online_score).toFixed(4);
    $("#ndcg-score").textContent = Number(info.ndcg_at_10).toFixed(4);
    $("#recall-score").textContent = Number(info.recall_at_10).toFixed(4);
    $("#model-name").textContent = `${info.model_name || "Oblivionis"} / ${info.implementation || info.model_version}`;
  } catch (error) {
    console.warn(error);
  }
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = event.currentTarget.querySelector("button[type=submit]");
  const errorNode = $("#login-error");
  errorNode.textContent = "";
  submit.disabled = true;
  try {
    const user = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username: $("#username").value, password: $("#password").value }),
    });
    showApp(user);
  } catch (error) {
    errorNode.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

$("#logout-button").addEventListener("click", async () => {
  await api("/api/auth/logout", { method: "POST", body: "{}" }).catch(() => null);
  showLogin();
});
$("#refresh-button").addEventListener("click", () => loadRecommendations({ replace: true }));
$("#load-more-button").addEventListener("click", () => loadRecommendations());
$("#dialog-close").addEventListener("click", () => $("#detail-dialog").close());
$("#detail-dialog").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) event.currentTarget.close();
});

const loadMoreObserver = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting && state.user && state.hasMore) loadRecommendations();
}, { rootMargin: "180px" });
loadMoreObserver.observe($("#feed-status"));

async function bootstrap() {
  await loadDemoAccounts();
  try {
    const status = await api("/api/auth/status");
    if (status.authenticated) showApp(status.user);
    else showLogin();
  } catch (_) {
    showLogin();
  }
  loadSystemInfo();
}

bootstrap();
