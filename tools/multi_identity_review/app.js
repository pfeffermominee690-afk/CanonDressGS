"use strict";

const candidateDimensions = [
  "identity_preservation", "garment_semantics", "full_body", "hands", "feet",
  "anatomy", "background", "blur", "view", "cross_view_consistency",
  "body_conforming_bias", "garment_volume"
];
const groupDimensions = [
  "color_consistency", "material_consistency", "sleeve_length_consistency",
  "hem_consistency", "front_back_semantic_consistency", "left_right_consistency",
  "identity_consistency", "accessory_consistency"
];
const slotOrder = ["front", "front_left_three_quarter", "left", "back_left_three_quarter", "back", "back_right_three_quarter", "right", "front_right_three_quarter"];
const state = { candidates: [], groups: [], index: 0 };
const byId = (id) => document.getElementById(id);

function mode() { return byId("reviewMode").value; }
function dimensions() { return mode() === "group" ? groupDimensions : candidateDimensions; }

function mediaUrl(path) {
  return path ? `/media?path=${encodeURIComponent(path)}` : "";
}

function draftKey(record) {
  return `multi_identity_review:${byId("reviewer").value}:${record.candidate_id || record.group_id}`;
}

function renderScores() {
  byId("scoreGrid").innerHTML = dimensions().map((name) => `
    <label><span>${name.replaceAll("_", " ")}</span>
      <select name="score_${name}" required>
        <option value="">Select</option><option>PASS</option><option>FAIL</option>
        <option>UNCERTAIN</option><option>NOT_APPLICABLE</option>
      </select>
    </label>`).join("");
}

function current() {
  return (mode() === "group" ? state.groups : state.candidates)[state.index];
}

function render() {
  const candidate = current();
  const records = mode() === "group" ? state.groups : state.candidates;
  byId("position").textContent = `${candidate ? state.index + 1 : 0} / ${records.length}`;
  document.querySelector(".evidence-grid").hidden = mode() === "group";
  byId("groupEvidence").hidden = mode() !== "group";
  renderScores();
  if (!candidate) return;
  if (mode() === "candidate") {
    const context = candidate.adjudication_context;
    const suffix = context ? ` / A:${context.REVIEWER_A} B:${context.REVIEWER_B}` : "";
    byId("recordMeta").textContent = `${candidate.identity_id} / ${candidate.garment_id} / ${candidate.semantic_pose_slot} / candidate ${candidate.candidate_index}${suffix}`;
    byId("candidateImage").src = mediaUrl(candidate.image_path);
    byId("maskImage").src = mediaUrl(candidate.mask_path);
    byId("identityImage").src = mediaUrl(candidate.identity_reference_path);
    byId("crossViewImage").src = mediaUrl(candidate.cross_view_reference_path);
  } else {
    const context = candidate.adjudication_context || candidate.members[0]?.group_adjudication_context;
    const suffix = context ? ` / A:${context.REVIEWER_A} B:${context.REVIEWER_B}` : "";
    byId("recordMeta").textContent = `${candidate.identity_id} / ${candidate.garment_id} / 8-slot group${suffix}`;
    byId("groupEvidence").innerHTML = candidate.members.map((member) => `
      <figure><figcaption>${member.semantic_pose_slot}</figcaption>
      <img src="${mediaUrl(member.image_path)}" alt="${member.semantic_pose_slot} candidate"></figure>`).join("");
  }
  byId("reviewForm").reset();
  const draft = JSON.parse(localStorage.getItem(draftKey(candidate)) || "null");
  if (draft) restoreDraft(draft);
}

function collectDraft() {
  const form = new FormData(byId("reviewForm"));
  const scores = Object.fromEntries(dimensions().map((name) => [name, form.get(`score_${name}`)]));
  return { decision: form.get("decision"), scores, comment: byId("comment").value };
}

function restoreDraft(draft) {
  if (draft.decision) {
    const radio = document.querySelector(`input[name=decision][value=${draft.decision}]`);
    if (radio) radio.checked = true;
  }
  dimensions().forEach((name) => {
    const select = document.querySelector(`[name=score_${name}]`);
    if (select) select.value = draft.scores?.[name] || "";
  });
  byId("comment").value = draft.comment || "";
}

async function loadQueue() {
  const reviewer = byId("reviewer").value;
  const response = await fetch(`/api/candidates?reviewer_id=${encodeURIComponent(reviewer)}&review_mode=${encodeURIComponent(mode())}`);
  if (!response.ok) throw new Error(`Queue load failed: ${response.status}`);
  const payload = await response.json();
  state.candidates = payload.candidates;
  const grouped = new Map();
  state.candidates.forEach((candidate) => {
    const groupId = `${candidate.identity_id}__${candidate.garment_id}`;
    if (!grouped.has(groupId)) grouped.set(groupId, { group_id: groupId, identity_id: candidate.identity_id, garment_id: candidate.garment_id, members: [] });
    grouped.get(groupId).members.push(candidate);
  });
  state.groups = [...grouped.values()]
    .map((group) => ({ ...group, members: group.members.sort((a, b) => slotOrder.indexOf(a.semantic_pose_slot) - slotOrder.indexOf(b.semantic_pose_slot)) }))
    .filter((group) => group.members.length === 8 && new Set(group.members.map((member) => member.semantic_pose_slot)).size === 8);
  state.index = 0;
  render();
  byId("status").textContent = payload.decision_visibility;
}

async function submitReview(event) {
  event.preventDefault();
  const candidate = current();
  if (!candidate) return;
  const draft = collectDraft();
  const reviewer = byId("reviewer").value;
  const payload = {
    [mode() === "group" ? "group_id" : "candidate_id"]: candidate.group_id || candidate.candidate_id,
    reviewer_id: reviewer,
    review_stage: reviewer === "ADJUDICATOR" ? "ADJUDICATION" : "INITIAL",
    decision: draft.decision,
    scores: draft.scores,
    comment: draft.comment,
    timestamp: new Date().toISOString()
  };
  const response = await fetch(mode() === "group" ? "/api/group-reviews" : "/api/reviews", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(`Append failed: ${response.status}`);
  localStorage.removeItem(draftKey(candidate));
  byId("status").textContent = "Decision appended";
  const recordCount = mode() === "group" ? state.groups.length : state.candidates.length;
  if (state.index < recordCount - 1) { state.index += 1; render(); }
}

renderScores();
byId("load").addEventListener("click", () => loadQueue().catch((error) => byId("status").textContent = error.message));
byId("previous").addEventListener("click", () => { if (state.index > 0) { state.index -= 1; render(); } });
byId("next").addEventListener("click", () => {
  const recordCount = mode() === "group" ? state.groups.length : state.candidates.length;
  if (state.index + 1 < recordCount) { state.index += 1; render(); }
});
byId("saveDraft").addEventListener("click", () => {
  if (current()) { localStorage.setItem(draftKey(current()), JSON.stringify(collectDraft())); byId("status").textContent = "Draft saved locally"; }
});
byId("reviewForm").addEventListener("submit", (event) => submitReview(event).catch((error) => byId("status").textContent = error.message));
byId("reviewMode").addEventListener("change", () => {
  state.index = 0;
  loadQueue().catch((error) => byId("status").textContent = error.message);
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("textarea, select, input")) return;
  if (event.key === "ArrowLeft") byId("previous").click();
  if (event.key === "ArrowRight") byId("next").click();
  const choice = { "1": "ACCEPT", "2": "MAYBE", "3": "REJECT" }[event.key];
  if (choice) document.querySelector(`input[name=decision][value=${choice}]`).checked = true;
});
