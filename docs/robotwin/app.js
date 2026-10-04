const scenes = {
  stack_blocks_two: { label: "stack_blocks_two", frames: 310, duration: "10.33 s" },
  hanging_mug: { label: "hanging_mug", frames: 337, duration: "11.23 s" },
};
const methods = {
  d0: { label: "D0 干净 GT", detail: "Clean GT metric depth" },
  d1: { label: "D1 传感器噪声", detail: "RealSense 模拟噪声" },
  d3: { label: "D3 LingBot 修复", detail: "D1 有效值保留 + 孔洞补全" },
};
const params = new URLSearchParams(location.search);
let scene = Object.hasOwn(scenes, params.get("scene")) ? params.get("scene") : "stack_blocks_two";
let method = Object.hasOwn(methods, params.get("method")) ? params.get("method") : "d0";
const video = document.getElementById("main-video");
const gallery = document.getElementById("gallery-grid");

function mediaPath(selectedScene, selectedMethod, extension) {
  return `media/${selectedScene}/episode0_${selectedMethod}.${extension}`;
}

function render() {
  const previousTime = video.currentTime || 0;
  const wasPlaying = !video.paused;
  const source = mediaPath(scene, method, "mp4");
  video.pause();
  video.poster = mediaPath(scene, method, "jpg");
  video.src = source;
  video.load();
  video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(previousTime, Math.max(0, video.duration - 0.05));
    if (wasPlaying) video.play().catch(() => {});
  }, { once: true });
  document.getElementById("video-title").textContent = `${scenes[scene].label} · ${methods[method].label}`;
  document.getElementById("video-meta").textContent = `${scenes[scene].frames} 帧 · ${scenes[scene].duration} · ${methods[method].detail}`;
  document.getElementById("video-download").href = source;
  document.getElementById("stack-results").hidden = scene !== "stack_blocks_two";
  document.getElementById("mug-results").hidden = scene !== "hanging_mug";
  document.querySelectorAll("[data-scene]").forEach(button => {
    const active = button.dataset.scene === scene;
    button.setAttribute("aria-pressed", active);
  });
  document.querySelectorAll("[data-method]").forEach(button => {
    const active = button.dataset.method === method;
    button.setAttribute("aria-pressed", active);
  });
  gallery.querySelectorAll("button").forEach(button => {
    button.setAttribute("aria-current", button.dataset.scene === scene && button.dataset.method === method ? "true" : "false");
  });
  const url = new URL(location.href);
  url.searchParams.set("scene", scene);
  url.searchParams.set("method", method);
  history.replaceState(null, "", url);
}

for (const [sceneKey, sceneValue] of Object.entries(scenes)) {
  for (const [methodKey, methodValue] of Object.entries(methods)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "gallery-item";
    button.dataset.scene = sceneKey;
    button.dataset.method = methodKey;
    const picture = document.createElement("img");
    picture.src = mediaPath(sceneKey, methodKey, "jpg");
    picture.alt = `${sceneValue.label} ${methodValue.label} 三视角 RGB-D 视频封面`;
    picture.loading = "lazy";
    const caption = document.createElement("span");
    caption.innerHTML = `<strong>${sceneValue.label}</strong><small>${methodValue.label}</small>`;
    button.append(picture, caption);
    button.addEventListener("click", () => {
      scene = sceneKey;
      method = methodKey;
      render();
      video.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    gallery.append(button);
  }
}
document.querySelectorAll("[data-scene]").forEach(button => button.addEventListener("click", () => {
  scene = button.dataset.scene;
  render();
}));
document.querySelectorAll("[data-method]").forEach(button => button.addEventListener("click", () => {
  method = button.dataset.method;
  render();
}));
render();
