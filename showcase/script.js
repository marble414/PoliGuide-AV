const routeData = {
  online: {
    title: "OpenMMLab human pose -> SkeletonTCN phase2",
    desc: "在线部署入口，支持实时摄像头、离线视频、诊断模式和 ONNX 导出。",
    points: [
      "accuracy = 0.8074",
      "macro F1 = 0.8091",
      "未知长度摄像头流已支持",
    ],
  },
  fair: {
    title: "CTPGesture v2 -> frame segmenter multitask ensemble",
    desc: "公平 split benchmark 路线，使用 VIBE / wholebody / rich / track 特征和多成员集成。",
    points: [
      "gesture macro Jaccard = 0.849077",
      "foreground macro Jaccard = 0.847576",
      "command macro Jaccard = 0.843301",
    ],
  },
  upper: {
    title: "VIBE 225-dim -> all-data train=evaluate",
    desc: "用于判断模型容量上限，数值已超过论文公开结果，但不能作为公平 split 结论。",
    points: [
      "gesture macro Jaccard = 0.890252",
      "foreground macro Jaccard = 0.889186",
      "command macro Jaccard = 0.901227",
    ],
  },
  wholebody: {
    title: "OpenMMLab wholebody -> online segmenter route",
    desc: "在线可部署的 wholebody 兼容路线，部署口径与 benchmark 冠军口径分开呈现。",
    points: [
      "gesture macro Jaccard = 0.673483",
      "foreground macro Jaccard = 0.667961",
      "system FPS estimate = 37.02",
    ],
  },
};

const commandData = {
  GO_STRAIGHT: {
    label: "直行",
    intent: "ALLOW_PASS",
    autowareSpeed: 8.0,
    carlaSpeed: 6.0,
    note: "交由高层规划器生成直行轨迹",
  },
  STOP: {
    label: "停止",
    intent: "STOP",
    autowareSpeed: 0.0,
    carlaSpeed: 0.0,
    note: "停车等待，安全优先级最高",
  },
  LEFT_TURN_WAIT: {
    label: "左转待转",
    intent: "KEEP_WAIT",
    autowareSpeed: 0.0,
    carlaSpeed: 0.0,
    note: "保持等待，避免误启动",
  },
  SLOW_DOWN: {
    label: "减速慢行",
    intent: "SLOW_DOWN",
    autowareSpeed: 3.0,
    carlaSpeed: 3.0,
    note: "降速通过，保留制动冗余",
  },
  TURN_LEFT: {
    label: "左转",
    intent: "TURN_LEFT",
    autowareSpeed: 8.0,
    carlaSpeed: 6.0,
    note: "输出方向语义，由规划器生成轨迹",
  },
  TURN_RIGHT: {
    label: "右转",
    intent: "TURN_RIGHT",
    autowareSpeed: 8.0,
    carlaSpeed: 6.0,
    note: "输出方向语义，由规划器生成轨迹",
  },
  CHANGE_LANE: {
    label: "变道",
    intent: "CHANGE_LANE",
    autowareSpeed: 8.0,
    carlaSpeed: 6.0,
    note: "输出变道语义，由规划器结合车道线和障碍物确认可行性",
  },
  PULL_OVER: {
    label: "靠边停车",
    intent: "PULL_OVER",
    autowareSpeed: 8.0,
    carlaSpeed: 6.0,
    note: "交由高层规划器生成靠边轨迹",
  },
};

const scenarioData = {
  signal: {
    tag: "高优先级人工指挥",
    title: "信号灯故障路口",
    pain: "固定信号失效后，交警现场手势成为最高优先级通行依据，系统需要连续理解停止、直行和转向放行。",
    flow: "OpenMMLab pose -> GreedyIoUTracker -> SkeletonTCN -> state machine",
    command: "STOP / GO_STRAIGHT / TURN_LEFT",
    evidence: "在线演示视频、JSONL 输出、状态机诊断信息",
    policy: "人工指挥优先",
    effect: "速度约束 + 通行方向",
    syncCommand: "GO_STRAIGHT",
  },
  accident: {
    tag: "事故占道绕行",
    title: "事故占道与临时疏导",
    pain: "道路被事故车辆占用时，交警动作可能从减速、停止切换到绕行，需要系统抑制短时抖动。",
    flow: "主体选择 -> rolling buffer -> multitask command -> hold / expiry",
    command: "SLOW_DOWN / STOP / CHANGE_LANE",
    evidence: "离线演示视频、连续评测指标、command macro Jaccard",
    policy: "先降速再确认",
    effect: "限速提示 + 变道约束",
    syncCommand: "SLOW_DOWN",
  },
  control: {
    tag: "大型活动与交通管制",
    title: "临时交通管制放行",
    pain: "大型活动或管制区域中，车辆需要听从人工指挥进入等待、直行或转向阶段。",
    flow: "33 类方向标签 -> traffic intent -> control command",
    command: "LEFT_TURN_WAIT / GO_STRAIGHT / TURN_RIGHT",
    evidence: "CTPGesture v2 方向相关连续评测、需求对齐说明",
    policy: "方向语义清晰",
    effect: "规划器方向约束",
    syncCommand: "LEFT_TURN_WAIT",
  },
  construction: {
    tag: "施工与道路作业",
    title: "施工绕行与非交警干扰",
    pain: "施工场景中存在作业人员、路锥、车辆遮挡和背景干扰，需要先确认有效指挥主体。",
    flow: "候选人体检测 -> 反光背心先验 -> 轨迹稳定评分 -> command filter",
    command: "TURN_LEFT / CHANGE_LANE / SLOW_DOWN",
    evidence: "主体选择权重、复杂背景需求分析、诊断可视化",
    policy: "主体可信后输出",
    effect: "绕行轨迹提示",
    syncCommand: "TURN_LEFT",
  },
  checkpoint: {
    tag: "检查点人工接管",
    title: "封闭测试场与检查点",
    pain: "测试场或检查点常见靠边停车、等待确认和重新放行，需要命令保持与过期清空机制。",
    flow: "camera stream -> state machine -> JSONL / Autoware / CARLA",
    command: "PULL_OVER / STOP / GO_STRAIGHT",
    evidence: "infer_camera 命令、AutowareBridge、CARLA adapter",
    policy: "保守停车优先",
    effect: "靠边停车规划提示",
    syncCommand: "PULL_OVER",
  },
  interference: {
    tag: "多人干扰与遮挡",
    title: "多人候选与短时遮挡",
    pain: "路口可能同时出现行人、协管、施工人员和交警，系统需要保持主目标轨迹，不因短时遮挡立即切换。",
    flow: "center / size / visibility / hand / track_age / sticky bonus",
    command: "保持上一稳定命令或进入低置信回退",
    evidence: "OfficerSelector、TemporalCommandFilter、诊断叠加",
    policy: "稳定优先",
    effect: "不可靠时不覆盖规划",
    syncCommand: "STOP",
  },
};

const apiTemplates = {
  json(command) {
    return JSON.stringify(
      {
        frame_index: 1716,
        timestamp_sec: 68.64,
        active_track_id: 1,
        gesture: command.label,
        gesture_confidence: 0.98,
        intent: command.intent,
        command: command.key,
        command_confidence: 0.96,
        safe_fallback: false,
        latency_ms: 10.04,
        debug: {
          buffer_len: 64,
          visible_keypoints: 133,
          selector_score: 0.96,
          state_machine: {
            window_size: 7,
            min_consensus: 4,
            conf_threshold: 0.55,
            current_command: command.key,
          },
        },
      },
      null,
      2,
    );
  },
  ros2(command) {
    return `topic: /perception/traffic_police_command
type: std_msgs/msg/String

payload:
  command: ${command.key}
  gesture: ${command.label}
  intent: ${command.intent}
  confidence: 0.96
  active_track_id: 1
  safe_fallback: false

publisher:
  Ros2JsonPublisher(topic_name="/perception/traffic_police_command")`;
  },
  autoware(command) {
    return `AutowareBridge.publish_json(message)
AutowareBridge.publish_velocity_hint("${command.key}")

velocity rule:
  STOP / KEEP_WAIT / LEFT_TURN_WAIT -> 0.0 m/s
  SLOW_DOWN -> 3.0 m/s
  GO / TURN / CHANGE_LANE / PULL_OVER -> 8.0 m/s

current command:
  label: ${command.label}
  max_velocity: ${command.autowareSpeed.toFixed(1)} m/s
  suggested use: planner constraint or external velocity limit`;
  },
  carla(command) {
    return `PlannerHint(
  command="${command.key}",
  target_speed_mps=${command.carlaSpeed.toFixed(1)},
  note="${command.note}"
)

CARLA adapter behavior:
  if target_speed <= 0.1:
    throttle = 0.0
    brake = 1.0
  elif command == "SLOW_DOWN":
    throttle <= 0.2
    brake >= 0.2`;
  },
};

const commandTemplates = {
  setup: `conda create -n tpgr-av python=3.10 -y
conda activate tpgr-av
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd)/src`,
  train: `python -m tpgr.cli.train \\
  --config configs/ctpgesture_v1v2_tcn_phase2.yaml

python -m tpgr.cli.evaluate_classifier \\
  --config configs/debug_openmmlab_rtmo_phase2.yaml \\
  --split test \\
  --output-dir runs/phase2_tta_eval`,
  camera: `python -m tpgr.cli.infer_camera \\
  --config configs/deploy_openmmlab_rtmo_phase2.yaml \\
  --camera-id 0 \\
  --output-video outputs/camera_phase2.mp4 \\
  --output-jsonl outputs/camera_phase2.jsonl \\
  --debug`,
  video: `python -m tpgr.cli.infer_video \\
  --config configs/deploy_openmmlab_rtmo_phase2.yaml \\
  --input your_video.mp4 \\
  --output-video outputs/your_video_phase2.mp4 \\
  --output-jsonl outputs/your_video_phase2.jsonl`,
  debug: `python -m tpgr.cli.debug_dataset \\
  --config configs/debug_openmmlab_rtmo_phase2.yaml

# headless environment
python -m tpgr.cli.debug_dataset \\
  --config configs/debug_openmmlab_rtmo_phase2.yaml \\
  --no-display`,
  toy: `python tools/generate_toy_dataset.py --output-dir data/processed/toy
python -m tpgr.cli.train --config configs/toy_tcn.yaml
python -m tpgr.cli.infer_video \\
  --config configs/deploy_toy_tcn_demo.yaml \\
  --input data/processed/toy/demo/toy_demo.mp4 \\
  --output-video outputs/toy_tcn_demo.mp4 \\
  --output-jsonl outputs/toy_tcn_demo.jsonl`,
  fair: `python tools/eval_ctpgesture_v2_segmenter_ensemble.py \\
  --config configs/eval_ctpgesture_v2_segmenter_ensemble_refine_plus_rawdir_0p075_track239_0p025_classbias_best.yaml \\
  --split test \\
  --output-dir runs/ctpgesture_v2_segmenter_ensemble_refine_plus_rawdir_0p075_track239_0p025_classbias_best/eval_test_best`,
  onnx: `python -m tpgr.cli.export_onnx \\
  --config configs/ctpgesture_v1v2_tcn_phase2.yaml \\
  --checkpoint runs/phase2_tcn_big/best.pt \\
  --output exports/phase2_tcn_big_final.onnx`,
};

const builderPresets = {
  camera: {
    title: "摄像头实时推理",
    config: "configs/deploy_openmmlab_rtmo_phase2.yaml",
    input: "camera://0",
    outputVideo: "outputs/camera_phase2.mp4",
    outputJsonl: "outputs/camera_phase2.jsonl",
    outputDir: "runs/phase2_tta_eval",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  video: {
    title: "离线视频推理",
    config: "configs/deploy_openmmlab_rtmo_phase2.yaml",
    input: "your_video.mp4",
    outputVideo: "outputs/your_video_phase2.mp4",
    outputJsonl: "outputs/your_video_phase2.jsonl",
    outputDir: "runs/phase2_tta_eval",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  debug: {
    title: "单视频诊断",
    config: "configs/deploy_openmmlab_rtmo_phase2.yaml",
    input: "your_video.mp4",
    outputVideo: "outputs/your_video_debug.mp4",
    outputJsonl: "outputs/your_video_debug.jsonl",
    outputDir: "runs/debug_dataset",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  fair: {
    title: "公平 split 评测",
    config: "configs/eval_ctpgesture_v2_segmenter_ensemble_refine_plus_rawdir_0p075_track239_0p025_classbias_best.yaml",
    input: "data/processed/ctpgesture_v2_vibe33",
    outputVideo: "outputs/fair_eval_preview.mp4",
    outputJsonl: "outputs/fair_eval_preview.jsonl",
    outputDir: "runs/ctpgesture_v2_segmenter_ensemble_refine_plus_rawdir_0p075_track239_0p025_classbias_best/eval_test_best",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  wholebody: {
    title: "在线 wholebody 路线",
    config: "configs/deploy_openmmlab_wholebody_segmenter_online.yaml",
    input: "outputs/_smoke_input_004_120.mp4",
    outputVideo: "outputs/_smoke_wholebody_segmenter_online.mp4",
    outputJsonl: "outputs/_smoke_wholebody_segmenter_online.jsonl",
    outputDir: "runs/ctpgesture_v2_online_wholebody_segmenter_eval_full",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  onnx: {
    title: "导出 ONNX",
    config: "configs/ctpgesture_v1v2_tcn_phase2.yaml",
    input: "runs/phase2_tcn_big/best.pt",
    outputVideo: "outputs/onnx_check.mp4",
    outputJsonl: "outputs/onnx_check.jsonl",
    outputDir: "exports",
    onnxOutput: "exports/phase2_tcn_big_final.onnx",
  },
  toy: {
    title: "闭环自检",
    config: "configs/deploy_toy_tcn_demo.yaml",
    input: "data/processed/toy/demo/toy_demo.mp4",
    outputVideo: "outputs/toy_tcn_demo.mp4",
    outputJsonl: "outputs/toy_tcn_demo.jsonl",
    outputDir: "runs/toy_tcn",
    onnxOutput: "exports/toy_tcn.onnx",
  },
};

const releaseCheckLabels = {
  environment: "环境可创建，pip install -e . 已通过",
  assets: "code.zip、报告、需求文档、在线和离线演示资源可访问",
  camera: "摄像头实时推理命令可运行并生成 JSONL",
  video: "离线视频推理已生成 MP4 和 JSONL 结果",
  interface: "JSON / ROS2 / Autoware / CARLA 字段映射已校验",
  metrics: "在线、fair split、wholebody、all-data 指标口径已分开说明",
  limits: "fair-best 调优边界和真实车端部署边界已明确披露",
  presentation: "桌面、平板和手机视口完成可用性检查",
};

const fallbackCommand = {
  label: "无有效指挥",
  intent: "NO_ACTIVE_COMMAND",
  autowareSpeed: 0.0,
  carlaSpeed: 0.0,
  note: "不主动覆盖下游规划",
};

let activeCommand = "GO_STRAIGHT";
let activeApi = "json";
let activeWorkspaceTab = "builder";

function readStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Local storage can be unavailable in restricted browsers; the UI still works without persistence.
  }
}

function getCommand(commandKey) {
  return { ...(commandData[commandKey] || fallbackCommand), key: commandKey || "NO_COMMAND" };
}

function shellQuote(value) {
  const text = String(value ?? "").trim();
  if (!text) return "''";
  return /^[\w./:=@+-]+$/.test(text) ? text : `'${text.replace(/'/g, "'\\''")}'`;
}

function joinCommand(parts) {
  return parts.filter(Boolean).join(" \\\n  ");
}

function getBuilderElements() {
  return {
    mode: document.querySelector("#builder-mode"),
    config: document.querySelector("#builder-config"),
    input: document.querySelector("#builder-input"),
    camera: document.querySelector("#builder-camera"),
    outputVideo: document.querySelector("#builder-output-video"),
    outputJsonl: document.querySelector("#builder-output-jsonl"),
    outputDir: document.querySelector("#builder-output-dir"),
    onnxOutput: document.querySelector("#builder-onnx-output"),
    debug: document.querySelector("#builder-debug"),
    noDisplay: document.querySelector("#builder-no-display"),
    includeEnv: document.querySelector("#builder-env"),
  };
}

function getBuilderState() {
  const els = getBuilderElements();
  return {
    mode: els.mode.value,
    config: els.config.value,
    input: els.input.value,
    camera: els.camera.value || "0",
    outputVideo: els.outputVideo.value,
    outputJsonl: els.outputJsonl.value,
    outputDir: els.outputDir.value,
    onnxOutput: els.onnxOutput.value,
    debug: els.debug.checked,
    noDisplay: els.noDisplay.checked,
    includeEnv: els.includeEnv.checked,
  };
}

function setBuilderState(state) {
  const els = getBuilderElements();
  Object.entries({
    mode: "mode",
    config: "config",
    input: "input",
    camera: "camera",
    outputVideo: "outputVideo",
    outputJsonl: "outputJsonl",
    outputDir: "outputDir",
    onnxOutput: "onnxOutput",
  }).forEach(([elementKey, stateKey]) => {
    if (state[stateKey] !== undefined) {
      els[elementKey].value = state[stateKey];
    }
  });
  if (state.debug !== undefined) els.debug.checked = Boolean(state.debug);
  if (state.noDisplay !== undefined) els.noDisplay.checked = Boolean(state.noDisplay);
  if (state.includeEnv !== undefined) els.includeEnv.checked = Boolean(state.includeEnv);
}

function applyBuilderPreset(modeKey) {
  const preset = builderPresets[modeKey];
  if (!preset) return;
  const current = getBuilderState();
  setBuilderState({
    ...current,
    mode: modeKey,
    config: preset.config,
    input: preset.input,
    outputVideo: preset.outputVideo,
    outputJsonl: preset.outputJsonl,
    outputDir: preset.outputDir,
    onnxOutput: preset.onnxOutput,
    debug: modeKey !== "video" && modeKey !== "fair" && modeKey !== "onnx",
    noDisplay: modeKey === "wholebody",
  });
}

function buildEnvBlock() {
  return `cd /path/to/poliguide-av
conda activate tpgr-av
export PYTHONPATH=$(pwd)/src`;
}

function generateBuilderCommand(state) {
  const prefix = state.includeEnv ? `${buildEnvBlock()}\n\n` : "";
  if (state.mode === "camera") {
    return `${prefix}${joinCommand([
      "python -m tpgr.cli.infer_camera",
      `--config ${shellQuote(state.config)}`,
      `--camera-id ${shellQuote(state.camera)}`,
      `--output-video ${shellQuote(state.outputVideo)}`,
      `--output-jsonl ${shellQuote(state.outputJsonl)}`,
      state.debug ? "--debug" : "",
    ])}`;
  }
  if (state.mode === "video" || state.mode === "debug" || state.mode === "wholebody") {
    return `${prefix}${joinCommand([
      "python -m tpgr.cli.infer_video",
      `--config ${shellQuote(state.config)}`,
      `--input ${shellQuote(state.input)}`,
      `--output-video ${shellQuote(state.outputVideo)}`,
      `--output-jsonl ${shellQuote(state.outputJsonl)}`,
      state.debug ? "--debug" : "",
      state.noDisplay ? "--no-display" : "",
    ])}`;
  }
  if (state.mode === "fair") {
    return `${prefix}${joinCommand([
      "python tools/eval_ctpgesture_v2_segmenter_ensemble.py",
      `--config ${shellQuote(state.config)}`,
      "--split test",
      `--output-dir ${shellQuote(state.outputDir)}`,
    ])}`;
  }
  if (state.mode === "onnx") {
    return `${prefix}${joinCommand([
      "python -m tpgr.cli.export_onnx",
      `--config ${shellQuote(state.config)}`,
      `--checkpoint ${shellQuote(state.input)}`,
      `--output ${shellQuote(state.onnxOutput)}`,
    ])}`;
  }
  return `${prefix}python tools/generate_toy_dataset.py --output-dir data/processed/toy
python -m tpgr.cli.train --config configs/toy_tcn.yaml
${joinCommand([
    "python -m tpgr.cli.infer_video",
    `--config ${shellQuote(state.config)}`,
    `--input ${shellQuote(state.input)}`,
    `--output-video ${shellQuote(state.outputVideo)}`,
    `--output-jsonl ${shellQuote(state.outputJsonl)}`,
  ])}`;
}

function renderBuilderCommand() {
  const state = getBuilderState();
  const preset = builderPresets[state.mode] || builderPresets.camera;
  document.querySelector("#builder-mode-title").textContent = preset.title;
  document.querySelector("#builder-command").textContent = generateBuilderCommand(state);
  writeStorage("poliguide.builder", state);
}

function setWorkspaceTab(tabKey) {
  activeWorkspaceTab = tabKey;
  document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
    const selected = button.dataset.workspaceTab === tabKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll("[data-workspace-panel]").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.workspacePanel === tabKey);
  });
}

function sampleMessage() {
  return apiTemplates.json(getCommand("GO_STRAIGHT"));
}

function sampleJsonl() {
  return [
    {
      frame_index: 1716,
      timestamp_sec: 68.64,
      active_track_id: 1,
      gesture: "直行",
      gesture_confidence: 0.98,
      intent: "ALLOW_PASS",
      command: "GO_STRAIGHT",
      command_confidence: 0.96,
      safe_fallback: false,
      latency_ms: 10.04,
    },
    {
      frame_index: 1717,
      timestamp_sec: 68.68,
      active_track_id: 1,
      gesture: "减速慢行",
      gesture_confidence: 0.82,
      intent: "SLOW_DOWN",
      command: "SLOW_DOWN",
      command_confidence: 0.80,
      safe_fallback: false,
      latency_ms: 12.91,
    },
    {
      frame_index: 1718,
      timestamp_sec: 68.72,
      active_track_id: 1,
      gesture: "停止",
      gesture_confidence: 0.74,
      intent: "STOP",
      command: "STOP",
      command_confidence: 0.72,
      safe_fallback: false,
      latency_ms: 11.18,
    },
    {
      frame_index: 1719,
      timestamp_sec: 68.76,
      active_track_id: 1,
      gesture: "停止",
      gesture_confidence: 0.41,
      intent: "KEEP_WAIT",
      command: "LEFT_TURN_WAIT",
      command_confidence: 0.48,
      safe_fallback: true,
      latency_ms: 13.42,
    },
  ].map((item) => JSON.stringify(item)).join("\n");
}

function normalizePayload(payload) {
  return {
    frameIndex: payload.frame_index ?? payload.frame ?? payload.index,
    timestamp: payload.timestamp_sec ?? payload.timestamp ?? payload.time_sec,
    trackId: payload.active_track_id ?? payload.track_id ?? payload.person_id,
    gesture: payload.gesture ?? payload.gesture_label ?? payload.label,
    gestureConfidence: payload.gesture_confidence ?? payload.gesture_score ?? payload.score,
    intent: payload.intent ?? payload.traffic_intent,
    command: payload.command ?? payload.control_command ?? payload.vehicle_command,
    commandConfidence: payload.command_confidence ?? payload.intent_confidence ?? payload.confidence,
    safeFallback: payload.safe_fallback ?? payload.fallback_state ?? payload.fallback,
    latencyMs: payload.latency_ms ?? payload.inference_latency_ms,
  };
}

function parseJsonFromTextarea(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("请输入 JSON 消息。");
  if (trimmed.startsWith("{")) return JSON.parse(trimmed);
  const firstLine = trimmed.split(/\n+/).find(Boolean);
  if (!firstLine) throw new Error("未找到可解析的 JSON 行。");
  return JSON.parse(firstLine);
}

function reportItem(type, title, detail) {
  const item = document.createElement("article");
  item.className = `report-item ${type}`;
  const strong = document.createElement("strong");
  strong.textContent = title;
  const small = document.createElement("small");
  small.textContent = detail;
  item.append(strong, small);
  return item;
}

function bridgeHint(payload) {
  const normalized = normalizePayload(payload);
  const command = getCommand(normalized.command);
  return `normalized command:
  frame_index: ${normalized.frameIndex ?? "missing"}
  active_track_id: ${normalized.trackId ?? "missing"}
  gesture: ${normalized.gesture ?? "missing"}
  intent: ${normalized.intent ?? command.intent}
  command: ${command.key}
  confidence: ${normalized.commandConfidence ?? "missing"}
  safe_fallback: ${String(Boolean(normalized.safeFallback))}

ROS2:
  topic: /perception/traffic_police_command
  message_type: std_msgs/msg/String
  payload: JSON string

Autoware:
  publish_json(message)
  publish_velocity_hint("${command.key}")
  max_velocity: ${command.autowareSpeed.toFixed(1)} m/s

CARLA:
  PlannerHint(command="${command.key}", target_speed_mps=${command.carlaSpeed.toFixed(1)})
  note: ${command.note}`;
}

function validateInterfacePayload() {
  const input = document.querySelector("#message-input");
  const status = document.querySelector("#message-status");
  const report = document.querySelector("#message-report");
  const bridge = document.querySelector("#bridge-output");
  report.replaceChildren();
  try {
    const payload = parseJsonFromTextarea(input.value);
    const normalized = normalizePayload(payload);
    const errors = [];
    const warnings = [];
    [
      ["frameIndex", "缺少 frame_index / frame"],
      ["timestamp", "缺少 timestamp_sec / timestamp"],
      ["trackId", "缺少 active_track_id / track_id"],
      ["gesture", "缺少 gesture / gesture_label"],
      ["gestureConfidence", "缺少 gesture_confidence"],
      ["intent", "缺少 intent / traffic_intent"],
      ["command", "缺少 command / control_command"],
      ["commandConfidence", "缺少 command_confidence"],
      ["safeFallback", "缺少 safe_fallback / fallback_state"],
      ["latencyMs", "缺少 latency_ms"],
    ].forEach(([key, message]) => {
      if (normalized[key] === undefined || normalized[key] === null || normalized[key] === "") {
        errors.push(message);
      }
    });
    ["gestureConfidence", "commandConfidence"].forEach((key) => {
      const value = Number(normalized[key]);
      if (normalized[key] !== undefined && (Number.isNaN(value) || value < 0 || value > 1)) {
        errors.push(`${key} 应在 0 到 1 之间`);
      }
    });
    if (normalized.command && !commandData[normalized.command]) {
      warnings.push(`命令 ${normalized.command} 不在标准控制命令字典中，将按 NO_COMMAND 处理。`);
    }
    if (Number(normalized.commandConfidence) < 0.55 && !normalized.safeFallback) {
      warnings.push("命令置信度低于 0.55，建议启用 safe_fallback 或进入 KEEP_WAIT。");
    }
    if (Number(normalized.latencyMs) > 80) {
      warnings.push("latency_ms 高于 80ms，真实车端部署前建议排查姿态后端或视频输入瓶颈。");
    }
    errors.forEach((message) => report.append(reportItem("error", "字段错误", message)));
    warnings.forEach((message) => report.append(reportItem("warn", "风险提示", message)));
    if (!errors.length) {
      report.prepend(reportItem("ok", "接口字段可对接", "必需字段已齐备，可进入 ROS2 / Autoware / CARLA 桥接层。"));
    }
    status.textContent = errors.length ? "未通过" : warnings.length ? "通过，存在提示" : "通过";
    bridge.textContent = bridgeHint(payload);
  } catch (error) {
    status.textContent = "解析失败";
    bridge.textContent = "";
    report.append(reportItem("error", "JSON 无法解析", error.message));
  }
}

function parseJsonl(text) {
  const lines = text.split(/\n/).map((line) => line.trim()).filter(Boolean);
  const valid = [];
  const invalid = [];
  lines.forEach((line, index) => {
    try {
      valid.push(JSON.parse(line));
    } catch (error) {
      invalid.push({ index: index + 1, message: error.message });
    }
  });
  return { valid, invalid, total: lines.length };
}

function metricCard(label, value) {
  const article = document.createElement("article");
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  article.append(span, strong);
  return article;
}

function analyzeJsonl() {
  const input = document.querySelector("#jsonl-input");
  const status = document.querySelector("#jsonl-status");
  const summary = document.querySelector("#jsonl-summary");
  const bars = document.querySelector("#jsonl-bars");
  const { valid, invalid, total } = parseJsonl(input.value);
  summary.replaceChildren();
  bars.replaceChildren();
  if (!total) {
    status.textContent = "未解析";
    summary.append(metricCard("状态", "无数据"));
    return;
  }
  if (!valid.length) {
    status.textContent = "没有有效行";
    summary.append(metricCard("解析失败", `${invalid.length} 行`));
    return;
  }
  const counts = new Map();
  let latencySum = 0;
  let latencyCount = 0;
  let fallbackCount = 0;
  valid.forEach((item) => {
    const normalized = normalizePayload(item);
    const command = normalized.command || "NO_COMMAND";
    counts.set(command, (counts.get(command) || 0) + 1);
    const latency = Number(normalized.latencyMs);
    if (!Number.isNaN(latency)) {
      latencySum += latency;
      latencyCount += 1;
    }
    if (normalized.safeFallback) fallbackCount += 1;
  });
  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map((entry) => entry[1]));
  status.textContent = invalid.length ? `已解析，${invalid.length} 行无效` : "解析完成";
  summary.append(
    metricCard("有效帧", String(valid.length)),
    metricCard("命令种类", String(entries.length)),
    metricCard("平均延迟", latencyCount ? `${(latencySum / latencyCount).toFixed(2)} ms` : "未提供"),
    metricCard("安全回退", `${fallbackCount} 次`),
    metricCard("首个命令", normalizePayload(valid[0]).command || "NO_COMMAND"),
    metricCard("末尾命令", normalizePayload(valid[valid.length - 1]).command || "NO_COMMAND"),
  );
  entries.forEach(([command, count]) => {
    const row = document.createElement("div");
    row.className = "distribution-row";
    const strong = document.createElement("strong");
    strong.textContent = command;
    const bar = document.createElement("i");
    bar.style.setProperty("--w", `${Math.max((count / max) * 100, 4)}%`);
    const value = document.createElement("em");
    value.textContent = `${count}`;
    row.append(strong, bar, value);
    bars.append(row);
  });
  if (invalid.length) {
    bars.append(reportItem("warn", "无效 JSONL 行", invalid.map((item) => `第 ${item.index} 行`).join("、")));
  }
}

function renderReleaseState() {
  const checks = [...document.querySelectorAll("[data-release-check]")];
  const checked = checks.filter((input) => input.checked);
  const percent = Math.round((checked.length / checks.length) * 100);
  document.querySelector("#release-score").textContent = `${percent}%`;
  document.querySelector("#release-bar").style.width = `${percent}%`;
  document.querySelector("#release-caption").textContent = checked.length === checks.length
    ? "上线检查已完成"
    : `还剩 ${checks.length - checked.length} 项需要确认`;
  checks.forEach((input) => {
    input.closest("label").classList.toggle("is-checked", input.checked);
  });
  const done = checked.map((input) => `- ${releaseCheckLabels[input.dataset.releaseCheck]}`);
  const pending = checks
    .filter((input) => !input.checked)
    .map((input) => `- ${releaseCheckLabels[input.dataset.releaseCheck]}`);
  document.querySelector("#release-summary").textContent = `PoliGuide-AV 上线检查摘要

完成度：${percent}%

已确认：
${done.length ? done.join("\n") : "- 暂无"}

待确认：
${pending.length ? pending.join("\n") : "- 无"}

边界说明：
- 在线部署主线、公平 split benchmark、online wholebody、all-data 上限必须分开表述。
- fair-best 已超过论文公开 Jaccard 0.842，但包含同一 test split 上的多轮集成 / class-bias 调优。
- 真实车端部署前仍需在目标车辆平台、传感器、道路环境和安全仲裁链路中复测。`;
  writeStorage(
    "poliguide.release",
    Object.fromEntries(checks.map((input) => [input.dataset.releaseCheck, input.checked])),
  );
}

function setRoute(routeKey) {
  const data = routeData[routeKey];
  if (!data) return;
  document.querySelectorAll("[data-route]").forEach((button) => {
    const selected = button.dataset.route === routeKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelector("#route-title").textContent = data.title;
  document.querySelector("#route-desc").textContent = data.desc;
  document.querySelector("#route-points").replaceChildren(
    ...data.points.map((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      return item;
    }),
  );
}

function setApi(apiKey) {
  activeApi = apiKey;
  document.querySelectorAll("[data-api]").forEach((button) => {
    const selected = button.dataset.api === apiKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  renderApi();
}

function setCommand(commandKey) {
  activeCommand = commandKey;
  const command = getCommand(commandKey);
  document.querySelectorAll("[data-command]").forEach((button) => {
    const selected = button.dataset.command === commandKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  const heroCommand = document.querySelector("#hero-command");
  if (heroCommand) heroCommand.textContent = command.label;
  renderApi();
}

function renderApi() {
  const command = getCommand(activeCommand);
  document.querySelector("#api-output").textContent = apiTemplates[activeApi](command);
}

function setRunCommand(tabKey) {
  document.querySelectorAll("[data-command-tab]").forEach((button) => {
    const selected = button.dataset.commandTab === tabKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelector("#run-command").textContent = commandTemplates[tabKey];
}

function setScenario(scenarioKey) {
  const data = scenarioData[scenarioKey];
  if (!data) return;
  document.querySelectorAll("[data-scenario]").forEach((button) => {
    const selected = button.dataset.scenario === scenarioKey;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelector("#scenario-tag").textContent = data.tag;
  document.querySelector("#scenario-title").textContent = data.title;
  document.querySelector("#scenario-pain").textContent = data.pain;
  document.querySelector("#scenario-flow").textContent = data.flow;
  document.querySelector("#scenario-command").textContent = data.command;
  document.querySelector("#scenario-evidence").textContent = data.evidence;
  document.querySelector("#scenario-policy").textContent = data.policy;
  document.querySelector("#scenario-effect").textContent = data.effect;
  if (data.syncCommand && commandData[data.syncCommand]) {
    setCommand(data.syncCommand);
  }
}

document.querySelectorAll("[data-route]").forEach((button) => {
  button.addEventListener("click", () => setRoute(button.dataset.route));
});

document.querySelectorAll("[data-scenario]").forEach((button) => {
  button.addEventListener("click", () => setScenario(button.dataset.scenario));
});

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => setCommand(button.dataset.command));
});

document.querySelectorAll("[data-api]").forEach((button) => {
  button.addEventListener("click", () => setApi(button.dataset.api));
});

document.querySelectorAll("[data-command-tab]").forEach((button) => {
  button.addEventListener("click", () => setRunCommand(button.dataset.commandTab));
});

document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
  button.addEventListener("click", () => setWorkspaceTab(button.dataset.workspaceTab));
});

document.querySelector("#builder-mode").addEventListener("change", (event) => {
  applyBuilderPreset(event.target.value);
  renderBuilderCommand();
});

document.querySelector("#run-builder").addEventListener("input", renderBuilderCommand);

document.querySelector("#validate-message").addEventListener("click", validateInterfacePayload);

document.querySelector("#load-sample-message").addEventListener("click", () => {
  document.querySelector("#message-input").value = sampleMessage();
  validateInterfacePayload();
});

document.querySelector("#analyze-jsonl").addEventListener("click", analyzeJsonl);

document.querySelector("#load-sample-jsonl").addEventListener("click", () => {
  document.querySelector("#jsonl-input").value = sampleJsonl();
  analyzeJsonl();
});

document.querySelectorAll("[data-release-check]").forEach((input) => {
  input.addEventListener("change", renderReleaseState);
});

document.querySelector("#reset-release-checks").addEventListener("click", () => {
  document.querySelectorAll("[data-release-check]").forEach((input) => {
    input.checked = false;
  });
  renderReleaseState();
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.querySelector(`#${button.dataset.copyTarget}`);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.textContent);
      const original = button.textContent;
      button.textContent = "已复制";
      setTimeout(() => {
        button.textContent = original;
      }, 1400);
    } catch {
      button.textContent = "复制失败";
    }
  });
});

document.querySelectorAll("video").forEach((video) => {
  video.addEventListener("play", () => {
    document.querySelectorAll("video").forEach((other) => {
      if (other !== video && !other.paused) {
        other.pause();
      }
    });
  });
});

const navLinks = [...document.querySelectorAll(".side-nav a")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

let navTicking = false;

function updateActiveNav() {
  const marker = window.scrollY + Math.min(window.innerHeight * 0.36, 240);
  const active = sections.reduce((current, section) => {
    return section.offsetTop <= marker ? section : current;
  }, sections[0]);
  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${active.id}`);
  });
}

window.addEventListener("scroll", () => {
  if (navTicking) return;
  navTicking = true;
  requestAnimationFrame(() => {
    updateActiveNav();
    navTicking = false;
  });
});

window.addEventListener("hashchange", updateActiveNav);

setRoute("online");
setApi("json");
setRunCommand("camera");
setScenario("signal");
setBuilderState(readStorage("poliguide.builder", { mode: "camera", ...builderPresets.camera, debug: true, noDisplay: false, includeEnv: true }));
renderBuilderCommand();
document.querySelector("#message-input").value = sampleMessage();
validateInterfacePayload();
document.querySelector("#jsonl-input").value = sampleJsonl();
analyzeJsonl();
const savedRelease = readStorage("poliguide.release", {});
document.querySelectorAll("[data-release-check]").forEach((input) => {
  input.checked = Boolean(savedRelease[input.dataset.releaseCheck]);
});
renderReleaseState();
setWorkspaceTab(activeWorkspaceTab);
updateActiveNav();
