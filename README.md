# z-video-study-webpage-qwen

> WorkBuddy 技能：把一段本地视频变成「图文并茂、可交互」的学习总结网页。
> 基于 Qwen 多模态大模型（通义千问 DashScope）逐段分析视频画面 + 语音转录。

---

## 这个技能能做什么

输入一个本地 MP4，自动产出一份自包含的 HTML 学习总结页：

1. **抽帧** —— 用 ffmpeg 按时间轴抽取关键帧
2. **转录** —— 用 Whisper（可选）把语音转成文字
3. **多模态分析** —— 调用 Qwen（qwen3.7-plus 等）逐段看懂画面 + 文案，提炼知识点、时间线、对比、思考题
4. **生成网页** —— 输出米色纸张风格的 `study-summary-*.html`，每个知识点都绑定到对应画面

适用于：教学视频二次加工、讲座/公开课整理、培训素材沉淀。

---

## 前置依赖

- **ffmpeg**：必需，用于抽帧与音频提取（脚本会自动探测 PATH 及常见 Windows 安装位置，找不到则回退 `shutil.which("ffmpeg")`）
- **DashScope API Key**：必需，用于调用 Qwen 多模态模型
  - 申请：https://dashscope.console.aliyun.com/ → API-KEY 管理
  - 运行前设置环境变量（**不要写进任何文件**）：
    ```powershell
    $env:DASHSCOPE_API_KEY = "你的key"
    ```

---

## 朋友怎么装这个技能

1. **下载 `SKILL.md` 直接拖进 WorkBuddy 聊天框**
   - 最轻量：只传一个文件，AI 自动识别为技能
2. **把整个仓库文件夹放进 `~/.workbuddy/skills/`**
   - 把本仓库 clone / 解压后，整个 `z-video-study-webpage-qwen/` 目录移到 `~/.workbuddy/skills/` 下即可
3. **用 npx 命令一键安装**
   ```bash
   npx skills add iithink88/z-video-study-webpage-qwen@z-video-study-webpage-qwen
   ```

装好后，对 AI 说「用 Qwen 分析这个视频 C:\路径\xxx.mp4」，AI 会自动跑完抽帧 → 分析 → 生成 HTML 的全流程。

---

## 目录结构

```
z-video-study-webpage-qwen/
├── SKILL.md                      # 核心：AI 执行指令 + 参数说明
├── scripts/qwen_video_study.py   # 主脚本：抽帧 + 转录 + Qwen 分析 + 生成 HTML
├── tests/test_qwen_video_study.py# 单元测试（mock 模式可离线跑）
└── evals/evals.json              # 技能评测用例
```

---

## 备注

- API Key 仅从环境变量 `DASHSCOPE_API_KEY` 读取，脚本内置 `scrub_secret()` 防泄露，输出不含密钥
- 支持 `--mock-analysis` 模式，无 Key 也能跑通整条流程做验证
