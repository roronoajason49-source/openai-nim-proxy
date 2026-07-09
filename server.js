// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Auto-fix API URL layouts
let NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
if (!NIM_API_BASE.startsWith('http://') && !NIM_API_BASE.startsWith('https://')) {
  NIM_API_BASE = 'https://' + NIM_API_BASE;
}
NIM_API_BASE = NIM_API_BASE.replace(/\/+$/, '');

const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = true; 

// Model mapping
const MODEL_MAPPING = {
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'stepfun-ai/step-3.7-flash': 'stepfun-ai/step-3.7-flash', 
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimaxai/minimax-m3': 'minimaxai/minimax-m3',
  'minimax-m2.7': 'minimaxai/minimax-m2.7',
  'qwen-122b': 'qwen/qwen3.5-122b-a10b',         
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'z-ai/glm-5.1': 'z-ai/glm-5.2', 
  'glm-5.1': 'z-ai/glm-5.2'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy', reasoning_display: SHOW_REASONING });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'stepfun-ai/step-3.7-flash';
    
    const normalizedMessages = [];
    let isFirstSystem = true;

    // Forces reasoning to trigger and keeps character dialogue outside the thought block
    const FORCE_THINKING_PROMPT = "\n\n[CRITICAL SYSTEM DIRECTIVE: You are an advanced reasoning model. You MUST ALWAYS start every single response by thinking. Wrap your internal thoughts, character logic, and planning strictly inside <think> and </think> tags. NEVER skip the <think> phase, even for short responses. NEVER put actual roleplay dialogue inside the <think> tags. Write your actual roleplay response only AFTER closing the </think> tag.]";

    for (const msg of messages) {
      if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
      
      let role = msg.role.toLowerCase();
      
      if (role === 'system') {
        if (isFirstSystem && normalizedMessages.length === 0) {
          normalizedMessages.push({ role: 'system', content: msg.content + FORCE_THINKING_PROMPT });
          isFirstSystem = false;
          continue;
        } else {
          role = 'user';
        }
      }
      
      if (normalizedMessages.length > 0 && normalizedMessages[normalizedMessages.length - 1].role === role) {
        normalizedMessages[normalizedMessages.length - 1].content += '\n\n' + msg.content;
      } else {
        normalizedMessages.push({ role, content: msg.content });
      }
    }
    
    if (normalizedMessages.length > 1 && normalizedMessages[1].role === 'assistant') {
      normalizedMessages.splice(1, 0, { role: 'user', content: 'Hello.' });
    }

    const safe_temp = (parseFloat(temperature) > 0) ? parseFloat(temperature) : 0.6;
    
    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      temperature: safe_temp,
      top_p: req.body.top_p ?? 1.0,
      max_tokens: 4096, 
      stream: stream || false
    };
    
    // Set proper API hardware keys for each model family
    if (nimModel.includes('step-3.7')) {
      nimRequest.reasoning_effort = "high";
    } else if (nimModel.includes('glm-5.2')) {
      nimRequest.reasoning_effort = "high"; 
    } else if (nimModel.includes('minimax')) {
      nimRequest.reasoning_effort = "high";
      nimRequest.thinking = { type: "enabled" }; 
    } else if (nimModel.includes('deepseek-v4')) {
      nimRequest.chat_template_kwargs = { enable_thinking: true, thinking: true };
    }
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); 
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          line = line.trim();
          if (!line) return; 
          
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              if (reasoningStarted) {
                const closeChunk = {
                  id: `chatcmpl-${Date.now()}`,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: nimModel,
                  choices: [{ index: 0, delta: { content: '\n
