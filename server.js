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

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = true; 

// Model mapping
const MODEL_MAPPING = {
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash', 
  'qwen-122b': 'qwen/qwen3.5-122b-a10b',         
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'z-ai/glm-5.1': 'z-ai/glm-5.2', // Auto-upgrades deprecated GLM 5.1 requests
  'glm-5.1': 'z-ai/glm-5.2'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Exact match or safe fallback to the new GLM-5.2 model
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'z-ai/glm-5.2';
    
    // 🔥 THE FIX 1: Message Sanitization
    // Strip out JanitorAI's hidden 'name' fields to prevent SGLang Pydantic validation 400 errors
    const sanitizedMessages = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    // Construct the standard payload
    const nimRequest = {
      model: nimModel,
      messages: sanitizedMessages,
      temperature: temperature ?? 0.6,
      top_p: req.body.top_p ?? 1.0,
      max_tokens: max_tokens ? Math.min(max_tokens, 8192) : 4096,
      stream: stream || false
    };
    
    // Model-specific reasoning triggers (Only apply if strictly required by the specific model family)
    if (nimModel.includes('deepseek-v4')) {
      nimRequest.chat_template_kwargs = { enable_thinking: true, thinking: true };
    }
    // Note: GLM-5.2 enables reasoning natively without flags. Extra flags cause strict validation drops.
    
    // Request execution
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
              res.write('data: [DONE]\n\n'); 
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                const reasoning = delta.reasoning_content || delta.reasoning || '';
                const content = delta.content || '';
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  data.choices[0].delta.content = combinedContent || content;
                } else {
                  data.choices[0].delta.content = content;
                }
                
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Stream processing interruption:', err);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const reasoning = choice.message?.reasoning_content || choice.message?.reasoning;
          
          if (SHOW_REASONING && reasoning) {
            fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    // 🔥 THE FIX 2: Transparent Error Logging
    // This forces the proxy to print the *exact* JSON breakdown of what NVIDIA didn't like
    const errorDetail = error.response?.data;
    const exactMessage = errorDetail 
      ? (typeof errorDetail === 'object' ? JSON.stringify(errorDetail) : errorDetail) 
      : error.message;

    console.error('Proxy connectivity error:', exactMessage);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: `NVIDIA API Error: ${exactMessage}`,
        type: 'proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
