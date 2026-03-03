# API Usage Reference

## Chat Stream Endpoint

```http
POST /api/chat/stream
Content-Type: application/json
X-API-Key: your-secret-key

{
  "message": "请帮我创建一个 hello.py 文件",
  "mode": "edit",
  "context": {
    "userId": "user123",
    "workingDir": "/path/to/work"
  },
  "agentId": "agt-abc123"  // Optional: for multi-turn conversations
}
```

**SSE Event Types**:
- `start`: Processing begins (includes `agentId` and `isNewSession`)
- `heartbeat`: Keep-alive every 2 seconds
- `text`: AI-generated text content
- `tool_use`: Tool invocation started
- `tool_result`: Tool execution result
- `done`: Processing complete
- `error`: Error occurred

## Multi-turn Conversations

To continue a conversation, include the `agentId` from the previous response's `start` event. The conversation manager will reuse the Agent instance if it hasn't expired (30 min timeout).
