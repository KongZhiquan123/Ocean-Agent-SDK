# Error Handling Guide

KODE SDK implements a comprehensive error handling mechanism with three core principles:

1. **Model-Aware Errors** - All errors are visible and actionable by the model
2. **Never Crash** - Multi-layer error catching ensures system stability
3. **Full Observability** - All errors trigger events for monitoring and debugging

---

## Error Types

| Error Type | Identifier | Retryable | Typical Scenarios |
|------------|-----------|-----------|-------------------|
| `validation` | `_validationError: true` | No | Parameter type error, missing required params |
| `runtime` | `_thrownError: true` | Yes | File not found, permission denied, network error |
| `logical` | Tool returns `{ok: false}` | Yes | Content mismatch, command execution failed |
| `aborted` | Timeout/interrupt | No | Tool execution timeout, user interrupt |
| `exception` | Unexpected exception | Yes | System exception, unknown error |

---

## Error Flow

```
Tool Execution
  ├─ Parameter validation fails → {ok: false, error: ..., _validationError: true}
  ├─ Execution throws → {ok: false, error: ..., _thrownError: true}
  ├─ Returns {ok: false} → Keep as-is (logical error)
  └─ Normal return → Keep as-is
     ↓
Agent Processing
  ├─ Identify error type: validation | runtime | logical | aborted | exception
  ├─ Determine retryability: validation not retryable, others retryable
  ├─ Generate recommendations: based on error type and tool name
  ├─ Emit tool:error event (ProgressEvent - user visible)
  └─ Emit error event (MonitorEvent - monitoring system)
     ↓
Return to Model
  └─ {
       ok: false,
       error: "Specific error message",
       errorType: "error type",
       retryable: true/false,
       recommendations: ["suggestion 1", "suggestion 2", ...]
     }
```

---

## Listening to Errors

### Progress Events (User Layer)

```typescript
// Listen to tool errors for UI
agent.on('tool:error', (event) => {
  console.log('Tool error:', event.error);
  console.log('Tool state:', event.call.state);
  // Show UI notification
});

// Using stream
for await (const envelope of agent.stream(input)) {
  if (envelope.event.type === 'tool:error') {
    showNotification({
      type: 'error',
      message: envelope.event.error,
    });
  }
}
```

### Monitor Events (System Layer)

```typescript
// Listen to all errors
agent.on('error', (event) => {
  if (event.phase === 'tool') {
    const { errorType, retryable } = event.detail || {};

    // Log to logging system
    logger.warn('Tool Error', {
      message: event.message,
      errorType,
      retryable,
      severity: event.severity,
      timestamp: Date.now(),
    });

    // Send alerts
    if (event.severity === 'error') {
      alerting.send('Tool execution failed', event);
    }
  }
});
```

---

## Model Self-Adjustment

### Example: File Not Found

**Tool returns:**
```json
{
  "ok": false,
  "error": "File not found: /src/utils/helper.ts",
  "errorType": "logical",
  "retryable": true,
  "recommendations": [
    "Verify the file path is correct",
    "Use fs_glob to search for files",
    "Check if file was externally modified"
  ]
}
```

**Model analysis:**
1. `errorType: "logical"` - Not a parameter issue, file genuinely doesn't exist
2. `retryable: true` - Can try alternative approaches
3. Recommendations suggest "Verify the file path"

**Model adjustment:**
```
1. Use fs_glob("src/**/*.ts") to find all ts files
2. Use fs_grep("helper", "src/**/*.ts") to search for helper
3. Continue with the correct file path
```

### Example: Validation Error

**Tool returns:**
```json
{
  "ok": false,
  "error": "Invalid parameters: path is required",
  "errorType": "validation",
  "retryable": false,
  "recommendations": [
    "Check tool parameters against schema",
    "Ensure all required parameters are provided",
    "Verify parameter types are correct"
  ]
}
```

**Model adjustment:**
```
1. Check tool call, found missing path parameter
2. Add the required path parameter
3. Retry the tool call
```

---

## Multi-Layer Protection

```
Layer 1: Tool Execution (tool.ts)
  └─ try-catch catches all exceptions → {ok: false, _thrownError: true}

Layer 2: Agent Call (agent.ts)
  └─ try-catch catches call exceptions → errorType: 'exception'

Layer 3: Parameter Validation
  └─ safeParse prevents validation exceptions → {ok: false, _validationError: true}

Layer 4: Hook Execution
  └─ Hook failures don't affect main flow → Log error and continue
```

### Error Isolation Principles

- Single tool error ≠ Agent crash
- Agent error ≠ System crash
- Tools are completely isolated
- All errors are traceable

---

## Best Practices

### For Tool Developers

```typescript
// ✅ Recommended: Use {ok: false} for expected business errors
if (!fileExists) {
  return {
    ok: false,
    error: 'File not found',
    recommendations: ['Check file path', 'Use fs_glob to search'],
  };
}

// ❌ Avoid: Throwing exceptions for business errors
throw new Error('File not found');  // Only use for unexpected exceptions
```

### For Application Developers

```typescript
// Listen to errors and show UI
agent.on('tool:error', (event) => {
  showNotification({
    type: 'error',
    message: event.error,
    action: event.call.state === 'FAILED' ? 'retry' : null,
  });
});

// Smart retry logic
if (result.status === 'paused' && result.permissionIds?.length) {
  // Pending permissions, wait for user decision
} else if (lastError?.retryable && retryCount < 3) {
  // Retryable error, auto-retry
  await agent.send('Please adjust and retry based on recommendations');
}
```

### For Operations

```typescript
// Error statistics and analysis
const errorStats = {
  validation: 0,
  runtime: 0,
  logical: 0,
  aborted: 0,
  exception: 0,
};

agent.on('error', (event) => {
  if (event.phase === 'tool') {
    const type = event.detail?.errorType || 'unknown';
    errorStats[type]++;

    // Analyze error patterns periodically
    if (errorStats.validation > 100) {
      alert('Too many validation errors, check tool schema config');
    }
  }
});
```

---

## Error Event Types

### ProgressToolErrorEvent

```typescript
interface ProgressToolErrorEvent {
  channel: 'progress';
  type: 'tool:error';
  call: ToolCallSnapshot;  // Tool call snapshot
  error: string;           // Error message
  bookmark?: Bookmark;
}
```

### MonitorErrorEvent

```typescript
interface MonitorErrorEvent {
  channel: 'monitor';
  type: 'error';
  severity: 'warn' | 'error';
  phase: 'model' | 'tool' | 'sandbox' | 'system';
  message: string;
  detail?: {
    errorType?: string;
    retryable?: boolean;
    [key: string]: any;
  };
}
```

---

## Summary

The error handling mechanism provides:

**Model Intelligence**
- Clear error types (validation/runtime/logical/aborted/exception)
- Explicit retryability (retryable: true/false)
- Actionable recommendations (customized by tool and error type)

**System Stability**
- Tool layer try-catch fallback
- Agent layer try-catch protection
- Parameter validation safeParse
- Hook execution isolation

**Full Observability**
- Progress events (tool:error) - user visible
- Monitor events (error) - system logging
- Tool records (ToolCallRecord) - complete audit
- Event timeline (EventBus) - traceable

---

## References

- [Events Guide](./events.md)
- [Tools Guide](./tools.md)
- [Resume/Fork Guide](./resume-fork.md)
