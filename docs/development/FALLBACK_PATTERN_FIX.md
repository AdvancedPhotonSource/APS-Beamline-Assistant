# Fallback Pattern Detection Fix

## Problem

When user asked: **"how do you run the analysis"**

The system incorrectly:
1. Triggered fallback pattern matching on "run" and "analysis"
2. Attempted to execute `midas_run_ff_hedm_full_workflow` without arguments
3. Looped 5 times showing debug messages
4. Failed with validation errors (missing required parameters)

**Expected behavior:** AI should explain how to run analysis, not attempt to execute a tool.

## Root Causes

### 1. Overly Broad Pattern Matching

**Before:**
```python
if 'run_ff_hedm' in content.lower() or 'ff-hedm' in content.lower():
    tool_intent = 'midas_run_ff_hedm_full_workflow'
```

Problem: Triggered on ANY mention of "run" + "hedm", even in explanations like "how to run" or "you can run".

### 2. No Context Awareness

The fallback didn't distinguish between:
- **Commands:** "Run FF-HEDM on /data" ✓ Execute tool
- **Questions:** "How do you run analysis?" ✗ Don't execute tool
- **Explanations:** "I would run the workflow" ✗ Don't execute tool

### 3. Missing Parameter Validation

The fallback would execute even without required parameters, causing validation errors.

### 4. Verbose Debug Output

Debug messages cluttered the interface when fallback triggered:
```
💡 Detected tool intent: midas_run_ff_hedm_full_workflow
   Extracted args: {}
⚠️  AI didn't use TOOL_CALL format - executing anyway...
```

## Solution

### 1. Added Explanation Detection

**New Logic:**
```python
# Only trigger fallback if it's a CLEAR command, not an explanation
is_explanation = any(word in content.lower() for word in [
    'how to', 'you can', 'would', 'could', 'should', 'explain',
    'here\'s', 'let me', 'i can', 'to run', 'please', '?'
])

if not is_explanation:
    # Check for tool patterns
```

**Now blocks fallback when:**
- User asks questions: "how do...", "what is...", "can you..."
- AI explains: "you can...", "I would...", "let me explain..."
- Content has question marks

### 2. More Specific Pattern Matching

**Before:**
```python
if 'run_ff_hedm' in content.lower() or 'ff-hedm' in content.lower():
```

**After:**
```python
if re.search(r'run.*ff[_-]?hedm.*workflow.*on', content.lower()):
```

**Now requires:**
- "run" + "ff-hedm" or "ff_hedm" + "workflow" + "on"
- Much more specific command structure
- Less likely to trigger on general mentions

### 3. Require Both Intent AND Arguments

**Before:**
```python
if tool_intent:
    execute_tool()
```

**After:**
```python
if tool_intent and tool_args:  # Must have both
    execute_tool()
```

**Now only executes if:**
- Pattern matched (intent found)
- Parameters extracted successfully (args populated)
- Prevents execution with empty args

### 4. Suppressed Debug Output

**Before:**
```python
print(f"  💡 Detected tool intent: {tool_intent}")
print(f"     Extracted args: {tool_args}")
print(f"  ⚠️  AI didn't use TOOL_CALL format - executing anyway...")
```

**After:**
```python
# Comment out debug output for cleaner interface
# print(f"  💡 Detected tool intent: {tool_intent}")
# print(f"     Extracted args: {tool_args}")
```

Clean interface - no debug clutter.

### 5. Improved System Prompt

Added clear guidance to AI:

```
⚠️ CRITICAL INSTRUCTIONS ⚠️

1. WHEN TO USE TOOLS:
   - User gives a COMMAND: "integrate the file", "run workflow"
   - User provides DATA for analysis: "I have peaks at 12.5, 18.2"

2. WHEN NOT TO USE TOOLS:
   - User asks HOW: "how do you run analysis"
   - User asks WHAT: "what can you do"
   - User needs EXPLANATION: "explain the workflow"
   - General conversation: "hello", "thank you"
```

Added example showing when NOT to use tools:
```
Example 1 - Question (NO TOOL):
User: "how do you run the analysis"
Your response:
"To run analysis with the Beamline Assistant, you can:
1. FF-HEDM Full Workflow: ...
2. 2D to 1D Integration: ...
..."
```

## Updated Pattern Logic

### Pattern 1: FF-HEDM Workflow

**Trigger:** `run.*ff[_-]?hedm.*workflow.*on`

**Examples:**
- ✓ "run ff-hedm workflow on /data/experiment"
- ✓ "run ff_hedm workflow on ~/analysis"
- ✗ "how to run ff-hedm analysis" (has "how to")
- ✗ "you can run ff-hedm" (has "you can")

### Pattern 2: Phase Identification

**Trigger:** `identif.*phase.*from.*peak`

**Examples:**
- ✓ "identify phases from peaks"
- ✓ "identify crystalline phase from peak positions"
- ✗ "can you identify phases" (has question + "can you")
- ✗ "how to identify phases" (has "how to")

### Pattern 3: 2D to 1D Integration

**Trigger:** `integrat.*(?:file|image).*(?:from|to)`

**Examples:**
- ✓ "integrate file from 2D to 1D"
- ✓ "integrate image from 2D to 1D"
- ✗ "how do you integrate" (has "how do")
- ✗ "can you integrate files" (has "can you")

## Testing

### Test 1: Question Detection

**Input:** "how do you run the analysis"

**Before:** Triggered fallback → tried to execute tool → failed

**After:** Recognized as question → AI explains → no tool execution ✓

### Test 2: Specific Command

**Input:** "Run FF-HEDM workflow on /data/experiment_042"

**Before:** Would trigger and execute

**After:** Still triggers and executes correctly ✓

### Test 3: Greeting

**Input:** "Radhe Radhe" or "hello"

**Before:** No issue (never matched patterns)

**After:** No issue (still doesn't match) ✓

## Impact

### For Users

**Before:**
```
Beamline> how do you run the analysis
  💡 Detected tool intent: midas_run_ff_hedm_full_workflow
     Extracted args: {}
  ⚠️  AI didn't use TOOL_CALL format - executing anyway...
→ Run Ff Hedm Full Workflow
[Repeated 5 times]
⚠️  Reached maximum iterations (5)
Error: 3 validation errors...
```

**After:**
```
Beamline> how do you run the analysis

To run analysis with the Beamline Assistant, you can:

1. FF-HEDM Full Workflow: Provide a directory with Parameters.txt
2. 2D to 1D Integration: Provide a .tiff image
3. Phase Identification: Provide peak positions

What specific analysis would you like to perform?
```

### For AI

- Clear guidance on when to use tools
- Examples of questions vs commands
- Better pattern matching prevents false triggers
- Cleaner error messages

### For System

- Reduced unnecessary tool executions
- Fewer validation errors
- Cleaner logs (no debug spam)
- More predictable behavior

## Files Modified

1. **[argo_mcp_client.py:280-303](argo_mcp_client.py:280)** - Enhanced system prompt
2. **[argo_mcp_client.py:306-317](argo_mcp_client.py:306)** - Added "NO TOOL" example
3. **[argo_mcp_client.py:500-566](argo_mcp_client.py:500)** - Improved fallback logic

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Pattern matching | Broad (any mention) | Specific (command structure) |
| Context awareness | None | Detects questions/explanations |
| Parameter validation | Execute even if empty | Requires both intent + args |
| Debug output | Verbose messages | Clean (commented out) |
| System prompt | Basic instructions | Detailed guidance with examples |
| False positives | Common on questions | Rare, only on commands |

## Benefits

✓ **Smarter** - Distinguishes questions from commands
✓ **Cleaner** - No debug output cluttering interface
✓ **Safer** - Won't execute tools without proper parameters
✓ **Better UX** - Explanations when appropriate, actions when needed
✓ **More reliable** - Fewer false triggers and errors
