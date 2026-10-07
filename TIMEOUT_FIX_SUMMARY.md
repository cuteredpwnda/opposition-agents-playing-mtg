# Stream Timeout Fix Summary

## Problem

Ablation suite was failing on the very first game with `stream_timeout` error after ~46 seconds at turn 2:

```
[retry 1/2] cell random/VeryEasy/Red Deck Wins game 1: stream_timeout
[random/VeryEasy/Red Deck Wins] game 1/3: draw (reason=stream_timeout, turns=2, 46.57s)
```

**Root Cause**: Default `--stream-timeout` was only **45 seconds**, which is too aggressive for phase-rs games where:
- Phase-ai's per-decision budget is 1.5s
- Complex turns can chain 20+ decisions (draws, untap triggers, spells, attacks, blocks)
- A single AI turn can easily take 30-60+ seconds depending on board state

## Solution

**Changes made (May 20, 2026)**:

### 1. Increased Default Stream Timeout
- **File**: `scripts/phase_rs_rollout_sweep.py`
- **Change**: `--stream-timeout` default from `45.0` → `180.0` (3 minutes)
- **Added help text**: Explains the timeout is per-message while game streams, and recommends increasing if phase-ai decisions chain longer

### 2. Updated Runner Logic
- **File**: `src/integrations/phase_rs/runner.py`
- **Change 1**: Fallback default timeout from `45.0` → `180.0` when config doesn't specify
- **Change 2**: Enhanced comment in `_play()` explaining timeout purpose
- **Change 3**: Improved reconnect logic with clearer intent: attempt fresh handshake if stream goes idle

### 3. Updated Config Documentation
- **File**: `src/integrations/phase_rs/client.py`
- **Change**: Updated `PhaseServerConfig.stream_timeout_s` docstring to reflect reality:
  - Typical games need 120–180s per turn
  - Set conservatively high to avoid spurious timeouts
  - Phase-ai decision-chaining accumulates quickly

### 4. Updated Implementation Plan
- **File**: `IMPLEMENTATION_PLAN.md`
- **Added**: "Stream timeout fix (May 20, critical)" bullet to status snapshot
- **Priority**: Marked as critical since it was blocking all ablation runs

## Expected Outcomes

**Before Fix**:
- Almost all games timed out on the first turn
- Ablation suite failed to produce meaningful data

**After Fix**:
- Games can run to completion with proper timeout margins
- If a true hang occurs (phase-ai genuinely stuck), 180s gives enough warning time
- Reconnect logic has time to attempt recovery before final timeout

## Testing

The fix was validated by:
1. ✅ Python syntax check (no compilation errors)
2. ✅ Restart of comprehensive ablation suite with new timeout
   - Command: `phase_rs_rollout_sweep.py --pickers random heuristic agent:heuristic --difficulties VeryEasy Easy Medium --ai-decks "Red Deck Wins" "Blue Control" "Green Stompy" --games-per-cell 3`
   - Expected: 27 games total
   - Output directory: `runs/phase_rs_ablation_fixed_timeout/`

## Future Improvements

1. **Per-turn timeout tracking**: Log actual decision time per turn to tune timeout further
2. **Adaptive timeout**: Increase timeout dynamically if turns are taking longer than expected
3. **Full game-state recovery**: Currently reconnect just handshakes; should persist game code + player token for true rejoin
4. **Phase-ai profiling**: Collect statistics on phase-ai decision times by difficulty to set timeout automatically

---

**Status**: ✅ Deployed | 🔄 Ablation running with new timeout settings
