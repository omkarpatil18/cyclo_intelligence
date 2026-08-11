# Action-step consumption acknowledgement from an external environment.

# One action row was applied by env.step().
uint8 STATUS_EXECUTED = 0
# Every action row in the chunk was applied.
uint8 STATUS_COMPLETED = 1
# The chunk was discarded before all rows were applied.
uint8 STATUS_CANCELLED = 2

uint64 session_id
uint64 seq_id
int32 action_index
int32 executed_steps
int32 chunk_size
uint8 status
float64[] executed_action
float64 timestamp
