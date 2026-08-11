# Sequenced raw action chunk delivered to an external step consumer.

# Identifies the active inference session. Consumers must discard chunks from
# an old session after reset, stop, or policy reload.
uint64 session_id

# Monotonic chunk id within the inference session.
uint64 seq_id

# chunk_size × action_dim = len(data). data is row-major. A zero-sized message
# for the currently active (session_id, seq_id) cancels its remaining actions.
int32 chunk_size
int32 action_dim
float64[] data
