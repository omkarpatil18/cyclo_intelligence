# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, dependency-light helpers for inference recording outcomes."""

from interfaces.srv import RecordingCommand


def validate_episode_outcome(value) -> int:
    outcome = int(value or 0)
    valid = {
        RecordingCommand.Request.EPISODE_OUTCOME_UNSPECIFIED,
        RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
        RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
    }
    if outcome not in valid:
        raise ValueError(f'Invalid episode outcome: {outcome}')
    return outcome


def forward_inference_record_stop(request, forward_recording):
    """Validate and forward an inference-record STOP with its outcome."""
    outcome = validate_episode_outcome(
        getattr(request, 'episode_outcome', 0)
    )
    if outcome == RecordingCommand.Request.EPISODE_OUTCOME_UNSPECIFIED:
        raise ValueError(
            'Inference recording outcome must be Success or Fail'
        )
    return forward_recording(
        RecordingCommand.Request.STOP,
        task_info=request.task_info,
        episode_outcome=outcome,
    )
