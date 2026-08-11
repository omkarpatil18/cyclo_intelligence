// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useCallback, useEffect, useRef } from 'react';
import {
  shallowEqual,
  useDispatch,
  useSelector,
  useStore,
} from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCheckCircle,
  MdFiberManualRecord,
  MdHighlightOff,
} from 'react-icons/md';
import { EpisodeOutcome } from '../constants/taskCommand';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import {
  InferenceRecordingUiPhase,
  selectInferenceRecordingControl,
  setInferenceRecordingUiPhase,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

const outcomeLabels = {
  [EpisodeOutcome.SUCCESS]: 'Success',
  [EpisodeOutcome.FAILURE]: 'Fail',
};

let inferenceRecordingCommandGeneration = 0;

export default function InferenceRLDataCollectPanel() {
  const dispatch = useDispatch();
  const store = useStore();
  const inferencePhase = useSelector(
    (state) => state.tasks.inferenceStatus.inferencePhase
  );
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const control = useSelector(selectInferenceRecordingControl, shallowEqual);
  const { sendRecordCommand } = useRosServiceCaller();
  const commandPendingRef = useRef(false);

  useEffect(() => {
    if (control.uiPhase === InferenceRecordingUiPhase.IDLE) {
      commandPendingRef.current = false;
    }
  }, [control.uiPhase]);

  const canStart =
    inferencePhase === InferencePhase.INFERENCING &&
    recordStatus.recordPhase === RecordPhase.READY &&
    control.uiPhase === InferenceRecordingUiPhase.IDLE;
  const canLabel =
    control.active &&
    !control.pending &&
    !control.serverSaving;

  const handleStartRecording = useCallback(async () => {
    if (!canStart || commandPendingRef.current) return;
    commandPendingRef.current = true;
    const commandGeneration = ++inferenceRecordingCommandGeneration;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.STARTING
    ));
    try {
      const result = await sendRecordCommand('start_inference_record');
      if (!result?.success) {
        throw new Error(result?.message || 'Recording start failed');
      }
      if (commandGeneration !== inferenceRecordingCommandGeneration) return;
      const currentPhase = store.getState().tasks.inferenceRecordingUi.phase;
      if (currentPhase === InferenceRecordingUiPhase.IDLE) return;
      if (currentPhase === InferenceRecordingUiPhase.STARTING) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.ACTIVE
        ));
      }
      commandPendingRef.current = false;
      toast.success('Inference recording started');
    } catch (error) {
      if (commandGeneration !== inferenceRecordingCommandGeneration) return;
      const currentPhase = store.getState().tasks.inferenceRecordingUi.phase;
      if (currentPhase === InferenceRecordingUiPhase.IDLE) return;
      if (currentPhase === InferenceRecordingUiPhase.STARTING) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.IDLE
        ));
      }
      commandPendingRef.current = false;
      toast.error(error?.message || 'Recording start failed');
    }
  }, [canStart, dispatch, sendRecordCommand, store]);

  const handleOutcome = useCallback(async (episodeOutcome) => {
    if (!canLabel || commandPendingRef.current) return;
    commandPendingRef.current = true;
    const commandGeneration = ++inferenceRecordingCommandGeneration;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.STOPPING
    ));
    try {
      const result = await sendRecordCommand('stop_inference_record', {
        episodeOutcome,
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording save failed');
      }
      if (commandGeneration !== inferenceRecordingCommandGeneration) return;
      toast.success(`Episode saved as ${outcomeLabels[episodeOutcome]}`);
      // Keep STOPPING until /data/recording/status confirms READY. This
      // prevents a stale RECORDING sample from allowing a duplicate label.
    } catch (error) {
      if (commandGeneration !== inferenceRecordingCommandGeneration) return;
      const currentPhase = store.getState().tasks.inferenceRecordingUi.phase;
      if (currentPhase === InferenceRecordingUiPhase.IDLE) return;
      if (currentPhase === InferenceRecordingUiPhase.STOPPING) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.ACTIVE
        ));
      }
      commandPendingRef.current = false;
      toast.error(error?.message || 'Recording save failed');
    }
  }, [canLabel, dispatch, sendRecordCommand, store]);

  const recordingLabel = {
    [InferenceRecordingUiPhase.STARTING]: 'Starting…',
    [InferenceRecordingUiPhase.ACTIVE]: 'Recording…',
    [InferenceRecordingUiPhase.STOPPING]: 'Saving…',
  }[control.uiPhase] || 'Recording';

  const statusMessage = control.serverSaving ||
    control.uiPhase === InferenceRecordingUiPhase.STOPPING
    ? 'Saving the labeled episode…'
    : control.active
      ? 'Choose Success or Fail to save this episode.'
      : inferencePhase === InferencePhase.INFERENCING
        ? 'Start recording the current inference rollout.'
        : 'Start Inference to enable recording.';

  return (
    <section
      className="mt-3 border-t border-gray-300 pt-3"
      aria-labelledby="rl-data-collect-title"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3
          id="rl-data-collect-title"
          className="text-sm font-semibold text-gray-700"
        >
          RL Data Collect
        </h3>
        {control.active && (
          <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
            REC
          </span>
        )}
      </div>

      <button
        type="button"
        onClick={handleStartRecording}
        disabled={!canStart}
        aria-label="Start inference recording"
        className={clsx(
          'flex h-10 w-full items-center justify-center gap-2 rounded-md',
          'text-sm font-semibold transition-colors focus:outline-none focus:ring-2',
          canStart
            ? 'bg-red-500 text-white hover:bg-red-600 focus:ring-red-300'
            : control.active
              ? 'bg-red-100 text-red-600'
              : 'cursor-not-allowed bg-gray-200 text-gray-500'
        )}
      >
        <MdFiberManualRecord size={19} />
        {recordingLabel}
      </button>

      <div className="mt-2 grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => handleOutcome(EpisodeOutcome.SUCCESS)}
          disabled={!canLabel}
          className="flex h-10 items-center justify-center gap-1.5 rounded-md bg-emerald-500 text-sm font-semibold text-white transition-colors hover:bg-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-300 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500"
        >
          <MdCheckCircle size={18} />
          Success
        </button>
        <button
          type="button"
          onClick={() => handleOutcome(EpisodeOutcome.FAILURE)}
          disabled={!canLabel}
          className="flex h-10 items-center justify-center gap-1.5 rounded-md bg-rose-500 text-sm font-semibold text-white transition-colors hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500"
        >
          <MdHighlightOff size={18} />
          Fail
        </button>
      </div>

      <p className="mt-2 text-xs leading-snug text-gray-500" aria-live="polite">
        {statusMessage}
      </p>
    </section>
  );
}
