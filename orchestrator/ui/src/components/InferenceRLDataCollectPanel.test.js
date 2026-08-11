import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import InferenceRLDataCollectPanel from './InferenceRLDataCollectPanel';
import { EpisodeOutcome } from '../constants/taskCommand';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import taskReducer, { setRecordStatus } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return {
    __esModule: true,
    default: toast,
  };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

const renderPanel = ({
  inferencePhase = InferencePhase.READY,
  recordPhase = RecordPhase.READY,
  taskType = '',
  recordInferenceMode = false,
  sendRecordCommand,
} = {}) => {
  const sendCommand = sendRecordCommand || jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: sendCommand });

  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType,
          recordInferenceMode,
          recordPhase,
          running: recordPhase !== RecordPhase.READY,
          topicReceived: true,
        },
      },
    },
  });

  const view = render(
    <Provider store={store}>
      <InferenceRLDataCollectPanel />
    </Provider>
  );

  return {
    store,
    sendRecordCommand: sendCommand,
    unmount: view.unmount,
  };
};

describe('InferenceRLDataCollectPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('keeps all collection controls disabled before inference starts', () => {
    renderPanel();

    expect(screen.getByRole('button', {
      name: /start inference recording/i,
    })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
  });

  test('starts one recording only after inference reaches INFERENCING', async () => {
    let resolveStart;
    const sendRecordCommand = jest.fn(() => new Promise((resolve) => {
      resolveStart = resolve;
    }));
    renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      sendRecordCommand,
    });

    const recordButton = screen.getByRole('button', {
      name: /start inference recording/i,
    });
    expect(recordButton).toBeEnabled();

    fireEvent.click(recordButton);
    fireEvent.click(recordButton);

    expect(sendRecordCommand).toHaveBeenCalledTimes(1);
    expect(sendRecordCommand).toHaveBeenCalledWith('start_inference_record');
    expect(recordButton).toBeDisabled();

    await act(async () => {
      resolveStart({ success: true, message: 'started' });
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Fail' })).toBeEnabled();
    });
    expect(toast.success).toHaveBeenCalledWith(
      'Inference recording started'
    );
  });

  test('saves a running inference episode with the Success outcome', async () => {
    const { sendRecordCommand } = renderPanel({
      inferencePhase: InferencePhase.PAUSED,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
    });

    const successButton = screen.getByRole('button', { name: 'Success' });
    await waitFor(() => expect(successButton).toBeEnabled());
    fireEvent.click(successButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        { episodeOutcome: EpisodeOutcome.SUCCESS }
      );
    });
    expect(toast.success).toHaveBeenCalledWith('Episode saved as Success');
    expect(successButton).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
  });

  test('saves a running inference episode with the Failure outcome', async () => {
    const { sendRecordCommand } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
    });

    const failButton = screen.getByRole('button', { name: 'Fail' });
    await waitFor(() => expect(failButton).toBeEnabled());
    fireEvent.click(failButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        { episodeOutcome: EpisodeOutcome.FAILURE }
      );
    });
    expect(toast.success).toHaveBeenCalledWith('Episode saved as Fail');
  });

  test('keeps outcome buttons retryable when a labeled save fails', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: false,
      message: 'archive busy',
    });
    renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
      sendRecordCommand,
    });

    const successButton = screen.getByRole('button', { name: 'Success' });
    await waitFor(() => expect(successButton).toBeEnabled());
    fireEvent.click(successButton);

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('archive busy');
      expect(successButton).toBeEnabled();
    });
  });

  test('does not label or start over an unrelated Record-page session', () => {
    const { sendRecordCommand } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'record',
    });

    expect(screen.getByRole('button', {
      name: /start inference recording/i,
    })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('returns to ready controls after the server completes saving', async () => {
    const { store } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled();
    });

    await act(async () => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordPhase: RecordPhase.SAVING,
        running: true,
      }));
      await Promise.resolve();
    });
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();

    await act(async () => {
      store.dispatch(setRecordStatus({
        taskType: '',
        recordInferenceMode: false,
        recordPhase: RecordPhase.READY,
        running: false,
      }));
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByRole('button', {
        name: /start inference recording/i,
      })).toBeEnabled();
    });
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
  });

  test('ignores a delayed START success after the server already returned READY', async () => {
    let resolveStart;
    const sendRecordCommand = jest.fn(() => new Promise((resolve) => {
      resolveStart = resolve;
    }));
    const { store } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /start inference recording/i,
    }));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledTimes(1));

    await act(async () => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordPhase: RecordPhase.RECORDING,
        running: true,
      }));
      store.dispatch(setRecordStatus({
        taskType: '',
        recordInferenceMode: false,
        recordPhase: RecordPhase.READY,
        running: false,
      }));
      await Promise.resolve();
    });

    await act(async () => {
      resolveStart({ success: true, message: 'late success' });
      await Promise.resolve();
    });

    expect(screen.getByRole('button', {
      name: /start inference recording/i,
    })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(toast.success).not.toHaveBeenCalledWith(
      'Inference recording started'
    );
  });

  test('ignores a delayed STOP failure after READY already closed the episode', async () => {
    let rejectStop;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectStop = reject;
    }));
    const { store } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
      sendRecordCommand,
    });

    const failButton = screen.getByRole('button', { name: 'Fail' });
    await waitFor(() => expect(failButton).toBeEnabled());
    fireEvent.click(failButton);
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledTimes(1));

    await act(async () => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordPhase: RecordPhase.SAVING,
        running: true,
      }));
      store.dispatch(setRecordStatus({
        taskType: '',
        recordInferenceMode: false,
        recordPhase: RecordPhase.READY,
        running: false,
      }));
      rejectStop(new Error('late transport failure'));
      await Promise.resolve();
    });

    expect(screen.getByRole('button', {
      name: /start inference recording/i,
    })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
    expect(toast.error).not.toHaveBeenCalledWith('late transport failure');
  });

  test('keeps a server-closed episode idle after the panel unmounts', async () => {
    let resolveStart;
    const sendRecordCommand = jest.fn(() => new Promise((resolve) => {
      resolveStart = resolve;
    }));
    const { store, unmount } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /start inference recording/i,
    }));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledTimes(1));
    unmount();

    await act(async () => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordPhase: RecordPhase.RECORDING,
        running: true,
      }));
      store.dispatch(setRecordStatus({
        taskType: '',
        recordInferenceMode: false,
        recordPhase: RecordPhase.READY,
        running: false,
      }));
      resolveStart({ success: true, message: 'late success' });
      await Promise.resolve();
    });

    expect(store.getState().tasks.inferenceRecordingUi.phase).toBe('idle');
    expect(toast.success).not.toHaveBeenCalledWith(
      'Inference recording started'
    );
  });

  test('restores IDLE when START fails after the panel unmounts', async () => {
    let rejectStart;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectStart = reject;
    }));
    const { store, unmount } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /start inference recording/i,
    }));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledTimes(1));
    expect(store.getState().tasks.inferenceRecordingUi.phase).toBe('starting');
    unmount();

    await act(async () => {
      rejectStart(new Error('start transport failure'));
      await Promise.resolve();
    });

    expect(store.getState().tasks.inferenceRecordingUi.phase).toBe('idle');
    expect(toast.error).toHaveBeenCalledWith('start transport failure');
  });

  test('restores ACTIVE when labeled STOP fails after the panel unmounts', async () => {
    let rejectStop;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectStop = reject;
    }));
    const { store, unmount } = renderPanel({
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
      sendRecordCommand,
    });

    const failButton = screen.getByRole('button', { name: 'Fail' });
    await waitFor(() => expect(failButton).toBeEnabled());
    fireEvent.click(failButton);
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledTimes(1));
    expect(store.getState().tasks.inferenceRecordingUi.phase).toBe('stopping');
    unmount();

    await act(async () => {
      rejectStop(new Error('stop transport failure'));
      await Promise.resolve();
    });

    expect(store.getState().tasks.inferenceRecordingUi.phase).toBe('active');
    expect(toast.error).toHaveBeenCalledWith('stop transport failure');
  });
});
