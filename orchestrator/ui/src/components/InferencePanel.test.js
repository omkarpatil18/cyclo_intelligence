import { configureStore } from '@reduxjs/toolkit';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import taskReducer from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('./FileBrowserModal', () => function MockFileBrowserModal() {
  return null;
});
jest.mock('./InferenceModelSelector', () => function MockModelSelector() {
  return <div data-testid="model-selector" />;
});
jest.mock('./InferenceRLDataCollectPanel', () => function MockRLPanel() {
  return <div data-testid="rl-data-collect" />;
});
jest.mock('./PolicyBackendControl', () => function MockBackendControl() {
  return <div data-testid="backend-control" />;
});
jest.mock('./TrtEngineControl', () => function MockTrtControl() {
  return <div data-testid="trt-control" />;
});
jest.mock('./Tooltip', () => function MockTooltip({ children }) {
  return <>{children}</>;
});

const renderPanel = ({ recordPhase = RecordPhase.READY, taskType = '' } = {}) => {
  const sendRecordCommand = jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });

  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase: InferencePhase.INFERENCING,
        },
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType,
          recordPhase,
          running: recordPhase !== RecordPhase.READY,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferencePanel />
    </Provider>
  );
  return { sendRecordCommand, store };
};

describe('InferencePanel RL collection safety', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('keeps deploy-target switching locked until recording is labeled', () => {
    const { sendRecordCommand } = renderPanel({
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
    });

    expect(screen.getByRole('button', {
      name: /use 3d sim deploy/i,
    })).toBeDisabled();
    expect(screen.getByRole('button', {
      name: /use real robot deploy/i,
    })).toBeDisabled();
    expect(sendRecordCommand).not.toHaveBeenCalledWith('finish');
  });
});
