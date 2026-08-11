import {
  getEpisodeOutcomeForCommand,
  getRecordCommandServiceTimeoutMs,
  normalizeEpisodeOutcome,
  shouldAutofillEmptyTaskFields,
  transformReplayDataResult,
} from './useRosServiceCaller';
import { EpisodeOutcome } from '../constants/taskCommand';
import PageType from '../constants/pageType';

describe('getRecordCommandServiceTimeoutMs', () => {
  test('does not time out recording save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('finish_episode')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('stop_inference_record')).toBe(0);
  });

  test('keeps shorter defaults for non-save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('refresh_topics')).toBe(10000);
    expect(getRecordCommandServiceTimeoutMs('start_inference')).toBe(30000);
  });

  test('allows callers to override the service timeout', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment', {
      serviceTimeoutMs: 45000,
    })).toBe(45000);
  });
});

describe('normalizeEpisodeOutcome', () => {
  test('defaults legacy commands to unspecified', () => {
    expect(normalizeEpisodeOutcome()).toBe(EpisodeOutcome.UNSPECIFIED);
  });

  test('preserves success and failure labels', () => {
    expect(normalizeEpisodeOutcome(EpisodeOutcome.SUCCESS)).toBe(
      EpisodeOutcome.SUCCESS
    );
    expect(normalizeEpisodeOutcome(EpisodeOutcome.FAILURE)).toBe(
      EpisodeOutcome.FAILURE
    );
  });

  test('maps unsupported values to unspecified', () => {
    expect(normalizeEpisodeOutcome(99)).toBe(EpisodeOutcome.UNSPECIFIED);
  });
});

describe('getEpisodeOutcomeForCommand', () => {
  test('allows labels only on inference-record stop', () => {
    expect(getEpisodeOutcomeForCommand(
      'stop_inference_record',
      EpisodeOutcome.SUCCESS
    )).toBe(EpisodeOutcome.SUCCESS);
    expect(getEpisodeOutcomeForCommand(
      'stop_inference_record',
      EpisodeOutcome.FAILURE
    )).toBe(EpisodeOutcome.FAILURE);
  });

  test('forces all other commands to unspecified', () => {
    expect(getEpisodeOutcomeForCommand(
      'start_inference_record',
      EpisodeOutcome.SUCCESS
    )).toBe(EpisodeOutcome.UNSPECIFIED);
    expect(getEpisodeOutcomeForCommand(
      'cancel_inference_record',
      EpisodeOutcome.FAILURE
    )).toBe(EpisodeOutcome.UNSPECIFIED);
    expect(getEpisodeOutcomeForCommand(
      'stop',
      EpisodeOutcome.SUCCESS
    )).toBe(EpisodeOutcome.UNSPECIFIED);
  });
});

describe('shouldAutofillEmptyTaskFields', () => {
  test('preserves empty inference text for backend recording fallback', () => {
    expect(shouldAutofillEmptyTaskFields(PageType.INFERENCE)).toBe(false);
  });

  test('keeps legacy record-page autofill unless explicitly disabled', () => {
    expect(shouldAutofillEmptyTaskFields(PageType.RECORD)).toBe(true);
    expect(shouldAutofillEmptyTaskFields(PageType.RECORD, false)).toBe(false);
  });
});

describe('transformReplayDataResult', () => {
  test('preserves replay robot metadata for the 3D viewer', () => {
    const result = transformReplayDataResult(
      {
        success: true,
        robot_type: 'ffw_sh5_rev1',
        urdf_path: '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf',
        end_effector_links: ['tool0'],
      },
      '/workspace/rosbag2/sh5/0'
    );

    expect(result.robot_type).toBe('ffw_sh5_rev1');
    expect(result.urdf_path).toBe(
      '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf'
    );
    expect(result.end_effector_links).toEqual(['tool0']);
    expect(result.bag_path).toBe('/workspace/rosbag2/sh5/0');
  });
});
