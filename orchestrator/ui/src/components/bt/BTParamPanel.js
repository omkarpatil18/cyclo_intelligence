// Copyright 2026 ROBOTIS CO., LTD.
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
//
// Author: Seongwoo Kim

import React, { useState, useEffect, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { MdClose, MdFolderOpen } from 'react-icons/md';
import FileBrowserModal from '../FileBrowserModal';
import { setSelectedNodeId } from '../../features/btmanager/btmanagerSlice';
import { DEFAULT_PATHS } from '../../constants/paths';

const NUMBER_PARAMS = new Set([
  'duration', 'angle_deg', 'lift_position', 'control_hz', 'inference_hz',
  'chunk_align_window_s', 'max_iterations', 'joint_threshold',
  'gripper_closed_value', 'gripper_open_value', 'gripper_threshold',
  'timeout_sec',
]);

const SG2_TARGET_JOINTS = {
  left_target_joints: [
    'arm_l_joint1',
    'arm_l_joint2',
    'arm_l_joint3',
    'arm_l_joint4',
    'arm_l_joint5',
    'arm_l_joint6',
    'arm_l_joint7',
    'gripper_l_joint1',
  ],
  right_target_joints: [
    'arm_r_joint1',
    'arm_r_joint2',
    'arm_r_joint3',
    'arm_r_joint4',
    'arm_r_joint5',
    'arm_r_joint6',
    'arm_r_joint7',
    'gripper_r_joint1',
  ],
};

const TARGET_POSITION_PARAM = {
  left_target_joints: 'left_target_positions',
  right_target_joints: 'right_target_positions',
};

const csvParts = (value) =>
  String(value || '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

function isSg2LikeRobot(robotType) {
  const normalized = String(robotType || '').toLowerCase();
  return !normalized || normalized.includes('sg2') || normalized.includes('bg2');
}

function getTargetJointOptions(robotType, key) {
  if (!isSg2LikeRobot(robotType)) return [];
  return SG2_TARGET_JOINTS[key] || [];
}

// Per-param helper text shown beneath the input. Keep these short — they
// render directly under the field as a small gray hint.
const HELP_TEXT = {
  max_iterations: '0 = loop forever',
};

// Boolean params render as checkboxes.
const BOOL_PARAMS = new Set([
  'enable_head',
  'enable_arms',
  'enable_lift',
  'detect_left_gripper',
  'detect_right_gripper',
]);

// Enum params surface as <select> dropdowns. Keep value lists in sync with
// the Python action definitions (send_command.COMMAND_MAP).
const ENUM_PARAMS = {
  command: ['LOAD', 'RESUME', 'STOP', 'CLEAR'],
  model: [
    'lerobot:act',
    'lerobot:smolvla',
    'lerobot:xvla',
    'lerobot:pi0',
    'lerobot:pi05',
    'lerobot:diffusion',
    'groot:n17',
    'groot',
    'lerobot',
  ],
  inference_mode: ['simulation', 'robot'],
  action_request_mode: ['async', 'sync'],
  acceleration_mode: ['pytorch', 'tensorrt_dit'],
};

// SendCommand inputs that are meaningful per command. Anything outside
// the set for the current command is rendered disabled — the value stays
// in params so flipping back to LOAD restores the user's earlier entries.
// 'command' itself is always editable.
const SEND_COMMAND_ACTIVE_FIELDS = {
  LOAD: new Set([
    'command', 'model', 'policy_path', 'task_instruction',
    'inference_mode', 'action_request_mode', 'inference_hz', 'control_hz',
    'chunk_align_window_s', 'acceleration_mode', 'acceleration_engine_path',
  ]),
  // Resume can re-condition language mid-run; output mode is fixed by LOAD.
  RESUME: new Set(['command', 'task_instruction']),
  STOP: new Set(['command']),
  CLEAR: new Set(['command']),
};

// JointControl: each group's positions input is gated on its enable_*
// flag. enable_* toggles themselves + duration are always editable.
const truthy = (v) => v === true || v === 'true';

function isSendCommandFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'SendCommand') return false;
  const cmd = String(params.command || 'LOAD').toUpperCase();
  const active = SEND_COMMAND_ACTIVE_FIELDS[cmd];
  if (!active) return false;
  return !active.has(key);
}

function isJointControlFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'JointControl') return false;
  if (key === 'head_positions') return !truthy(params.enable_head);
  if (key === 'left_positions' || key === 'right_positions') {
    return !truthy(params.enable_arms);
  }
  if (key === 'lift_position') return !truthy(params.enable_lift);
  return false;  // enable_*, duration stay editable
}

function isArmStateGateFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'ArmStateGate') return false;
  if (key === 'left_gripper_joint') {
    return !truthy(params.detect_left_gripper);
  }
  if (key === 'right_gripper_joint') {
    return !truthy(params.detect_right_gripper);
  }
  return false;
}

function isFieldDisabled(nodeType, key, params) {
  return (
    isSendCommandFieldDisabled(nodeType, key, params) ||
    isJointControlFieldDisabled(nodeType, key, params) ||
    isArmStateGateFieldDisabled(nodeType, key, params)
  );
}

export default function BTParamPanel({ nodes, selectedNodeId, onParamChange, onNameChange }) {
  const dispatch = useDispatch();
  const robotType = useSelector((state) => state.tasks?.robotType || '');

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Local param state — isolates keystrokes from parent re-renders (preserves cursor)
  const [localParams, setLocalParams] = useState({});
  // Local name buffer — same cursor-preservation trick as localParams.
  const [localName, setLocalName] = useState('');
  const [showPolicyBrowser, setShowPolicyBrowser] = useState(false);

  const policyBrowserPath = useMemo(() => {
    const model = String(localParams.model || '').toLowerCase();
    return model.startsWith('groot')
      ? DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH
      : DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH;
  }, [localParams.model]);

  // Reset local state only when switching to a different node
  useEffect(() => {
    if (selectedNode) {
      setLocalParams(selectedNode.data.params || {});
      setLocalName(selectedNode.data.label || '');
    }
    setShowPolicyBrowser(false);
    // Keep mid-edit cursor position stable; reset only when the selection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedNodeId]); // intentionally excludes selectedNode to avoid resetting mid-edit

  if (!selectedNode) return null;

  const { label, nodeType } = selectedNode.data;
  const paramEntries = Object.entries(localParams);

  const commitName = () => {
    const trimmed = localName.trim();
    if (!trimmed) {
      // Reject empty — snap input back to current label.
      setLocalName(label);
      return;
    }
    if (trimmed !== label) {
      onNameChange?.(selectedNodeId, trimmed);
    }
  };

  const handleChange = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
  };

  const handleBlur = (paramName) => {
    onParamChange(selectedNodeId, paramName, localParams[paramName]);
  };

  const commitParam = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
    onParamChange(selectedNodeId, paramName, value);
  };

  const handlePolicyFolderSelect = (item) => {
    const fullPath = item?.full_path || '';
    if (fullPath) {
      commitParam('policy_path', fullPath);
    }
    setShowPolicyBrowser(false);
  };

  const commitTargetJointSelection = (paramName, nextJoints) => {
    const positionsParam = TARGET_POSITION_PARAM[paramName];
    const currentJoints = csvParts(localParams[paramName]);
    const currentPositions = csvParts(localParams[positionsParam]);
    const positionByJoint = new Map(
      currentJoints.map((joint, idx) => [
        joint,
        currentPositions[idx] || '0.0',
      ])
    );
    const nextPositions = nextJoints.map(
      (joint) => positionByJoint.get(joint) || '0.0'
    );
    const jointsValue = nextJoints.join(', ');
    const positionsValue = nextPositions.join(', ');

    setLocalParams((prev) => ({
      ...prev,
      [paramName]: jointsValue,
      [positionsParam]: positionsValue,
    }));
    onParamChange(selectedNodeId, paramName, jointsValue);
    onParamChange(selectedNodeId, positionsParam, positionsValue);
  };

  const renderTargetJointSelector = (key, value, disabled = false) => {
    const options = getTargetJointOptions(robotType, key);
    const selected = csvParts(value);
    const selectedSet = new Set(selected);
    const disabledCls = disabled
      ? ' bg-gray-100 text-gray-400 cursor-not-allowed'
      : '';

    if (options.length === 0) {
      return (
        <textarea
          value={value}
          disabled={disabled}
          onChange={(e) => handleChange(key, e.target.value)}
          onBlur={() => handleBlur(key)}
          rows={String(value).length > 60 ? 3 : 1}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y${disabledCls}`}
        />
      );
    }

    return (
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5">
          {options.map((jointName) => {
            const isSelected = selectedSet.has(jointName);
            return (
              <button
                key={jointName}
                type="button"
                disabled={disabled}
                onClick={() => {
                  const nextJoints = isSelected
                    ? selected.filter((joint) => joint !== jointName)
                    : options.filter((joint) => (
                        joint === jointName || selectedSet.has(joint)
                      ));
                  commitTargetJointSelection(key, nextJoints);
                }}
                className={`px-2 py-1 border rounded text-xs transition-colors ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {jointName}
              </button>
            );
          })}
        </div>
        <textarea
          value={value}
          disabled={disabled}
          onChange={(e) => handleChange(key, e.target.value)}
          onBlur={() => handleBlur(key)}
          rows={String(value).length > 60 ? 3 : 1}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y${disabledCls}`}
        />
      </div>
    );
  };

  const renderInput = (key, value, disabled = false) => {
    const disabledCls = disabled
      ? ' bg-gray-100 text-gray-400 cursor-not-allowed'
      : '';

    if (nodeType === 'ArmStateGate' && TARGET_POSITION_PARAM[key]) {
      return renderTargetJointSelector(key, value, disabled);
    }

    if (ENUM_PARAMS[key]) {
      return (
        <select
          value={value}
          disabled={disabled}
          onChange={(e) => {
            handleChange(key, e.target.value);
            // select has no meaningful blur event for this; sync immediately
            onParamChange(selectedNodeId, key, e.target.value);
          }}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white focus:outline-none focus:ring-1 focus:ring-blue-400${disabledCls}`}
        >
          {ENUM_PARAMS[key].map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
    }

    if (BOOL_PARAMS.has(key)) {
      return (
        <label className={`flex items-center gap-2 ${disabled ? 'cursor-not-allowed text-gray-400' : 'cursor-pointer'}`}>
          <input
            type="checkbox"
            disabled={disabled}
            checked={value === 'true' || value === true}
            onChange={(e) => {
              const v = e.target.checked ? 'true' : 'false';
              handleChange(key, v);
              onParamChange(selectedNodeId, key, v);
            }}
            className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-400"
          />
          <span className="text-sm text-gray-600">{value === 'true' || value === true ? 'true' : 'false'}</span>
        </label>
      );
    }

    if (nodeType === 'SendCommand' && key === 'policy_path') {
      return (
        <div className="flex flex-row items-start gap-2">
          <textarea
            value={value}
            disabled={disabled}
            onChange={(e) => handleChange(key, e.target.value)}
            onBlur={() => handleBlur(key)}
            rows={String(value).length > 60 ? 3 : 1}
            placeholder="Enter Policy Path or Repo ID"
            className={`flex-1 min-w-0 px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y${disabledCls}`}
          />
          <button
            type="button"
            onClick={() => !disabled && setShowPolicyBrowser(true)}
            disabled={disabled}
            className="flex items-center justify-center w-8 h-8 text-blue-500 bg-gray-100 border border-gray-300 rounded hover:text-blue-700 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            aria-label="Browse for policy model folder"
            title="Browse for policy model folder"
          >
            <MdFolderOpen size={18} />
          </button>
        </div>
      );
    }

    if (NUMBER_PARAMS.has(key)) {
      return (
        <input
          type="number"
          step="any"
          value={value}
          disabled={disabled}
          onChange={(e) => handleChange(key, e.target.value)}
          onBlur={() => handleBlur(key)}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400${disabledCls}`}
        />
      );
    }

    return (
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(key, e.target.value)}
        onBlur={() => handleBlur(key)}
        rows={String(value).length > 60 ? 3 : 1}
        className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y${disabledCls}`}
      />
    );
  };

  return (
    <div className="absolute right-0 top-0 bottom-0 w-[320px] bg-white border-l border-gray-200 shadow-lg z-10 flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between px-4 py-3 border-b border-gray-200">
        <div className="flex-1 min-w-0 pr-2">
          <div className="text-xs text-gray-500 mb-1">{nodeType}</div>
          <input
            type="text"
            value={localName}
            onChange={(e) => setLocalName(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.currentTarget.blur();
              } else if (e.key === 'Escape') {
                setLocalName(label);
                e.currentTarget.blur();
              }
            }}
            className="w-full text-sm font-bold text-gray-800 bg-transparent border-0 border-b border-transparent hover:border-gray-300 focus:border-blue-400 focus:outline-none px-0 py-0.5"
          />
        </div>
        <button
          onClick={() => dispatch(setSelectedNodeId(null))}
          className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <MdClose size={20} />
        </button>
      </div>

      {/* Params */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {paramEntries.length === 0 ? (
          <p className="text-sm text-gray-400">No parameters</p>
        ) : (
          paramEntries.map(([key, value]) => {
            const disabled = isFieldDisabled(nodeType, key, localParams);
            const help = HELP_TEXT[key];
            return (
              <div key={key}>
                <label
                  className={`block text-xs font-medium mb-1 ${
                    disabled ? 'text-gray-400' : 'text-gray-600'
                  }`}
                >
                  {key}
                </label>
                {renderInput(key, value, disabled)}
                {help && !disabled && (
                  <div className="mt-1 text-xs text-gray-500">{help}</div>
                )}
              </div>
            );
          })
        )}
      </div>
      <FileBrowserModal
        isOpen={showPolicyBrowser}
        onClose={() => setShowPolicyBrowser(false)}
        onFileSelect={handlePolicyFolderSelect}
        title="Select policy model folder"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={policyBrowserPath}
        defaultPath={policyBrowserPath}
        homePath={DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH}
      />
    </div>
  );
}
