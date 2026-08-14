import { fireEvent, render, screen } from '@testing-library/react';
import FileBrowserModal from './FileBrowserModal';

jest.mock('./FileBrowser', () => function MockFileBrowser({ onPathChange }) {
  return (
    <div>
      <button
        type="button"
        onClick={() => onPathChange(
          '/workspace/rosbag2/Task_existing_inference_MCAP'
        )}
      >
        Open valid folder
      </button>
      <button
        type="button"
        onClick={() => onPathChange(
          '/workspace/rosbag2/Task_existing_inference_MCAP/0'
        )}
      >
        Open episode folder
      </button>
    </div>
  );
});

const isInferenceFolder = (item) => (
  /^\/workspace\/rosbag2\/Task_.+_inference_MCAP\/?$/.test(
    item.full_path
  )
);

describe('FileBrowserModal directory filtering', () => {
  test('only allows a valid current directory to be confirmed', () => {
    const onFileSelect = jest.fn();
    render(
      <FileBrowserModal
        isOpen
        onClose={jest.fn()}
        onFileSelect={onFileSelect}
        initialPath="/workspace/rosbag2/"
        selectButtonText="Use Folder"
        allowDirectorySelect
        allowFileSelect={false}
        directoryFilter={isInferenceFolder}
      />
    );

    const confirm = screen.getByRole('button', { name: 'Use Folder' });
    expect(confirm).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Open valid folder' }));
    expect(confirm).toBeEnabled();

    fireEvent.click(screen.getByRole('button', { name: 'Open episode folder' }));
    expect(confirm).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Open valid folder' }));
    fireEvent.click(confirm);
    expect(onFileSelect).toHaveBeenCalledWith(expect.objectContaining({
      full_path: '/workspace/rosbag2/Task_existing_inference_MCAP',
    }));
  });
});
