import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_container_script(tmp_path):
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    shutil.copy2(REPO_ROOT / "docker" / "container.sh", docker_dir / "container.sh")
    (docker_dir / "container.sh").chmod(0o755)
    (docker_dir / "config").mkdir()
    return docker_dir


def _write_start_stub(tmp_path):
    log_path = tmp_path / "docker.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.3.1-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.3-arm64\"}}}'\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)
    return log_path


def _write_enter_stub(tmp_path, running_container):
    log_path = tmp_path / "docker.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "if [ \"$1\" = ps ]; then\n"
        f"  printf '%s\\n' '{running_container}'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)
    return log_path


def _stub_env(tmp_path, log_path):
    return {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
        "CYCLO_STORAGE_MODE": "local",
    }


def test_start_creates_ros_zenoh_env_from_default(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    default_env = docker_dir / "config" / "ros_zenoh.default.env"
    default_env.write_text(
        "export ROS_DOMAIN_ID=30\n"
        "export RMW_IMPLEMENTATION=rmw_zenoh_cpp\n"
        "export ZENOH_ROUTER_IP=10.20.30.40\n"
        "export ZENOH_ROUTER_PORT=7447\n"
    )
    log_path = _write_start_stub(tmp_path)

    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    assert runtime_env.read_text() == default_env.read_text()
    assert "Created ROS/Zenoh runtime env from default" in result.stdout


def test_start_preserves_existing_ros_zenoh_env(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    (docker_dir / "config" / "ros_zenoh.default.env").write_text(
        "export ZENOH_ROUTER_IP=127.0.0.1\n"
    )
    runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    runtime_env.parent.mkdir(parents=True)
    runtime_env.write_text("export ZENOH_ROUTER_IP=192.168.60.139\n")
    log_path = _write_start_stub(tmp_path)

    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    assert runtime_env.read_text() == "export ZENOH_ROUTER_IP=192.168.60.139\n"
    assert "Created ROS/Zenoh runtime env from default" not in result.stdout


def test_enter_lerobot_creates_ros_zenoh_env_from_default(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    default_env = docker_dir / "config" / "ros_zenoh.default.env"
    default_env.write_text("export ZENOH_ROUTER_IP=127.0.0.1\n")
    log_path = _write_enter_stub(tmp_path, "lerobot_server")

    subprocess.run(
        [str(docker_dir / "container.sh"), "enter-lerobot"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    assert runtime_env.read_text() == default_env.read_text()
    assert any(
        "exec -it lerobot_server bash -lc" in call
        for call in log_path.read_text().splitlines()
    )


def test_start_pulls_main_image_only(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.3.1-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.3-arm64\"}}}'\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
        "CYCLO_STORAGE_MODE": "local",
        "CYCLO_LOCAL_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "CYCLO_LOCAL_HUGGINGFACE_DIR": str(tmp_path / "huggingface"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert any(
        "pull --ignore-pull-failures cyclo_intelligence" in call
        for call in docker_calls
    )
    assert not any("pull --ignore-pull-failures lerobot" in call for call in docker_calls)
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)


def test_start_does_not_remove_policy_container_with_stale_workspace_mount(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.3.1-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.3-arm64\"}}}'\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ] && [ \"$3\" = '{{.Image}}' ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ]; then\n"
        "  case \"$3\" in\n"
        "    *'.Destination \"/workspace\"'*)\n"
        "      if [ \"$4\" = lerobot_server ]; then\n"
        "        printf '%s\\n' '/old/workspace'\n"
        "      else\n"
        "        printf '%s\\n' \"$CYCLO_LOCAL_WORKSPACE_DIR\"\n"
        "      fi\n"
        "      exit 0\n"
        "      ;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
        "CYCLO_STORAGE_MODE": "local",
        "CYCLO_LOCAL_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "CYCLO_LOCAL_HUGGINGFACE_DIR": str(tmp_path / "huggingface"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert "rm -f lerobot_server" not in docker_calls
    assert "rm -f groot_server" not in docker_calls


def test_start_lerobot_removes_stale_workspace_mount(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.3.1-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.3-arm64\"}}}'\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ] && [ \"$3\" = '{{.Image}}' ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ]; then\n"
        "  case \"$3\" in\n"
        "    *'.Destination \"/workspace\"'*)\n"
        "      printf '%s\\n' '/old/workspace'\n"
        "      exit 0\n"
        "      ;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
        "CYCLO_STORAGE_MODE": "local",
        "CYCLO_LOCAL_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "CYCLO_LOCAL_HUGGINGFACE_DIR": str(tmp_path / "huggingface"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start-lerobot"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert any(
        "pull --ignore-pull-failures lerobot" in call
        for call in docker_calls
    )
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)
    assert any(
        call == "rm -f lerobot_server"
        for call in docker_calls
    )
