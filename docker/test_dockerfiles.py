from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_main_dockerfiles_install_compose_v2():
    for dockerfile in (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
    ):
        contents = dockerfile.read_text()

        assert "docker-compose-v2" in contents, (
            f"{dockerfile.name} must install docker-compose-v2 so "
            "supervisor_api can recreate policy containers with docker compose"
        )


def test_main_compose_mounts_shared_s6_runner():
    contents = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    assert (
        "./s6-services/common/ros2_service_run.sh:"
        "/usr/local/lib/s6-services/ros2_service_run.sh:ro"
    ) in contents


def test_interactive_bash_fallbacks_include_ros_zenoh_router_defaults():
    dockerfiles = (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.amd64",
    )
    required = (
        "export ROS_DOMAIN_ID=\\${ROS_DOMAIN_ID:-30}",
        "export RMW_IMPLEMENTATION=\\${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}",
        "export ZENOH_ROUTER_IP=\\${ZENOH_ROUTER_IP:-127.0.0.1}",
        "export ZENOH_ROUTER_PORT=\\${ZENOH_ROUTER_PORT:-7447}",
        "export ZENOH_CONFIG_OVERRIDE=\\${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=true}",
    )

    for dockerfile in dockerfiles:
        contents = dockerfile.read_text()
        for expected in required:
            assert expected in contents, f"{dockerfile} is missing {expected}"
