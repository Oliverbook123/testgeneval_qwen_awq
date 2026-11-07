# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/run_docker.py

import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import dotenv
from swebench_docker.constants import KEY_MODEL, MAP_VERSION_TO_INSTALL
from swebench_docker.evaluation_status import write_evaluation_status

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


async def _stop_evaluation_container(container_name, cidfile, process):
    """Stop only the container owned by this evaluation and reap its client."""
    identifier = container_name
    try:
        container_id = Path(cidfile).read_text().strip()
        if container_id:
            identifier = container_id
    except OSError:
        pass

    stopper = None
    try:
        stopper = await asyncio.create_subprocess_exec(
            "docker", "stop", "--time", "5", identifier,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(stopper.communicate(), timeout=12)
        if stopper.returncode and b"No such container" not in (output or b""):
            logger.warning("Could not stop evaluation container %s: %s",
                           identifier, (output or b"").decode(errors="replace"))
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("Could not stop evaluation container %s: %s", identifier, exc)
    finally:
        if stopper is not None and stopper.returncode is None:
            try:
                stopper.kill()
            except ProcessLookupError:
                pass
            await stopper.communicate()
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()


async def run_docker_evaluation(
    task_instance: dict,
    namespace: str,
    log_dir: str,
    setting: str,
    ind: int,
    timeout: int = 60,
    verbose: bool = False,
    base64_instance: bool = True,
    only_baseline: bool = False,
    skip_mutation: bool = False,
) -> bool:
    repo_name = task_instance["repo"].replace("/", "_")
    task_id = task_instance["id"]
    model = "baseline" if only_baseline else task_instance[KEY_MODEL]

    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]
    image_prefix = "swe-bench"

    # TODO: Change this when deciding
    if "packages" in specifications and specifications["packages"] == "environment.yml":
        container_log_dir = "/home/swe-bench/logs"
    else:
        container_log_dir = "/opt/logs"

    if specifications.get("instance_image", False):
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
        )

    swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")

    temporary_directory = tempfile.TemporaryDirectory(prefix="testgeneval-eval-")
    cidfile = os.path.join(temporary_directory.name, "container.cid")
    container_name = "testgeneval-eval-" + uuid.uuid4().hex
    container_options = ["--name", container_name, "--cidfile", cidfile]

    if swebench_docker_fork_dir:
        # Create a temporary file to store the task_instance JSON
        tmpfile_path = os.path.join(temporary_directory.name, "task_instance.json")
        with open(tmpfile_path, "w", encoding="utf-8") as f:
            json.dump(task_instance, f)

        docker_command = [
            "docker",
            "run",
            "--rm",
            *container_options,
            "--network",
            "host",
            "-v",
            f"{log_dir}:{container_log_dir}",
            # Map the swebench_docker fork dir to the container
            # for some reason, swebench_docker has different locations for the different containers :(
            # so we need to map all of them to make it work
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/opt/swebench_docker:ro",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench_docker:ro",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench:ro",
            # =======
            # Map file instead pass the instance as env var to avoid "Argument list too long" error
            "-v",
            f"{tmpfile_path}:/home/swe-bench/task_instance.json:ro",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"IND={ind}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            docker_image,
        ]
    else:
        # Base64 encode the instance JSON to be sure it can be passed as an environment variable
        instance_b64 = base64.b64encode(
            json.dumps(task_instance).encode("utf-8")
        ).decode("utf-8")
        docker_command = [
            "docker",
            "run",
            "--rm",
            *container_options,
            "--network",
            "host",
            "-v",
            f"{log_dir}:{container_log_dir}",
            "-e",
            f"INSTANCE={instance_b64}",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"IND={ind}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            docker_image,
        ]

    cmd_string = shlex.join(docker_command)

    if verbose:
        logger.info(cmd_string)

    start_time = time.monotonic()
    process = None
    context = f"[{task_id}][{docker_image}][setting={setting}]"

    try:
        write_evaluation_status(log_dir, task_id, model, setting, "running",
                                container_name=container_name)
        logger.info("%s Starting evaluation container.", context)
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *docker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        )
        try:
            # Keep the subprocess handle even if cancellation arrives while the
            # OS process is being created, so cleanup can still reap it.
            process = await asyncio.shield(launch)
        except asyncio.CancelledError:
            process = await launch
            raise
        stdout, stderr = await process.communicate()
        # Decode stdout and stderr from bytes to str
        str_stdout = stdout.decode(errors="replace") if stdout else ""
        str_stderr = stderr.decode(errors="replace") if stderr else ""

        elapsed_time = time.monotonic() - start_time
        completed = process.returncode == 0 and "Evaluation succeeded" in str_stdout
        write_evaluation_status(
            log_dir, task_id, model, setting,
            "completed" if completed else "failed",
            returncode=process.returncode, elapsed_seconds=elapsed_time,
            container_name=container_name,
        )

        if process.returncode != 0:
            logger.warning(
                f"{context} Error running container:"
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"Stdout - {str_stdout}")
            logger.warning(f"Stderr - {str_stderr}")

        elif "Evaluation succeeded" not in str_stdout:
            logger.warning(
                f"{context} Container ran successfully in {elapsed_time} seconds, but evaluation failed."
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"stdout - {str_stdout}")
        else:
            logger.info(
                f"{context} Container ran successfully in {elapsed_time} seconds."
            )
        return completed
    except asyncio.CancelledError:
        await _stop_evaluation_container(container_name, cidfile, process)
        write_evaluation_status(log_dir, task_id, model, setting, "interrupted",
                                container_name=container_name)
        logger.info("%s Evaluation interrupted; it will be retried on resume.", context)
        raise
    except Exception as e:
        if process is not None:
            await _stop_evaluation_container(container_name, cidfile, process)
        write_evaluation_status(log_dir, task_id, model, setting, "failed",
                                container_name=container_name, error=str(e))
        logger.warning(
            f"{context} Error running container: {e}"
        )
        return False
    finally:
        temporary_directory.cleanup()
