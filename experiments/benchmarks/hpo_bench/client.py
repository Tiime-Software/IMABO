"""HPO Benchmark Client Module"""

import asyncio
import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import aiohttp

# Host-side port of the hpo-server container (see docker-compose.yml: mapped
# to 8901 because localhost:8000 is occupied by an unrelated host service).
URL = "http://localhost:8901"

# Global session variable
session: Optional[aiohttp.ClientSession] = None


async def start_container():
    """Start the HPO benchmark server container"""
    print("Building and starting HPO benchmark server container...")

    # Get root directory for docker commands
    root_dir = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(root_dir)

    try:
        # Build the container
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "compose",
            "build",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            print("Failed to build container")
            return False

        # Start the server container
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "compose",
            "up",
            "-d",
            "hpo-server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            print("Failed to start container")
            return False

        print("Container started successfully")
        return True
    finally:
        os.chdir(original_cwd)


async def wait_for_server(timeout=60):
    """Wait for the HTTP server to be ready"""
    print(f"Waiting for server at {URL}...")

    async with aiohttp.ClientSession() as temp_session:
        for _ in range(timeout // 2):
            try:
                # Check if server is responding (we'll use load endpoint)
                async with temp_session.post(f"{URL}/load", json={}) as resp:
                    if resp.status in [
                        200,
                        400,
                        500,
                    ]:  # Any response means server is up
                        print("Server is ready!")
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)

    print(f"Timeout waiting for server after {timeout} seconds")
    return False


def stop_container():
    """Stop the HPO benchmark server container"""
    print("Stopping container...")
    root_dir = Path(__file__).parent.parent.parent
    original_cwd = os.getcwd()
    os.chdir(root_dir)
    try:
        subprocess.run(["docker", "compose", "down"], capture_output=True)
        print("Container stopped")
    finally:
        os.chdir(original_cwd)


async def api_call(endpoint, data):
    global session
    if session is None:
        raise RuntimeError("Server not started. Call start_hpo_server() first.")

    # Set longer timeout for /load endpoint (package installation takes time)
    timeout = (
        aiohttp.ClientTimeout(total=600)
        if endpoint == "load"
        else aiohttp.ClientTimeout(total=60)
    )

    try:
        async with session.post(
            f"{URL}/{endpoint}", json=data or {}, timeout=timeout
        ) as resp:
            result = await resp.json()
            if resp.status == 200 and "error" not in result:
                return result
            error = result.get("error", result) if isinstance(result, dict) else result
            print(f"✗ API error on /{endpoint}: {error}")
            return None
    except Exception as e:
        print(f"✗ Exception calling /{endpoint}: {e}")
        print(f"  Config: {data.get('config') if data else 'N/A'}")
        return None


async def start_hpo_server():
    """
    Start the HPO benchmark server
    """
    global session

    atexit.register(stop_container)

    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, cleaning up...")
        stop_container()
        sys.exit(130)  # Standard exit code for Ctrl+C (128 + SIGINT(2))

    signal.signal(signal.SIGINT, signal_handler)

    if not await start_container():
        print("Failed to start container")
        return False

    if not await wait_for_server():
        print("Server failed to start")
        return False

    session = aiohttp.ClientSession()
    print("Server started successfully")
    return True


async def stop_hpo_server():
    """
    Stop the HPO benchmark server
    """
    global session
    if session:
        await session.close()
        session = None
    stop_container()
