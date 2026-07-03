#!/usr/bin/env python3
"""
Subprocess management utilities to prevent memory leaks.
Provides context managers for subprocess lifecycle management.
"""
import asyncio
import subprocess
import threading
from typing import Optional, Dict, Any, List, Union
import logging
import os
import contextlib

logger = logging.getLogger(__name__)

# Global registry to track active subprocess processes
_ACTIVE_PROCESSES: Dict[int, subprocess.Popen] = {}
_PROCESS_LOCK = threading.Lock()
_PROCESS_COUNTER = 0


class SubprocessManager:
    """Context manager for subprocess lifecycle management."""

    def __init__(self, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.env = env or {}
        self.process = None
        self._id = None

    def __enter__(self):
        global _PROCESS_COUNTER, _ACTIVE_PROCESSES, _PROCESS_LOCK

        # Create subprocess with proper resource management
        try:
            # For async operations, we'll store the process but not start it here
            return self
        except Exception as e:
            logger.error(f"Failed to initialize subprocess manager: {e}")
            raise

    async def __aenter__(self):
        """Support async context management."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure async cleanup on exit."""
        try:
            await self._cleanup()
        except Exception:
            pass

    async def run_async(self, cmd: List[str]) -> Dict[str, Any]:
        """Run subprocess asynchronously with proper cleanup."""
        global _PROCESS_COUNTER, _ACTIVE_PROCESSES, _PROCESS_LOCK

        with _PROCESS_LOCK:
            _PROCESS_COUNTER += 1
            self._id = _PROCESS_COUNTER

        try:
            # Create the subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )

            # Register the process
            _ACTIVE_PROCESSES[self._id] = self.process

            # Determine effective timeout without overwriting caller-supplied value
            if self.timeout is None:
                timeout_str = os.environ.get("SUBPROCESS_DEFAULT_TIMEOUT", "600")
                try:
                    eff_timeout = float(timeout_str)
                except ValueError:
                    eff_timeout = 600.0
            else:
                eff_timeout = self.timeout

            try:
                stdout, stderr = await asyncio.wait_for(
                    self.process.communicate(),
                    timeout=eff_timeout
                )
                return {
                    "ok": self.process.returncode == 0,
                    "code": self.process.returncode,
                    "stdout": stdout.decode("utf-8", errors="ignore") if stdout else "",
                    "stderr": stderr.decode("utf-8", errors="ignore") if stderr else "",
                }
            except asyncio.TimeoutError:
                logger.warning(f"Subprocess {self._id} timed out after {self.timeout}s, terminating")
                try:
                    self.process.kill()
                except Exception:
                    pass
                return {
                    "ok": False,
                    "code": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {self.timeout}s",
                }
            finally:
                # Cleanup
                await self._cleanup()

        except Exception as e:
            logger.error(f"Failed to run subprocess {self._id}: {e}")
            await self._cleanup()
            return {
                "ok": False,
                "code": -2,
                "stdout": "",
                "stderr": str(e),
            }

    def run_sync(self, cmd: List[str]) -> Dict[str, Any]:
        """Run subprocess synchronously with proper cleanup."""
        global _PROCESS_COUNTER, _ACTIVE_PROCESSES, _PROCESS_LOCK

        with _PROCESS_LOCK:
            _PROCESS_COUNTER += 1
            self._id = _PROCESS_COUNTER

        try:
            # Determine effective timeout without overwriting caller-supplied value
            if self.timeout is None:
                timeout_str = os.environ.get("SUBPROCESS_DEFAULT_TIMEOUT", "600")
                try:
                    eff_timeout = float(timeout_str)
                except ValueError:
                    eff_timeout = 600.0
            else:
                eff_timeout = self.timeout

            # Create the subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
            )

            # Register the process
            _ACTIVE_PROCESSES[self._id] = self.process

            try:
                stdout, stderr = self.process.communicate(timeout=eff_timeout)
                return {
                    "ok": self.process.returncode == 0,
                    "code": self.process.returncode,
                    "stdout": stdout.decode("utf-8", errors="ignore") if stdout else "",
                    "stderr": stderr.decode("utf-8", errors="ignore") if stderr else "",
                }
            except subprocess.TimeoutExpired:
                logger.warning(f"Subprocess {self._id} timed out after {eff_timeout}s, terminating")
                try:
                    self.process.kill()
                except Exception:
                    pass
                return {
                    "ok": False,
                    "code": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {self.timeout}s",
                }
            finally:
                # Cleanup
                self._cleanup()

        except Exception as e:
            logger.error(f"Failed to run subprocess {self._id}: {e}")
            self._cleanup()
            return {
                "ok": False,
                "code": -2,
                "stdout": "",
                "stderr": str(e),
            }

    async def _cleanup(self):
        """Clean up subprocess resources."""
        if self.process is not None:
            try:
                # Close pipes
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

                # Wait for process to terminate
                with contextlib.suppress(Exception):
                    await self.process.wait()

                # Remove from active processes
                global _ACTIVE_PROCESSES, _PROCESS_LOCK
                with _PROCESS_LOCK:
                    if self._id in _ACTIVE_PROCESSES:
                        del _ACTIVE_PROCESSES[self._id]
            except Exception as e:
                logger.error(f"Error during subprocess cleanup {self._id}: {e}")
            finally:
                self.process = None
                self._id = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        # For sync context manager usage
        if self.process is not None:
            try:
                self._cleanup_sync()
            except Exception as e:
                logger.error(f"Error during subprocess cleanup: {e}")

    def _cleanup_sync(self):
        """Synchronous cleanup for context manager exit."""
        if self.process is not None:
            try:
                # Close pipes
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

                # Terminate process if still running
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                        # Give it a moment to terminate gracefully
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # Force kill if it doesn't terminate
                            self.process.kill()
                    except Exception:
                        pass

                # Ensure process is reaped
                with contextlib.suppress(Exception):
                    self.process.wait()

                # Remove from active processes
                global _ACTIVE_PROCESSES, _PROCESS_LOCK
                with _PROCESS_LOCK:
                    if self._id in _ACTIVE_PROCESSES:
                        del _ACTIVE_PROCESSES[self._id]
            except Exception as e:
                logger.error(f"Error during subprocess sync cleanup: {e}")
            finally:
                self.process = None
                self._id = None


def get_active_processes() -> Dict[int, str]:
    """Get information about currently active subprocess processes."""
    global _ACTIVE_PROCESSES, _PROCESS_LOCK

    with _PROCESS_LOCK:
        info = {}
        for pid, proc in _ACTIVE_PROCESSES.items():
            try:
                args = ' '.join(proc.args) if hasattr(proc, 'args') else 'unknown'
            except Exception:
                args = 'unknown'
            try:
                running = (proc.poll() is None) if hasattr(proc, 'poll') else (getattr(proc, 'returncode', None) is None)
            except Exception:
                running = False
            status = 'running' if running else f"returncode: {getattr(proc, 'returncode', 'unknown')}"
            info[pid] = f"cmd: {args}, status: {status}"
        return info


def cleanup_all_processes():
    """Force cleanup of all active subprocess processes."""
    global _ACTIVE_PROCESSES, _PROCESS_LOCK

    with _PROCESS_LOCK:
        for pid, proc in list(_ACTIVE_PROCESSES.items()):
            try:
                if (hasattr(proc, 'poll') and proc.poll() is None) or (getattr(proc, 'returncode', None) is None):
                    proc.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        proc.kill()
                    except Exception:
                        pass

                # Ensure process is reaped
                with contextlib.suppress(Exception):
                    proc.wait()
            except Exception as e:
                logger.error(f"Error cleaning up process {pid}: {e}")

        _ACTIVE_PROCESSES.clear()


# Context manager helper functions
async def run_subprocess_async(cmd: List[str], timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Convenience function to run subprocess asynchronously with proper cleanup."""
    async with SubprocessManager(timeout=timeout, env=env) as manager:
        return await manager.run_async(cmd)


def run_subprocess_sync(cmd: List[str], timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Convenience function to run subprocess synchronously with proper cleanup."""
    with SubprocessManager(timeout=timeout, env=env) as manager:
        return manager.run_sync(cmd)


# Import os for timeout environment variable
import os
import contextlib