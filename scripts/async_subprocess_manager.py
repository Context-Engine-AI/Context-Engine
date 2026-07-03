#!/usr/bin/env python3
"""
Async subprocess manager for Context-Engine.

This module provides async subprocess management with proper resource cleanup
and non-blocking execution patterns.
"""

import os
import asyncio
import logging
import signal
import threading
import weakref
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import subprocess as _subprocess

logger = logging.getLogger("async_subprocess_manager")


class AsyncSubprocessManager:
    """
    Async subprocess manager with proper resource cleanup and non-blocking execution.
    
    Features:
    - Async subprocess execution with proper resource cleanup
    - Context manager support for automatic cleanup
    - Process tracking and statistics
    - Timeout handling and graceful cancellation
    - Thread pool for CPU-bound operations
    """
    
    def __init__(self, timeout: float = 30.0, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._process_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = {
            'started': 0,
            'completed': 0,
            'failed': 0,
            'timeout': 0,
            'cancelled': 0,
            'active_count': 0
        }
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.debug(f"Initialized AsyncSubprocessManager with timeout={timeout}s, max_workers={max_workers}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful process cleanup."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, cleaning up processes...")
            asyncio.create_task(self.cleanup_all_processes())
        
        # Register signal handlers
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except (ValueError, OSError) as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    async def run_async(
        self,
        cmd: Union[List[str], str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        input_data: Optional[Union[str, bytes]] = None,
        capture_output: bool = True,
        shell: bool = False
    ) -> Dict[str, Any]:
        """
        Run subprocess asynchronously with proper resource cleanup.

        Args:
            cmd: Command to execute (list for exec, string or list for shell)
            env: Environment variables
            cwd: Working directory
            input_data: Input data for subprocess
            capture_output: Whether to capture stdout/stderr
            shell: Whether to use shell

        Returns:
            Dictionary with execution results
        """
        import shlex
        process_id = f"proc_{len(self._active_processes)}_{id(cmd)}"

        try:
            # Choose correct creation API: shell vs exec
            if shell:
                # Ensure command is a string for shell
                if isinstance(cmd, list):
                    cmd_str = ' '.join(shlex.quote(str(x)) for x in cmd)
                else:
                    cmd_str = str(cmd)
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    env=env or os.environ,
                    cwd=cwd,
                    stdin=asyncio.subprocess.PIPE if input_data else None,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                )
            else:
                # Ensure command is a list for exec
                if isinstance(cmd, str):
                    exec_args = shlex.split(cmd)
                else:
                    exec_args = cmd
                process = await asyncio.create_subprocess_exec(
                    *exec_args,
                    env=env or os.environ,
                    cwd=cwd,
                    stdin=asyncio.subprocess.PIPE if input_data else None,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                )

            # Track process
            async with self._process_lock:
                self._active_processes[process_id] = process
                self._stats['started'] += 1
                self._stats['active_count'] = len(self._active_processes)

            logger.debug(f"Started async subprocess {process_id}: {cmd}")

            # Handle input data if provided
            if input_data and process.stdin:
                if isinstance(input_data, str):
                    input_data = input_data.encode('utf-8')
                process.stdin.write(input_data)
                await process.stdin.drain()
                process.stdin.close()

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )

                # Update statistics
                async with self._process_lock:
                    if process_id in self._active_processes:
                        del self._active_processes[process_id]
                        self._stats['completed'] += 1
                        self._stats['active_count'] = len(self._active_processes)

                return {
                    "ok": process.returncode == 0,
                    "code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='ignore') if stdout else "",
                    "stderr": stderr.decode('utf-8', errors='ignore') if stderr else "",
                    "process_id": process_id,
                }

            except asyncio.TimeoutError:
                logger.warning(f"Async subprocess {process_id} timed out after {self.timeout}s")

                # Terminate the process
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process_id} did not terminate, killing")
                    process.kill()
                    await process.wait()

                # Update statistics
                async with self._process_lock:
                    if process_id in self._active_processes:
                        del self._active_processes[process_id]
                        self._stats['timeout'] += 1
                        self._stats['active_count'] = len(self._active_processes)

                return {
                    "ok": False,
                    "code": -1,
                    "stdout": "",
                    "stderr": f"Process timed out after {self.timeout}s",
                    "process_id": process_id,
                    "timeout": True,
                }

        except Exception as e:
            logger.error(f"Error running async subprocess {process_id}: {e}")

            # Clean up on error
            async with self._process_lock:
                if process_id in self._active_processes:
                    del self._active_processes[process_id]
                    self._stats['failed'] += 1
                    self._stats['active_count'] = len(self._active_processes)

            return {
                "ok": False,
                "code": -1,
                "stdout": "",
                "stderr": str(e),
                "process_id": process_id,
                "error": str(e),
            }

    async def run_sync_in_executor(
        self,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        input_data: Optional[Union[str, bytes]] = None,
        shell: bool = False,
    ) -> Dict[str, Any]:
        """
        Run synchronous subprocess in thread pool for CPU-bound operations.
        
        This is useful for operations that don't benefit from true async I/O
        but need to run without blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run in thread pool
            result = await loop.run_in_executor(
                self._executor,
                self._run_sync_subprocess,
                cmd, env, cwd, input_data, shell
            )
            
            self._stats['completed'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error in executor subprocess: {e}")
            self._stats['failed'] += 1
            return {
                "ok": False,
                "code": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e),
            }
    
    def _run_sync_subprocess(
        self,
        cmd: Union[List[str], str],
        env: Optional[Dict[str, str]],
        cwd: Optional[str],
        input_data: Optional[Union[str, bytes]],
        shell: bool,
    ) -> Dict[str, Any]:
        """Internal method to run synchronous subprocess."""
        try:
            import shlex
            # Normalize command based on shell flag
            if shell:
                if isinstance(cmd, list):
                    cmd_str = ' '.join(shlex.quote(str(x)) for x in cmd)
                else:
                    cmd_str = str(cmd)
                popen_args = cmd_str
            else:
                exec_args = shlex.split(cmd) if isinstance(cmd, str) else cmd
                popen_args = exec_args

            process = _subprocess.Popen(
                popen_args,
                env=env or os.environ,
                cwd=cwd,
                stdin=_subprocess.PIPE if input_data else None,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
                shell=shell,
                text=True,
            )

            # Handle input data
            if input_data and process.stdin:
                if isinstance(input_data, bytes):
                    input_str = input_data.decode('utf-8', errors='ignore')
                else:
                    input_str = input_data
                process.stdin.write(input_str)
                process.stdin.close()

            # Wait for completion
            stdout, stderr = process.communicate()
            
            return {
                "ok": process.returncode == 0,
                "code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            
        except Exception as e:
            logger.error(f"Error in sync subprocess: {e}")
            return {
                "ok": False,
                "code": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e),
            }
    
    async def cancel_process(self, process_id: str) -> bool:
        """
        Cancel a running subprocess by ID.
        
        Args:
            process_id: ID of process to cancel
            
        Returns:
            True if process was cancelled, False if not found
        """
        async with self._process_lock:
            if process_id not in self._active_processes:
                return False
            
            process = self._active_processes[process_id]
            
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"Cancelled subprocess {process_id}")
                
                # Update statistics
                self._stats['cancelled'] += 1
                
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Process {process_id} did not terminate, killing")
                process.kill()
                await process.wait()
                
                self._stats['cancelled'] += 1
                return True
                
            except Exception as e:
                logger.error(f"Error cancelling subprocess {process_id}: {e}")
                return False
            
            finally:
                if process_id in self._active_processes:
                    del self._active_processes[process_id]
                    self._stats['active_count'] = len(self._active_processes)
    
    async def cleanup_all_processes(self) -> None:
        """Clean up all active subprocesses."""
        process_ids = list(self._active_processes.keys())
        
        for process_id in process_ids:
            await self.cancel_process(process_id)
        
        logger.info(f"Cleaned up {len(process_ids)} subprocesses")
    
    async def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a running subprocess.
        
        Args:
            process_id: ID of process to check
            
        Returns:
            Status information or None if not found
        """
        async with self._process_lock:
            if process_id not in self._active_processes:
                return None
            
            process = self._active_processes[process_id]
            
            try:
                returncode = process.returncode
                
                return {
                    "process_id": process_id,
                    "running": returncode is None,
                    "returncode": returncode,
                    "pid": process.pid,
                }
                
            except Exception as e:
                logger.error(f"Error getting process status: {e}")
                return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get subprocess execution statistics."""
        return {
            **self._stats,
            "active_processes": len(self._active_processes),
            "max_workers": self.max_workers,
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup_all_processes()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            # Clean up remaining processes
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                loop.create_task(self.cleanup_all_processes())
        except Exception:
            pass


# Global async subprocess manager instance
_async_manager: Optional[AsyncSubprocessManager] = None
_manager_lock = threading.Lock()


async def get_async_subprocess_manager(timeout: float = 30.0, max_workers: int = 10) -> AsyncSubprocessManager:
    """Get or create the global async subprocess manager."""
    global _async_manager

    def _create_manager():
        global _async_manager
        if _async_manager is None:
            _async_manager = AsyncSubprocessManager(
                timeout=timeout,
                max_workers=max_workers
            )
            logger.debug("Created global async subprocess manager")
        return _async_manager

    # Use regular lock (threading.Lock supports context manager)
    with _manager_lock:
        return _create_manager()


async def run_subprocess_async(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    input_data: Optional[Union[str, bytes]] = None,
    timeout: Optional[float] = None,
    shell: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run subprocess asynchronously.
    
    Args:
        cmd: Command to execute
        env: Environment variables
        cwd: Working directory
        input_data: Input data for subprocess
        timeout: Custom timeout (overrides manager default)
        shell: Whether to use shell
        
    Returns:
        Dictionary with execution results
    """
    manager = await get_async_subprocess_manager()
    
    if timeout is not None:
        # Create a temporary manager with custom timeout
        temp_manager = AsyncSubprocessManager(timeout=timeout)
        return await temp_manager.run_async(
            cmd, env=env, cwd=cwd, input_data=input_data, shell=shell
        )
    else:
        return await manager.run_async(
            cmd, env=env, cwd=cwd, input_data=input_data, shell=shell
        )


async def run_subprocess_sync_in_executor(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    input_data: Optional[Union[str, bytes]] = None,
    shell: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run synchronous subprocess in thread pool.
    
    This is useful for CPU-bound operations that don't benefit from true async I/O.
    """
    manager = await get_async_subprocess_manager()
    return await manager.run_sync_in_executor(
        cmd, env=env, cwd=cwd, input_data=input_data, shell=shell
    )


async def cancel_subprocess(process_id: str) -> bool:
    """
    Convenience function to cancel a running subprocess.
    
    Args:
        process_id: ID of process to cancel
        
    Returns:
        True if process was cancelled, False if not found
    """
    manager = await get_async_subprocess_manager()
    return await manager.cancel_process(process_id)


async def get_subprocess_status(process_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get status of a running subprocess.
    
    Args:
        process_id: ID of process to check
        
    Returns:
        Status information or None if not found
    """
    manager = await get_async_subprocess_manager()
    return await manager.get_process_status(process_id)


async def get_subprocess_stats() -> Dict[str, Any]:
    """
    Convenience function to get subprocess execution statistics.
    
    Returns:
        Statistics dictionary
    """
    manager = await get_async_subprocess_manager()
    return manager.get_stats()


# Context manager for automatic cleanup
class AsyncSubprocessContext:
    """Context manager for async subprocess operations."""
    
    def __init__(self, timeout: Optional[float] = None, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self._manager = None
    
    async def __aenter__(self):
        """Context manager entry."""
        self._manager = await get_async_subprocess_manager(
            timeout=self.timeout or 30.0,
            max_workers=self.max_workers
        )
        return self._manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._manager:
            await self._manager.cleanup_all_processes()