"""
ProcessAndThreadHandlers.py
"""

import os
import threading
import time
import warnings
from inspect import signature
from typing import Dict, Tuple, Set, Optional, Callable

from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.Misc.Singleton import Singleton


class ProcessOrThreadHandler:
    """
    For internal use only. Use class ProcessHandler instead.
    """

    def __init__(
            self, is_process_handler: bool, process_name: Optional[int],
            acquire_data_generation_lock: Optional[Callable], release_data_generation_lock: Optional[Callable],
            register_thread: Optional[Callable]) -> None:
        """
        :param is_process_handler: True if this instance is used to handle processes, False if it is used to handle
        threads
        :param process_name: Must be None if is_process_handler, else the id of the process that this instance is
        assigned to
        :param acquire_data_generation_lock: Must be None if is_process_handler, else a reference to the
        acquire_data_generation_lock method of the ProcessOrThreadHandler that this instance is assigned to
        :param release_data_generation_lock: Must be None if is_process_handler, else a reference to the
        release_data_generation_lock method of the ProcessOrThreadHandler that this instance is assigned to
        :param register_thread: Must be None if is_process_handler, else a reference to the
        register_thread method of the ProcessOrThreadHandler that this instance is assigned to
        """
        self.__is_process_handler: bool = is_process_handler
        assert (self.__is_process_handler and process_name is None and acquire_data_generation_lock is None and
                release_data_generation_lock is None and register_thread is None) or \
               (not self.__is_process_handler and process_name is not None and acquire_data_generation_lock is not
                None and release_data_generation_lock is not None and register_thread is not None)
        cl: ConfigLoader = ConfigLoader.instance

        self.acquire_data_generation_lock = acquire_data_generation_lock
        self.release_data_generation_lock = release_data_generation_lock
        self.register_thread = register_thread

        self.__process_name = process_name
        self.__threads_to_borrow: Optional[Set[str]] = None
        self.__threads_borrowed: Optional[Set[str]] = None

        if self.__is_process_handler:
            self.__name = 'ProcessHandler'
        else:
            self.__name = f'Process: {process_name}: ThreadHandler'
        if cl.single_thread_mode:
            self.__available: int = 0
        else:
            if self.__is_process_handler:
                if cl.database_generation_in_progress:
                    self.__available: int = 1  # Multiple Processes must not write to the same files
                else:
                    self.__available: int = cl.n_cores
            else:
                self.__available: int = cl.number_of_threads_to_use
        if not self.__is_process_handler:
            self.__threads_to_borrow: Set[int] = set()
            self.__threads_borrowed: Set[int] = set()
        self.__transaction_lock: threading.Lock = threading.Lock()

    def register_idle(self, message: str) -> None:
        """
        This method allows a thread to announce itself as idle.
        If the parameter "overbook-threads" in the config file is true then an additional thread will be made available
        until the calling thread calls the method "unregister_idle"
        :param message: Message that is printed as additional information if debug output is enabled
        :return: None
        """
        assert not self.__is_process_handler, 'Only threads can register as idle'
        self.register_thread()
        cl: ConfigLoader = ConfigLoader.instance
        if not cl.overbook_threads:
            return
        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'Process: {self.__process_name}: Thread {threading.get_ident()}: '
                  f'I am registering as idle because {message}')
        self.__transaction_lock.acquire(blocking=True, timeout=-1)
        self.__threads_to_borrow.add(threading.get_ident())
        self.__transaction_lock.release()

    def unregister_idle(self) -> None:
        """
        This method allows a thread to announce itself as not being idle anymore.
        See description of method "register_idle" for more details.
        :return: None
        """
        assert not self.__is_process_handler, 'Only threads can unregister as idle'
        cl: ConfigLoader = ConfigLoader.instance
        if not cl.overbook_threads:
            return
        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'Process: {self.__process_name}: Thread {threading.get_ident()}: I am unregistering as idle')
        self.__transaction_lock.acquire(blocking=True, timeout=-1)
        self.__threads_to_borrow.remove(threading.get_ident())
        self.__transaction_lock.release()

    @property
    def ready(self) -> bool:
        """
        :return: If this instance is a ProcessHandler then it returns True if at least two cores are available, if it
        is a ThreadHandler it returns True if at least 2 threads are available. False is returned otherwise.
        """
        result = self.__available
        if not self.__is_process_handler:
            result += len(self.__threads_to_borrow)
        return result >= 2

    def __reserve(self, amount: int) -> None:
        """
        :param amount: The number of cores or threads to reserve
        :return: None
        """
        self.__available -= amount
        if self.__available < 0:
            warnings.warn(
                'Inefficient use of threads detected. Please only open a Pool if property \'ready\' is True.')

    def __free(self, amount: int) -> None:
        """
        :param amount: The number of cores or threads which to free up
        :return: None
        """
        self.__available += amount

    def exec_in_pool(self, function: Callable, maximum: int, args: Tuple = None, kwargs: Dict = None) -> any:
        """
        :param function: A callable function that profits of multithreading and has a keyword argument
        'thread_handler_automatic_parameter' which defaults to None and which name start with '__thread_handled_'
        :param maximum: Maximum number of cores or threads that would be profitable for that function
        :param args: The args that are passed to that function
        :param kwargs: The kwargs that are passed to that function
        :return: The result of the given function, executed with the given args and kwargs
        """
        args = args or tuple()
        kwargs = kwargs or dict()

        assert isinstance(maximum, int), 'Parameter \'max_threads\' must be an int'
        assert isinstance(args, tuple), 'Parameter \'args\' must be a tuple'
        assert isinstance(kwargs, dict), 'Parameter \'kwargs\' must be a dict'
        assert maximum >= 0, f'Invalid number of threads: {maximum}'
        assert function.__name__.startswith('__thread_handled_') or function.__name__.startswith('thread_handled_'), \
            'Function name needs to start with __thread_handled_'
        assert callable(function), 'Only callable functions may be passed'

        sig = signature(function)

        assert 'thread_handler_automatic_parameter' in sig.parameters.keys(), \
            'Function must have a parameter called \'thread_handler_automatic_parameter\'.'

        param = sig.parameters['thread_handler_automatic_parameter']
        assert param.KEYWORD_ONLY and param.default is not param.empty, \
            'The default of the parameter \'thread_handler_automatic_parameter\' must be None.'

        # Reserve only as many as needed. If not at least 2 cores are available, then multithreading would be slow.
        # Therefore do not reserve cores in that case.
        reserved: int = 0

        currently_borrowed: Set[int] = set()

        self.__transaction_lock.acquire(blocking=True, timeout=-1)

        if not self.__is_process_handler and self.ready and maximum > 1:
            threads_left = maximum
            to_borrow: Set[int] = self.__threads_to_borrow - self.__threads_borrowed
            for i in range(maximum):
                if len(to_borrow) == 0:
                    break
                current = to_borrow.pop()
                currently_borrowed.add(current)
                self.__threads_borrowed.add(current)
                threads_left -= 1
                if threads_left == 0:
                    break
            self.__reserve(reserved := min(threads_left, self.__available))
        self.__transaction_lock.release()

        cl: ConfigLoader = ConfigLoader.instance
        if cl.process_and_thread_handler_debug_output_enabled and reserved + len(currently_borrowed) > 0:
            print(f'{self.__name}: I am reserving {reserved + len(currently_borrowed)} '
                  f'{"processes" if self.__is_process_handler else "threads"} for method '
                  f'{function.__name__}')
        kwargs['thread_handler_automatic_parameter'] = reserved + len(currently_borrowed)
        result = function(*args, **kwargs)
        if cl.process_and_thread_handler_debug_output_enabled and reserved > 0:
            print(f'{self.__name}: I am freeing {reserved + len(currently_borrowed)} '
                  f'{"processes" if self.__is_process_handler else "threads"} after executing method '
                  f'{function.__name__}')
        self.__transaction_lock.acquire(blocking=True, timeout=-1)
        self.__free(reserved)
        for t in currently_borrowed:
            self.__threads_borrowed.remove(t)
        self.__transaction_lock.release()
        return result


@Singleton
class ProcessHandler:
    """
    Multithreading can slower performance if the number of threads exceeds the number of logical cores.
    This class solves the problem of running a method efficiently that runs in one or multiple processes,
    that itself executes a method running in one or multiple threads. It does this by assigning as many cores to a
    method as are meaningful and available. Otherwise 1 Thread is assigned.
    """

    def __init__(self):
        self.__process_handler = ProcessOrThreadHandler(
            is_process_handler=True, process_name=None, acquire_data_generation_lock=None,
            release_data_generation_lock=None, register_thread=None)
        self.exec_in_pool = self.__process_handler.exec_in_pool
        self.__data_generation_lock = threading.Lock()
        self.__data_generation_lock_owner: Optional[str] = None
        self.__process_thread_handler_dict: Dict[int, ProcessOrThreadHandler] = dict()
        self.__thread_process_dict: Dict[int, int] = dict()
        self.__thread_handler_user_number_dict: Dict[int, int] = dict()
        self.__register_thread_lock = threading.Lock()

    def acquire_data_generation_lock(self) -> bool:
        """
        In order to prevent multiple threads doing data generation at the same time, this lock is provided
        :return: True if another thread generated a database in the meanwhile. If this was the case then the calling
        method needs to check whether the generation of a database is still necessary. False otherwise.
        In any case the method release_data_generation_lock must be called to release the lock again so that other
        methods and threats can acquire it.

        usage:
        while True:
            other_generation_process_was_active: bool = th.acquire_data_generation_lock()
            if not other_generation_process_was_active:
                # Generate a dataframe
                break
        th.release_data_generation_lock()
        """
        if threading.get_ident() == self.__data_generation_lock_owner:
            return False
        if os.getpid() in self.__process_thread_handler_dict.keys():
            self.__process_thread_handler_dict[os.getpid()].register_idle('I am waiting for a dataframe generation to '
                                                                          'be finished.')
        had_to_wait: bool = not self.__data_generation_lock.acquire(blocking=False, timeout=-1)
        if had_to_wait:
            self.__data_generation_lock.acquire(blocking=True, timeout=-1)

        self.__data_generation_lock_owner = threading.get_ident()

        cl: ConfigLoader = ConfigLoader.instance

        if not had_to_wait:
            if cl.process_and_thread_handler_debug_output_enabled:
                print(f'ProcessHandler: I have assigned the data generation lock to thread {threading.get_ident()}')
                print('Waiting 3 seconds for other threads to call in as idle')
            time.sleep(3)
            if cl.process_and_thread_handler_debug_output_enabled:
                print('Waited for 3 seconds, continuing now')

        if os.getpid() in self.__process_thread_handler_dict.keys():
            self.__process_thread_handler_dict[os.getpid()].unregister_idle()

        return had_to_wait

    def release_data_generation_lock(self) -> None:
        """
        Releases the lock that is acquired by the method acquire_data_generation_lock
        :return: None
        """
        assert self.__data_generation_lock_owner == threading.get_ident(), 'Lock can only be released by the thread ' \
                                                                           'that acquired the lock.'
        cl: ConfigLoader = ConfigLoader.instance
        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'ProcessHandler: I am removing the data generation lock from thread {threading.get_ident()}')
        self.__data_generation_lock_owner = None
        self.__data_generation_lock.release()

    def register_thread(self) -> None:
        """
        Registers a thread to its process. This way the method get_name_and_register can associate the pair
        :return: None
        """
        self.__register_thread_lock.acquire(blocking=True, timeout=-1)
        if threading.get_ident() in self.__thread_process_dict.keys() and \
                self.__thread_process_dict[threading.get_ident()] == os.getpid():
            self.__register_thread_lock.release()
            return
        cl: ConfigLoader = ConfigLoader.instance
        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'ProcessHandler: I am registering thread {threading.get_ident()} to process {os.getpid()}')
        self.__thread_process_dict[threading.get_ident()] = os.getpid()
        self.__register_thread_lock.release()

    def unregister_thread(self, thread_name: int) -> None:
        """
        :param thread_name: thread.ident to unregister
        :return: None
        """
        self.__register_thread_lock.acquire(blocking=True, timeout=-1)
        if thread_name not in self.__thread_process_dict.keys():
            self.__register_thread_lock.release()
            return
        cl: ConfigLoader = ConfigLoader.instance
        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'ProcessHandler: I am unregistering thread {thread_name} from process {os.getpid()}')
        del self.__thread_process_dict[thread_name]
        self.__register_thread_lock.release()

    def get_thread_name_and_register(self):
        """
        :return: The ident of a thread as well as the id of the process it is running on as string
        """
        self.register_thread()
        search = threading.get_ident()
        return f'Process {self.__thread_process_dict[search]}: Thread {search}'

    def get_thread_handler(self):
        """
        :return: A ThreadHandler belonging to the calling process
        """
        if os.getpid() in self.__process_thread_handler_dict.keys():
            self.__thread_handler_user_number_dict[os.getpid()] += 1
            return self.__process_thread_handler_dict[os.getpid()]

        cl: ConfigLoader = ConfigLoader.instance

        if cl.process_and_thread_handler_debug_output_enabled:
            print(f'ProcessHandler: I am adding a ThreadHandler to Process {os.getpid()}')

        self.__process_thread_handler_dict[os.getpid()] = ProcessOrThreadHandler(
            is_process_handler=False, process_name=os.getpid(),
            acquire_data_generation_lock=self.acquire_data_generation_lock,
            release_data_generation_lock=self.release_data_generation_lock, register_thread=self.register_thread)
        self.__thread_handler_user_number_dict[os.getpid()] = 1
        return self.__process_thread_handler_dict[os.getpid()]

    def release_cores(self) -> None:
        """
        Registers the cores that were occupied by the calling process as free
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance

        # If no cores were assigned, it is allowed to call this method nevertheless (with no effect)
        if os.getpid() not in self.__thread_handler_user_number_dict.keys():
            return

        self.__thread_handler_user_number_dict[os.getpid()] -= 1

        if self.__thread_handler_user_number_dict[os.getpid()] == 0:
            if cl.process_and_thread_handler_debug_output_enabled:
                print(f'ProcessHandler: I am removing the ThreadHandler from Process {os.getpid()}')
            del self.__process_thread_handler_dict[os.getpid()]
            del self.__thread_handler_user_number_dict[os.getpid()]

    def execute_threads(self, threads: Set[threading.Thread]) -> None:
        """
        Note: Threads are not registered in this method, as this would only be possible after the thread has been
        started.
        :param threads: A Set of threads to execute and then unregister
        :return: None
        """
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            self.unregister_thread(t.ident)
