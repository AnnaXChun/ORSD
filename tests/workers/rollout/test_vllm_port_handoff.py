"""Exercise the real run_server startup method without importing Ray/vLLM/CUDA."""
import ast
import asyncio
import inspect
import logging
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ReachedEngine(Exception):
    pass


class PortHandoffTests(unittest.TestCase):
    def test_reserved_ports_released_before_engine_start(self):
        tree = ast.parse((ROOT / 'verl/workers/rollout/vllm_rollout/vllm_async_server.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'vLLMHttpServer')
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_server')
        for arg in method.args.args:
            arg.annotation = None
        sockets = []
        try:
            for _ in range(2):
                reservation = socket.socket()
                reservation.bind(('127.0.0.1', 0))
                sockets.append(reservation)
            ports = [s.getsockname()[1] for s in sockets]
            actor = SimpleNamespace(_dp_rpc_sock=sockets[0], _dp_master_sock=sockets[1],
                                    _dp_rpc_port=ports[0], _dp_master_port=ports[1],
                                    _master_port=12345, replica_rank=0)
            config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_master_port=None))

            def create_engine_config(**kwargs):
                # Reservations must still protect the ports while configuring.
                self.assertTrue(all(s.fileno() >= 0 for s in sockets))
                return config

            def start_engine(vllm_config, usage_context, **kwargs):
                self.assertEqual(vllm_config.parallel_config.data_parallel_master_port, ports[1])
                self.assertIsNone(actor._dp_rpc_sock)
                self.assertIsNone(actor._dp_master_sock)
                # Simulate the child's real bind: both must now succeed.
                for port in ports:
                    with socket.socket() as child:
                        child.bind(('127.0.0.1', port))
                        child.listen(1)
                raise ReachedEngine()

            namespace = {'inspect': inspect, 'logger': logging.getLogger(__name__),
                         'UsageContext': SimpleNamespace(OPENAI_API_SERVER='test'),
                         'AsyncEngineArgs': SimpleNamespace(from_cli_args=lambda args: SimpleNamespace(create_engine_config=create_engine_config)),
                         'AsyncLLM': SimpleNamespace(from_vllm_config=start_engine)}
            exec(compile(ast.Module(body=[method], type_ignores=[]), '<real-run-server>', 'exec'), namespace)
            with self.assertRaises(ReachedEngine):
                asyncio.run(namespace['run_server'](actor, SimpleNamespace()))
        finally:
            for reservation in sockets:
                reservation.close()


if __name__ == '__main__':
    unittest.main()
