import unittest

from qgrip.rpc import RESPONSE, MessagePackRpcClient, _Pending


class RpcTests(unittest.TestCase):
    def test_response_dispatches_by_sequence(self) -> None:
        client = MessagePackRpcClient(timeout=0.01)
        pending = _Pending()
        client._pending[7] = pending
        client._handle([RESPONSE, 7, None, {"ok": True}])
        self.assertTrue(pending.event.is_set())
        self.assertEqual(pending.result, {"ok": True})

    def test_bad_frames_are_ignored(self) -> None:
        client = MessagePackRpcClient()
        client._handle([99])
        self.assertEqual(client._pending, {})
