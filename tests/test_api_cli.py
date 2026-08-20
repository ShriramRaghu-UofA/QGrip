import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from qgrip.api import create_app
from qgrip.cli import main
from tests.helpers import write_profile


class AdapterTests(unittest.TestCase):
    def test_openapi_and_token_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(write_profile(Path(directory)), token="secret")
            with TestClient(app) as client:
                self.assertEqual(client.get("/openapi.json").status_code, 200)
                self.assertEqual(client.get("/api/v1/bootstrap").status_code, 401)
                self.assertEqual(
                    client.get(
                        "/api/v1/bootstrap", headers={"X-QGrip-Token": "secret"}
                    ).status_code,
                    200,
                )
                self.assertEqual(client.get("/api/v1/artifacts").status_code, 401)
                response = client.get("/api/v1/artifacts", headers={"X-QGrip-Token": "secret"})
                self.assertEqual(response.status_code, 200)

    def test_cli_help_entry_points(self) -> None:
        with self.assertRaises(SystemExit) as exit_context:
            main(["--help"])
        self.assertEqual(exit_context.exception.code, 0)

    def test_infer_defaults_to_live_with_an_explicit_once_mode(self) -> None:
        from qgrip.cli import build_parser

        live = build_parser().parse_args(["infer", "model.pt", "--profile", "profile.json"])
        once = build_parser().parse_args(
            ["infer", "model.pt", "--profile", "profile.json", "--once"]
        )
        self.assertFalse(live.once)
        self.assertTrue(once.once)
