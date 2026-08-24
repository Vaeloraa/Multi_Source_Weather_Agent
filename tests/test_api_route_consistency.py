from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ApiRouteConsistencyTests(unittest.TestCase):
    def test_frontend_only_calls_registered_server_routes(self) -> None:
        app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        server = (
            ROOT / "sevir_world" / "api" / "server.py"
        ).read_text(encoding="utf-8")
        frontend_bases = set(
            re.findall(r"fetch\((?:`|\")(/api/[A-Za-z0-9_-]+)", app)
        )
        self.assertTrue(frontend_bases)
        missing = sorted(route for route in frontend_bases if route not in server)
        self.assertEqual(missing, [], f"front-end routes missing on server: {missing}")
        self.assertNotIn("/api/render-jobs/", app)
        self.assertFalse((ROOT / "sevir_world" / "api" / "render_jobs.py").exists())
        self.assertFalse((ROOT / "sevir_world" / "api" / "render_jobs_v2.py").exists())


if __name__ == "__main__":
    unittest.main()
